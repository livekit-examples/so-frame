"""Environment construction and the per-encoder observation pipeline.

This is the only place the two encoders differ structurally, and the difference is entirely in
the wrapper stack:

    squint      render at render_size -> downsample to res -> jitter -> sensor aug
    dino_patch  render at render_size -> jitter -> sensor aug -> tokenize at res

Ordering matters in both. The augmentations need images, so they come before tokenisation;
the tokeniser is always last, because what it emits are features, not pixels.
"""

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from soframe_policy import ProprioSpec

from .. import wrappers


@dataclass
class EnvBundle:
    """The envs plus everything read off them that the networks and checkpoint need."""

    envs: Any
    eval_envs: Any
    max_episode_steps: int
    eval_output_dir: str | None
    env_kwargs: dict
    eval_env_kwargs: dict
    num_cams: int
    n_act: int
    n_state: int
    proprio: ProprioSpec
    """Measured field layout of the actor's state vector; goes into the checkpoint so deploy
    assembles its proprio in the trained order rather than assuming a width."""


def env_kwargs_from_args(args):
    """Build the (train, eval) gym.make kwargs, including the domain-randomization overrides."""
    obs_mode = args.obs_mode
    common = dict(
        obs_mode=obs_mode,
        sim_backend=args.sim_backend,
        sensor_configs=dict(width=args.render_size, height=args.render_size),
    )
    env_kwargs = dict(common, render_mode=args.render_mode)
    # "sensors" = wrist + overhead only (no third-person camera) for eval_envs recordings
    eval_env_kwargs = dict(
        common,
        render_mode="sensors",
        human_render_camera_configs=dict(
            shader_pack="default", width=args.render_size, height=args.render_size
        ),
    )

    for kw in (env_kwargs, eval_env_kwargs):
        if args.control_mode is not None:
            kw["control_mode"] = args.control_mode
        if args.env_domain_randomization:
            kw["domain_randomization"] = True

        dr_cfg = {}
        if args.randomize_colors:
            dr_cfg.update(randomize_item_color=True, randomize_bin_color=True)
        if args.overhead_camera_fov is not None:
            dr_cfg["overhead_camera_fov"] = np.deg2rad(args.overhead_camera_fov)
        if args.wrist_camera_fov is not None:
            dr_cfg["wrist_camera_fov"] = np.deg2rad(args.wrist_camera_fov)
        if args.overhead_camera_pos_offset is not None:
            dr_cfg["overhead_camera_pos_offset"] = list(args.overhead_camera_pos_offset)
        if args.overhead_camera_rot_offset is not None:
            dr_cfg["overhead_camera_rot_offset"] = [np.deg2rad(v) for v in args.overhead_camera_rot_offset]
        dr_cfg["binary_gripper"] = args.binary_gripper
        if args.visual_fidelity in ("flat", "raster"):
            # pass the flag so flat can override the raster default; the minimal sensor
            # shader renders shadows/textures fine and avoids the default shader's OOM past ~384 envs
            dr_cfg["visual_fidelity"] = args.visual_fidelity
        if dr_cfg:
            kw["domain_randomization_config"] = dr_cfg

    if args.visual_fidelity == "raytraced":
        # ray-traced sensor cameras exist only on the cpu backend (single env), so both
        # pools drop to cpu with rt-fast sensors; training is slow and --num_updates must scale down
        for kw in (env_kwargs, eval_env_kwargs):
            kw["sensor_configs"]["shader_pack"] = "rt-fast"
            kw["domain_randomization_config"] = dict(
                kw.get("domain_randomization_config", {}), visual_fidelity="raytraced"
            )
            kw["sim_backend"] = "cpu"
        if not args.evaluate:
            assert args.num_envs == 1 and args.num_eval_envs == 1, (
                "raytraced training runs on the cpu backend, which supports a single "
                "env: pass --num_envs=1 --num_eval_envs=1 (and a small --num_updates)"
            )

    return env_kwargs, eval_env_kwargs


def apply_obs_pipeline(env, args, device):
    """Wrap a flattened-RGB env with the observation pipeline for ``args.encoder``."""
    if args.encoder == "squint":
        if args.render_size != args.res:
            env = wrappers.DownsampleObsWrapper(env, target_size=args.res)
        if args.apply_jitter:
            env = wrappers.ColorJitterWrapper(env)
        if args.sensor_aug:
            # after ColorJitter; applied to eval too so best-ckpt selection favors robustness
            env = wrappers.SensorAugWrapper(env)
    elif args.encoder == "dino_patch":
        if args.apply_jitter:
            env = wrappers.ColorJitterWrapper(env)
        if args.sensor_aug:
            env = wrappers.SensorAugWrapper(env)
        # LAST: everything above needs pixels, and this emits features.
        env = wrappers.DinoTokenWrapper(env, res=args.res, device=device)
    else:
        raise ValueError(f"no observation pipeline defined for encoder {args.encoder!r}")
    return env


def make_envs(args, run_name, device):
    """Build the train and eval vector envs, fully wrapped.

    Returns an EnvBundle: the two envs plus the shapes and paths the training loop needs.
    """
    env_kwargs, eval_env_kwargs = env_kwargs_from_args(args)

    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq, **eval_env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)

    # Measure the actor's proprio layout from the raw env, before flattening collapses it to a
    # bare width. This is the contract deploy has to reproduce field-for-field.
    proprio = ProprioSpec.from_obs_agent(envs.unwrapped._get_obs_agent())

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)

    # Read the camera count here, while the observation is still pixels. After tokenisation the
    # rgb space is (n_tok, EMBED) and the channel count is gone, so this cannot be recovered
    # downstream -- and it must not be assumed, since it sets the encoder's camera embeddings.
    n_channels = envs.unwrapped.single_observation_space['rgb'].shape[-1]
    assert n_channels % 3 == 0, f"expected 3*num_cams RGB channels, got {n_channels}"
    num_cams = n_channels // 3

    envs = apply_obs_pipeline(envs, args, device)
    eval_envs = apply_obs_pipeline(eval_envs, args, device)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    eval_output_dir = None
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"runs/{run_name}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            def save_video_trigger(x):
                return (x // max_episode_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos",
                                 save_trajectory=False, save_video_trigger=save_video_trigger,
                                 max_steps_per_video=max_episode_steps, video_fps=10)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir,
                                  save_trajectory=args.save_trajectory,
                                  save_video=args.capture_video, trajectory_name="trajectory",
                                  max_steps_per_video=max_episode_steps, video_fps=10)

    envs = ManiSkillVectorEnv(envs, args.num_envs,
                              ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs,
                                   ignore_terminations=not args.eval_partial_reset,
                                   record_metrics=True)

    assert isinstance(envs.unwrapped.single_action_space, gym.spaces.Box), \
        "only continuous action space is supported"

    return EnvBundle(
        envs=envs,
        eval_envs=eval_envs,
        max_episode_steps=max_episode_steps,
        eval_output_dir=eval_output_dir,
        env_kwargs=env_kwargs,
        eval_env_kwargs=eval_env_kwargs,
        num_cams=num_cams,
        n_act=math.prod(envs.unwrapped.single_action_space.shape),
        n_state=math.prod(envs.unwrapped.single_observation_space['state'].shape),
        proprio=proprio,
    )
