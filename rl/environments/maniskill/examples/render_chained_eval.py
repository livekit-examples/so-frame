"""Render a trained checkpoint as one continuous video: episodes play back-to-back with no gaps.

A successful episode cuts to a new one right away; one that has not succeeded runs for
--fail_seconds first, so failures get consistent screen time. Only env 0 is filmed, wrist +
overhead cameras, no third-person view.

--visual_fidelity picks the shading: "raster" (the training default, gpu sim backend), "flat"
(shadowless, fastest) or "raytraced" (rt-fast, forced to the cpu sim backend since ray tracing is
not supported on the gpu-parallelized sensor camera path).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import cv2
import numpy as np
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401
from soframe_policy import checkpoint as ckpt_io

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--sim_backend", choices=["gpu", "cpu"], default="gpu",
                    help="'cpu' supports a single env and lets this run without CUDA")
parser.add_argument("--env_id", type=str, default="SOFramePickPlaceBin-v1")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--fail_seconds", type=float, default=5.0, help="how long an episode that hasn't succeeded runs before cutting to a fresh one")
parser.add_argument("--render_size", type=int, default=512)
# 10 fps = the 10 Hz control rate, so the video plays at real time.
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--visual_fidelity", type=str, default="raster", choices=["flat", "raster", "raytraced"])
parser.add_argument("--overlay", action="store_true", help="composite the greenscreen background overlay. OFF matches training: the background is black by construction now that the ground plane sits beyond config.SENSOR_FAR, so the overlay (and the segmentation render that drives it) is no longer part of the training obs. Only pass this to composite a real background photo")
parser.add_argument("--sim_envs", type=int, default=16, help="number of parallel sim envs; only env 0 is filmed. Policies trained in batched GPU physics measurably degrade when rolled out in a lone env (contact dynamics resolve differently in a single-scene solver), so render with a training-sized batch. Forced to 1 for raytraced (cpu backend).")
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="/tmp/chained_eval.mp4")
args = parser.parse_args()

fail_step_cap = int(args.fail_seconds * args.fps)

# Camera jitter reads as shake in a video, so keep the cameras steady even under
# --domain_randomization.
steady_cameras = dict(
    wrist_camera_pos_noise=(0.0, 0.0, 0.0), wrist_camera_rot_noise=(0.0, 0.0, 0.0),
    wrist_camera_fov_noise=0.0,
    overhead_camera_pos_noise=(0.0, 0.0, 0.0), overhead_camera_rot_noise=(0.0, 0.0, 0.0),
    overhead_camera_fov_noise=0.0,
)

env_kwargs = dict(
    # The greenscreen overlay only engages when segmentation is in the obs mode (base_random_env
    # skips it otherwise). Training is plain "rgb" with a black background from the far-plane cull,
    # so asking for the overlay is what would differ from the training obs.
    obs_mode="rgb+segmentation" if args.overlay else "rgb",
    render_mode="sensors",
    num_envs=1 if args.visual_fidelity == "raytraced" else args.sim_envs,
    domain_randomization=args.domain_randomization,
    domain_randomization_config=dict(
        visual_fidelity=args.visual_fidelity,
        apply_overlay=args.overlay,
        **steady_cameras,
    ),
    sensor_configs=dict(width=args.render_size, height=args.render_size),
)
if args.visual_fidelity == "raytraced":
    env_kwargs["sensor_configs"]["shader_pack"] = "rt-fast"
    env_kwargs["sim_backend"] = "cpu"
else:
    env_kwargs["sim_backend"] = args.sim_backend

env = gym.make(args.env_id, **env_kwargs)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

# Policy inference stays on cuda even with raytraced fidelity (only the sim drops to cpu there);
# obs are moved to this device explicitly each step.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs, info = env.reset(seed=args.seed)
# The checkpoint states its own encoder, resolution, camera count and proprio layout, so there is
# nothing to pass and nothing to keep in sync with the run that produced it.
encoder, actor, meta = ckpt_io.load(args.checkpoint, device=device)
print(f"loaded {meta['kind']} @ res {meta['res']}, {meta['num_cams']} cameras, "
      f"step {meta['global_step']}")
print(f"proprio {meta['proprio'].describe()}")

writer = cv2.VideoWriter(
    args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * args.render_size, args.render_size),
)

episode = 0
step_in_episode = 0
successes = 0
print(f"Rendering {args.num_episodes} chained episodes to {args.out} "
      f"(cut on success, else after {args.fail_seconds}s / {fail_step_cap} steps)")

while episode < args.num_episodes:
    with torch.no_grad():
        # preprocess() is the same transform the training obs pipeline applied.
        features = encoder(encoder.preprocess(obs["rgb"].to(device)))
        action = actor.get_eval_action(features, obs["state"].to(device))
    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
    step_in_episode += 1

    # obs["rgb"] is the flattened (H, W, 3*num_cams) tensor from FlattenRGBDObservationWrapper;
    # split back into per-camera images for display.
    rgb = obs["rgb"][0].cpu().numpy().astype(np.uint8)
    num_channels = rgb.shape[-1]
    cams = [rgb[..., c:c + 3] for c in range(0, num_channels, 3)]
    frame = np.concatenate(cams, axis=1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # Cut right away on success (ManiSkill maps info["success"] straight to `terminated`,
    # so this is the same signal); otherwise cut once the episode has had its runway.
    success = bool(info.get("success", torch.zeros(1, dtype=torch.bool))[0])
    done = success or step_in_episode >= fail_step_cap
    if done:
        episode += 1
        successes += int(success)
        print(f"episode {episode}/{args.num_episodes}: {'success' if success else 'no success'} after {step_in_episode} steps")
        obs, info = env.reset(seed=args.seed + episode)
        step_in_episode = 0

writer.release()
env.close()
print(f"OK: wrote {args.out} ({successes}/{args.num_episodes} episodes succeeded)")
