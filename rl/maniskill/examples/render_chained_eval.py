"""Render a trained checkpoint as one continuous video: episodes play back-to-back with
no gaps. A successful episode cuts to a new one right away; one that hasn't succeeded
runs for --fail_seconds before cutting, so failed attempts get a consistent amount of
screen time. Single env, wrist + overhead cameras only (no third-person view, no
multi-env tiling).

Run from rl/maniskill/:
    uv run python examples/render_chained_eval.py \
        --checkpoint checkpoints/model_best.pt --num_episodes 10 --out /tmp/chained.mp4

--visual_fidelity picks the shading: "raster" (training default: PBR materials +
softbox-like lighting with a faint shadow, stays on the gpu sim backend), "flat"
(shadowless; what the older model_best.pt checkpoint saw), or "raytraced" (rt-fast;
drops to the cpu sim backend since ray tracing isn't supported on the gpu-parallelized
sensor camera path, which is fine for a single-env render).
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
from train_squint import CNNEncoder, Actor, DeployAgent  # noqa: F401 (Actor used by DeployAgent)

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--env_id", type=str, default="SOFramePickPlaceBin-v1")
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--fail_seconds", type=float, default=5.0, help="how long an episode that hasn't succeeded runs before cutting to a fresh one")
parser.add_argument("--render_size", type=int, default=512)
# 10 fps = the 10 Hz control rate, so the video plays at real time. (It used to default
# to 20, which played every rollout at 2x and made the arm/rail look twice as fast as
# they move in sim -- and on the real robot.)
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--target_image_size", type=int, default=32, help="must match the checkpoint's own image_size (32 is the current default; older checkpoints like slider_pickplace_v7 used 16)")
parser.add_argument("--visual_fidelity", type=str, default="raster", choices=["flat", "raster", "raytraced"])
parser.add_argument("--no_overlay", action="store_true", help="skip the greenscreen background overlay for a showcase render; policies are TRAINED with the overlay (background composited to black), so leaving it on is what matches their training obs and success rates")
parser.add_argument("--sim_envs", type=int, default=16, help="number of parallel sim envs; only env 0 is filmed. Policies trained in batched GPU physics measurably degrade when rolled out in a lone env (contact dynamics resolve differently in a single-scene solver), so render with a training-sized batch. Forced to 1 for raytraced (cpu backend).")
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="/tmp/chained_eval.mp4")
args = parser.parse_args()

fail_step_cap = int(args.fail_seconds * args.fps)

# Camera jitter is part of domain randomization (sim-to-real), but in a video it just
# reads as shake -- keep the cameras steady here even when --domain_randomization is on.
steady_cameras = dict(
    wrist_camera_pos_noise=(0.0, 0.0, 0.0), wrist_camera_rot_noise=(0.0, 0.0, 0.0),
    wrist_camera_fov_noise=0.0,
    overhead_camera_pos_noise=(0.0, 0.0, 0.0), overhead_camera_rot_noise=(0.0, 0.0, 0.0),
    overhead_camera_fov_noise=0.0,
)

env_kwargs = dict(
    # The greenscreen overlay only engages when segmentation is in the obs mode
    # (base_random_env skips it otherwise), and training always ran with it -- plain
    # "rgb" here silently fed the policy backgrounds it never saw in training.
    obs_mode="rgb" if args.no_overlay else "rgb+segmentation",
    render_mode="sensors",
    num_envs=1 if args.visual_fidelity == "raytraced" else args.sim_envs,
    domain_randomization=args.domain_randomization,
    domain_randomization_config=dict(
        visual_fidelity=args.visual_fidelity,
        apply_overlay=not args.no_overlay,
        **steady_cameras,
    ),
    sensor_configs=dict(width=args.render_size, height=args.render_size),
)
if args.visual_fidelity == "raytraced":
    env_kwargs["sensor_configs"]["shader_pack"] = "rt-fast"
    env_kwargs["sim_backend"] = "cpu"
else:
    env_kwargs["sim_backend"] = "gpu"

env = gym.make(args.env_id, **env_kwargs)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

# Policy inference stays on cuda even with raytraced fidelity (only the *sim* drops to cpu
# there); obs are moved to this device explicitly each step.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs, info = env.reset(seed=args.seed)
deploy_agent = DeployAgent(env, obs, target_image_size=args.target_image_size, device=device).to(device)
deploy_agent.load_checkpoint(args.checkpoint)
deploy_agent.eval()

writer = cv2.VideoWriter(
    args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * args.render_size, args.render_size),
)

episode = 0
step_in_episode = 0
successes = 0
print(f"Rendering {args.num_episodes} chained episodes to {args.out} "
      f"(cut on success, else after {args.fail_seconds}s / {fail_step_cap} steps)")

while episode < args.num_episodes:
    obs_gpu = {"rgb": obs["rgb"].to(device), "state": obs["state"].to(device)}
    with torch.no_grad():
        action = deploy_agent.get_action(obs_gpu)
    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
    step_in_episode += 1

    # obs["rgb"] here is the flattened (H, W, 3*num_cams) tensor from
    # FlattenRGBDObservationWrapper -- split back into per-camera images for display.
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
