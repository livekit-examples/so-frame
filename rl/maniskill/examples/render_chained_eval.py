"""Render a trained checkpoint as one continuous video: episodes play back-to-back with
no gaps, each ending (and a fresh one starting) on success, failure/timeout, or a fixed
per-episode step cap -- whichever comes first. Single env, wrist + overhead cameras only
(no third-person view, no multi-env tiling).

Run from rl/maniskill/:
    uv run python examples/render_chained_eval.py \
        --checkpoint checkpoints/model_best.pt --num_episodes 10 --out /tmp/chained.mp4

Add --realism_mode for ray-traced shading (see examples/render_realistic.py); this drops
to the cpu sim backend since ray tracing isn't supported on the gpu-parallelized sensor
camera path, which is fine for a single-env render.
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
parser.add_argument("--episode_cap", type=int, default=200, help="force a reset after this many steps even without success/failure")
parser.add_argument("--render_size", type=int, default=512)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--realism_mode", action="store_true")
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default="/tmp/chained_eval.mp4")
args = parser.parse_args()

env_kwargs = dict(
    obs_mode="rgb",
    render_mode="sensors",
    num_envs=1,
    domain_randomization=args.domain_randomization,
    sensor_configs=dict(width=args.render_size, height=args.render_size),
)
if args.realism_mode:
    env_kwargs["domain_randomization_config"] = dict(realism_mode=True)
    env_kwargs["sensor_configs"]["shader_pack"] = "rt-fast"
    env_kwargs["sim_backend"] = "cpu"
else:
    env_kwargs["sim_backend"] = "gpu"

env = gym.make(args.env_id, **env_kwargs)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

device = torch.device("cuda" if torch.cuda.is_available() and not args.realism_mode else "cpu")

obs, info = env.reset(seed=args.seed)
deploy_agent = DeployAgent(env, obs, target_image_size=16, device=device).to(device)
deploy_agent.load_checkpoint(args.checkpoint)
deploy_agent.eval()

writer = cv2.VideoWriter(
    args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * args.render_size, args.render_size),
)

episode = 0
step_in_episode = 0
successes = 0
print(f"Rendering {args.num_episodes} chained episodes (cap {args.episode_cap} steps each) to {args.out}")

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

    success = bool(info.get("success", torch.zeros(1, dtype=torch.bool))[0])
    done = bool(terminated[0]) or bool(truncated[0]) or step_in_episode >= args.episode_cap
    if done:
        episode += 1
        successes += int(success)
        print(f"episode {episode}/{args.num_episodes}: {'success' if success else 'no success'} after {step_in_episode} steps")
        obs, info = env.reset(seed=args.seed + episode)
        step_in_episode = 0

writer.release()
env.close()
print(f"OK: wrote {args.out} ({successes}/{args.num_episodes} episodes succeeded)")
