"""Sanity-check the SOFramePickPlaceBin-v1 scene, cameras, and domain randomization
before spending GPU time on training. Steps a batch of environments through a scripted
probe (rail sliding back and forth, gripper open then close -- not a policy) and shows
each env's wrist camera | overhead camera | third-person render side by side. Also prints
item/bin/tcp positions so reach can be checked numerically, not just by eye.

By default opens a cv2 window; pass --headless to write PNG frames to --out instead
(no display needed over SSH).

Run from rl/environments/maniskill/:
    uv run python examples/visualize_sim.py
    uv run python examples/visualize_sim.py --headless --out /tmp/frames
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np
import cv2
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.visualization.misc import tile_images

import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--reset_interval", type=int, default=20)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--domain_randomization", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--headless", action="store_true",
                    help="write PNG frames to --out instead of opening a window")
parser.add_argument("--out", type=str, default="/tmp/frames")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

env = gym.make(
    "SOFramePickPlaceBin-v1",
    num_envs=args.num_envs,
    obs_mode="rgb+segmentation",
    render_mode="rgb_array",
    domain_randomization=args.domain_randomization,
    sim_backend="gpu",
    sensor_configs=dict(width=args.image_size, height=args.image_size),
    human_render_camera_configs=dict(width=args.image_size, height=args.image_size),
)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

obs, info = env.reset(seed=args.seed)
unwrapped = env.unwrapped


def print_positions(step):
    item = unwrapped.item.pose.p[0]
    bin_p = unwrapped.bin.pose.p[0]
    tcp = unwrapped.agent.tcp_pos[0]
    print(f"step {step:3d}: item={item.cpu().numpy().round(3)} bin={bin_p.cpu().numpy().round(3)} "
          f"tcp={tcp.cpu().numpy().round(3)} dist(tcp,item)={torch.linalg.norm(tcp - item).item():.3f}")


if args.headless:
    os.makedirs(args.out, exist_ok=True)
nrows = int(np.sqrt(args.num_envs))
action_shape = env.action_space.shape
print_positions(0)

for step in range(args.steps):
    action = np.zeros(action_shape)
    action[..., 0] = 0.5 if (step // 15) % 2 == 0 else -0.5  # dof_slider back and forth
    action[..., -1] = 1 if step < args.steps // 3 else -1    # gripper open then close
    obs, reward, terminated, truncated, info = env.step(action)

    # Per env: wrist cam (channels 0:3) | overhead cam (3:6) | third-person render.
    render_rgb = env.render().to(torch.uint8)  # (N, H, W, 3)
    frame = torch.cat([obs["rgb"][..., :3], obs["rgb"][..., 3:6], render_rgb], dim=2)
    tiled = tile_images(frame, nrows=nrows).cpu().numpy().astype(np.uint8)
    tiled = cv2.cvtColor(tiled, cv2.COLOR_RGB2BGR)

    if args.headless:
        if step % 15 == 0:
            cv2.imwrite(f"{args.out}/step{step:03d}.png", tiled)
            print_positions(step)
    else:
        cv2.imshow("wrist | overhead | third-person", tiled)
        cv2.waitKey(30)

    done = (terminated | truncated).any()
    if (step + 1) % args.reset_interval == 0 or done:
        obs, info = env.reset()
        print_positions(step + 1)

env.close()
if args.headless:
    print(f"OK: wrote frames to {args.out}")
else:
    cv2.destroyAllWindows()
