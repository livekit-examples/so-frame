"""Headless sanity check: save PNG frames (wrist cam + third-person) to disk instead of
cv2.imshow (no display needed over SSH). Also prints cube/bin/tcp positions each step so
reach can be checked numerically, not just by eye.

Run from rl/maniskill/:
    uv run python examples/render_frames.py --steps 90 --out /tmp/frames
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

import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--out", type=str, default="/tmp/frames")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--domain_randomization", action="store_true")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

env = gym.make(
    "SOFramePickPlaceBin-v1",
    num_envs=args.num_envs,
    obs_mode="rgb+segmentation+state",
    render_mode="rgb_array",
    domain_randomization=args.domain_randomization,
    sim_backend="gpu",
    sensor_configs=dict(width=256, height=256),
    human_render_camera_configs=dict(width=512, height=512),
)
env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

obs, info = env.reset(seed=0)
unwrapped = env.unwrapped

print("item pose (env0):", unwrapped.item.pose.p[0].cpu().numpy())
print("bin pose (env0):", unwrapped.bin.pose.p[0].cpu().numpy())
print("tcp pos (env0):", unwrapped.agent.tcp_pos[0].cpu().numpy())
print("dist tcp->item (env0):", torch.linalg.norm(unwrapped.agent.tcp_pos[0] - unwrapped.item.pose.p[0]).item())
print("dist item->bin (env0):", torch.linalg.norm(unwrapped.item.pose.p[0, :2] - unwrapped.bin.pose.p[0, :2]).item())

action_shape = env.action_space.shape
for step in range(args.steps):
    action = np.zeros(action_shape)
    # crude scripted probe: drive the slider (last-but-one dim isn't it; slider is index 0)
    # back and forth, and cycle the gripper, just to get varied frames -- not a real policy.
    action[..., 0] = 0.5 if (step // 15) % 2 == 0 else -0.5
    action[..., -1] = 1 if step < 30 else -1

    obs, reward, terminated, truncated, info = env.step(action)

    if step % 15 == 0:
        render_rgb = env.render()[0].cpu().numpy().astype(np.uint8)
        # DualCameraEnv orders sensors [wrist_camera, overhead_camera]; FlattenRGBDObservationWrapper
        # concatenates their RGB along the channel axis, so channels 0:3 are the wrist cam, 3:6 overhead.
        wrist_rgb = obs["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
        overhead_rgb = obs["rgb"][0, ..., 3:6].cpu().numpy().astype(np.uint8)
        cv2.imwrite(f"{args.out}/step{step:03d}_third_person.png", cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{args.out}/step{step:03d}_wrist_cam.png", cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{args.out}/step{step:03d}_overhead_cam.png", cv2.cvtColor(overhead_rgb, cv2.COLOR_RGB2BGR))
        print(f"step {step}: item={unwrapped.item.pose.p[0].cpu().numpy()} "
              f"tcp={unwrapped.agent.tcp_pos[0].cpu().numpy()} reward={float(reward[0]):.3f}")

env.close()
print(f"OK: wrote frames to {args.out}")
