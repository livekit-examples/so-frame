"""One-off high-fidelity render: ray-traced shading + softbox lighting, just to look at the sim.

Never used for training, which keeps the fast rasterizer (see
``RandomizationConfig.visual_fidelity`` in ``envs/base_random_env.py``). ``--shader rt`` is full path
tracing, ``rt-fast`` a quicker denoised approximation, ``default`` SAPIEN's rasterizer (no real
shadows/reflections/GI).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import cv2
import numpy as np
import gymnasium as gym

import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shader", choices=["default", "rt", "rt-fast"], default="rt-fast",
    help="rt-fast: fast path-traced (recommended). rt: slower/highest quality. "
         "default: SAPIEN's rasterizer, no ray tracing.",
)
parser.add_argument(
    "--steps", type=int, default=20,
    help="Steps to run before capturing, so the arm settles into a natural pose.",
)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
parser.add_argument("--out", type=str, default="/tmp/realistic.png")
parser.add_argument("--domain_randomization", action="store_true")
args = parser.parse_args()

env = gym.make(
    "SOFramePickPlaceBin-v1",
    num_envs=1,
    obs_mode="state",
    render_mode="rgb_array",
    domain_randomization=args.domain_randomization,
    domain_randomization_config=dict(visual_fidelity="raytraced"),
    sim_backend="gpu",
    human_render_camera_configs=dict(shader_pack=args.shader, width=args.width, height=args.height),
)

obs, info = env.reset(seed=0)
unwrapped = env.unwrapped

action_shape = env.action_space.shape
for step in range(args.steps):
    action = np.zeros(action_shape)
    action[..., -1] = 1 if step < args.steps // 2 else -1  # open then close the gripper
    obs, reward, terminated, truncated, info = env.step(action)

render_rgb = env.render()[0].cpu().numpy().astype(np.uint8)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
cv2.imwrite(args.out, cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR))
env.close()
print(f"OK: wrote {args.shader}-shaded frame to {args.out}")
