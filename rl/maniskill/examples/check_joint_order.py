"""Headless sanity check: print the SO-101-on-frame agent's active joint order and the
task's keyframe/qpos alignment. No display needed (unlike visualize_sim.py's cv2.imshow),
so this is safe to run over SSH on a GPU box with no X server.

Run from rl/maniskill/:
    uv run python examples/check_joint_order.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym

import soframe_rl_maniskill.envs  # noqa: F401 (registers agent + task)
import mani_skill.envs  # noqa: F401

from soframe_rl_maniskill.robot.so101_on_frame import SO101OnFrame

env = gym.make(
    "SOFramePickPlaceBin-v1",
    num_envs=4,
    obs_mode="state",
    render_mode=None,
    domain_randomization=False,
    sim_backend="gpu",
)
env.reset(seed=0)

agent = env.unwrapped.agent
joint_names = [j.name for j in agent.robot.active_joints]

print("=" * 70)
print("ACTIVE JOINT ORDER (as loaded by SAPIEN's URDF loader):")
print(joint_names)
print()

expected = ["dof_slider", "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
print("ASSUMED ORDER IN so101_on_frame.py / pick_place.py:")
print(expected)
print()
print("MATCH:", joint_names == expected)
if joint_names != expected:
    print("!!! MISMATCH -- keyframes / qpos-index-based code needs fixing !!!")
    print("Per-joint mapping (assumed_index -> actual_name):")
    for i, name in enumerate(expected):
        actual = joint_names[i] if i < len(joint_names) else "<out of range>"
        print(f"  index {i}: assumed={name!r}  actual={actual!r}  {'OK' if name == actual else 'MISMATCH'}")
print("=" * 70)

# Also check qpos/qlimits shapes and the qpos right after reset (should be close to the
# 'rest' keyframe if the keyframe qpos array lines up with the real joint order).
qpos = agent.robot.get_qpos()
qlimits = agent.robot.get_qlimits()
print("qpos after reset (env 0):", qpos[0].cpu().numpy())
print("rest keyframe qpos (as authored):", SO101OnFrame.keyframes["rest"].qpos)
print("qlimits (env 0):")
for i, name in enumerate(joint_names):
    lo, hi = qlimits[0, i].cpu().numpy()
    print(f"  [{i}] {name:16s} lower={lo:.4f} upper={hi:.4f}")

print()
print("wrist_camera_link pose:", agent.wrist_camera_link.pose.raw_pose[0].cpu().numpy())
print("overhead_camera_link pose:", agent.overhead_camera_link.pose.raw_pose[0].cpu().numpy())
print("tcp_pos:", agent.tcp_pos[0].cpu().numpy())
print("gripper_link pose:", agent.finger1_link.pose.raw_pose[0].cpu().numpy())

env.close()
print("OK: environment created, reset, and inspected without error.")
