"""Render each sim camera to a PNG, for calibrating the real cameras against.

This is the whole handoff from sim to deploy calibration. Run it here (needs the simulator),
then fit the real cameras against the PNGs with rl/deploy/utils/calibrate_camera.py (needs only
OpenCV, so it also runs on the robot host).

    uv run python examples/dump_reference_views.py
    uv run python examples/dump_reference_views.py --out ../../rl/deploy/utils/reference_views

Domain randomization is OFF and the objects are hidden by default: what you are aligning is the
rig's fixed geometry -- frame edges, panel corners, the arm base -- not a particular episode.
Pass --show-objects if you want the cube and bin in frame as a sanity check.
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401
import soframe_rl_maniskill.envs  # noqa: F401
from mani_skill.utils.structs.pose import Pose
from soframe_rl_maniskill import config

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", default="reference_views",
                    help="directory for the PNGs (default: rl/environments/maniskill/reference_views)")
parser.add_argument("--size", type=int, default=config.SENSOR_RESOLUTION,
                    help="render size; the default matches what the policy trains on")
parser.add_argument("--rail", type=float, default=0.5,
                    help="rail position as a fraction of travel, 0..1. The overhead camera is "
                         "static so this only moves the arm and its wrist camera.")
parser.add_argument("--show-objects", action="store_true",
                    help="leave the cube and bin in frame instead of parking them out of view")
parser.add_argument("--overhead-fov", type=float, default=None,
                    help="override the overhead FOV in degrees, to re-reference after changing it")
parser.add_argument("--wrist-fov", type=float, default=None,
                    help="override the wrist FOV in degrees")
args = parser.parse_args()

dr_cfg = {"visual_fidelity": "raster"}
if args.overhead_fov is not None:
    dr_cfg["overhead_camera_fov"] = np.deg2rad(args.overhead_fov)
if args.wrist_fov is not None:
    dr_cfg["wrist_camera_fov"] = np.deg2rad(args.wrist_fov)

env = gym.make(
    "SOFramePickPlaceBin-v1",
    num_envs=1,
    obs_mode="rgb",
    sim_backend="cpu",              # single env, no training: cpu keeps this runnable anywhere
    domain_randomization=False,     # reference = the nominal rig, not a randomized episode
    domain_randomization_config=dr_cfg,
    sensor_configs=dict(width=args.size, height=args.size),
)
env.reset(seed=0)
u = env.unwrapped

# Put the rail where asked; the arm (and wrist camera) move with it.
qpos = u.agent.robot.get_qpos().clone()
rail_idx = u.agent.joint_names.index("dof_slider")
lo, hi = u.agent.robot.get_qlimits()[0, rail_idx].cpu().numpy()
qpos[0, rail_idx] = float(lo + float(np.clip(args.rail, 0.0, 1.0)) * (hi - lo))
u.agent.robot.set_qpos(qpos)

if not args.show_objects:
    # Park them far away rather than hiding them, so lighting and shadows stay honest.
    u.item.set_pose(Pose.create_from_pq(torch.tensor([[5.0, 5.0, 0.5]])))
    u.bin.set_pose(Pose.create_from_pq(torch.tensor([[6.0, 6.0, 0.5]])))

u._update_camera_poses()
u.scene.update_render()
obs = u.get_obs()

out_dir = pathlib.Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
for cam in ("overhead_camera", "wrist_camera"):
    rgb = obs["sensor_data"][cam]["rgb"][0].cpu().numpy()
    path = out_dir / f"{cam}.png"
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    fov = getattr(u.domain_randomization_config, f"{cam.split('_')[0]}_camera_fov")
    print(f"wrote {path}  ({args.size}x{args.size}, fov {np.rad2deg(fov):.1f} deg)")

print()
print("Now fit each real camera against these, from rl/deploy/:")
print(f"  uv run python utils/calibrate_camera.py utils/captures/real_overhead_camera.png \\")
print(f"      --reference {out_dir}/overhead_camera.png --camera overhead")
print(f"  uv run python utils/calibrate_camera.py utils/captures/real_arm_camera.png \\")
print(f"      --reference {out_dir}/wrist_camera.png --camera arm")
env.close()
