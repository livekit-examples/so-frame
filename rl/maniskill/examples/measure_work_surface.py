"""Measure the lightbox bottom panel's actual world-space top surface height and XY
footprint, directly from the loaded articulation + the STL mesh bounds -- rather than
guessing from rl/mjlab's MJCF-frame numbers, which may use a different root convention.

`part_1_1` (the panel) has no <collision> in the URDF, only <visual> -- same issue rl/mjlab
found and fixed with an invisible collision pad. This script measures where that pad needs
to go for OUR loader/robot pose so pick_place.py's WORK_SURFACE_Z/SPAWN_CENTER can be
corrected precisely instead of guessed.

Run from rl/maniskill/:
    uv run python examples/measure_work_surface.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trimesh
import gymnasium as gym

import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401

from soframe_rl_maniskill.robot.so101_on_frame import _REPO_ROOT, SO101_ON_FRAME_URDF

env = gym.make(
    "SOFramePickPlaceBin-v1",
    num_envs=1,
    obs_mode="state",
    render_mode=None,
    domain_randomization=False,
    sim_backend="gpu",
)
env.reset(seed=0)
agent = env.unwrapped.agent

link = agent.robot.links_map["part_1_1"]
link_pose = link.pose  # world pose of the link origin (env 0)
p = link_pose.p[0].cpu().numpy()
q = link_pose.q[0].cpu().numpy()
print("part_1_1 link world pose: p =", p, " q =", q)

# The URDF's <visual><origin> offset for this link's mesh (see so101_on_frame.urdf).
visual_origin_xyz = np.array([0.52, 0.255, -0.002])
visual_origin_rpy = np.array([0.0, -0.0, 0.0])  # identity rotation

mesh_path = SO101_ON_FRAME_URDF.parent / "components" / "base_frame" / "meshes" / "Part_1_1.stl"
mesh = trimesh.load_mesh(str(mesh_path))
bounds = mesh.bounds  # (2, 3): [min_xyz, max_xyz] in mesh-local coordinates
print("Part_1_1.stl local bounds (min, max):", bounds)

# link pose is identity rotation for this rig (fixed base, no rail/arm motion affects it).
from scipy.spatial.transform import Rotation
link_rot = Rotation.from_quat([q[1], q[2], q[3], q[0]])  # sapien q is (w,x,y,z)

# mesh-local bounds -> visual-frame (identity rot for this mesh) -> link-local -> world.
corners_mesh = np.array([[bounds[i, 0], bounds[j, 1], bounds[k, 2]]
                          for i in (0, 1) for j in (0, 1) for k in (0, 1)])
corners_link_local = corners_mesh + visual_origin_xyz
corners_world = link_rot.apply(corners_link_local) + p

print()
print("Panel corners in WORLD frame:")
print(corners_world)
print()
print("World-frame footprint: x in [{:.4f}, {:.4f}], y in [{:.4f}, {:.4f}], top z = {:.4f}".format(
    corners_world[:, 0].min(), corners_world[:, 0].max(),
    corners_world[:, 1].min(), corners_world[:, 1].max(),
    corners_world[:, 2].max(),
))
print("(bottom z = {:.4f})".format(corners_world[:, 2].min()))

env.close()
