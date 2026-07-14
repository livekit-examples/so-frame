"""Validate so101_on_frame.usd's physics (rigid bodies, joints, collision) actually
simulates correctly in Isaac Sim -- a real drop test, not just schema-presence checks.

Run on the Isaac Lab box (after `source ~/isaac/activate_isaac.sh`):
    python3 simulation/usd/isaac_validate.py
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default="simulation/usd/mjcf_import/so101_on_frame/so101_on_frame.usda")
parser.add_argument("--steps", type=int, default=300)
args = parser.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane(z_position=-1.0)  # safety catch-all, well below the work surface

robot_path = "/World/so101_on_frame"
add_reference_to_stage(usd_path=args.usd, prim_path=robot_path)

# Test cube dropped just above the (fixed) floor panel's known surface, matching the same
# XY used for the URDF/MJCF drop tests (see simulation/urdf/README.md, examples/measure_work_surface.py).
test_cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/test_cube",
        name="test_cube",
        position=np.array([-0.245, -0.5, 0.05]),
        scale=np.array([0.025, 0.025, 0.025]),
        mass=0.03,
    )
)

world.reset()

print("Stepping physics...")
for step in range(args.steps):
    world.step(render=False)
    if step % 30 == 0:
        pos, _ = test_cube.get_world_pose()
        vel = test_cube.get_linear_velocity()
        print(f"step {step}: cube z={pos[2]:.5f}, vz={vel[2]:.5f}")

pos, _ = test_cube.get_world_pose()
vel = test_cube.get_linear_velocity()
print(f"FINAL: cube z={pos[2]:.5f}, vz={vel[2]:.5f}")
print(f"Expected resting z (WORK_SURFACE_Z=0 + half_size=0.0125): ~0.0125")

simulation_app.close()
