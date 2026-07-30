"""Programmatic MjSpec assets for the pick-and-place task: a cube and a bin.

Each is its own mjlab entity, kept separate from the robot so the task objects are not
baked into the shared ``simulation/`` model. The cube is a floating free body;
the bin is a fixed-base tray (mjlab auto-wraps it as a mocap body, so it can be
repositioned per-episode by the command but is otherwise immovable and cannot be
knocked over).
"""

from __future__ import annotations

import mujoco

##
# Cube.
##

CUBE_HALF_SIZE = 0.0125  # 2.5 cm cube.
CUBE_MASS = 0.03  # 30 g.


def get_cube_spec(
  half_size: float = CUBE_HALF_SIZE,
  mass: float = CUBE_MASS,
  rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half_size,) * 3,
    mass=mass,
    rgba=rgba,
    # High tangential friction so the gripper can hold it.
    friction=(1.0, 0.02, 0.001),
  )
  return spec


##
# Bin.
#
# Interior footprint is 2*BIN_INNER_HALF square; walls rise to BIN_RIM_HEIGHT above
# the bin's base. The command uses these to decide "cube is in the bin".
##

BIN_INNER_HALF = 0.05  # 10 cm interior width.
BIN_WALL_HALF = 0.004  # 8 mm walls.
BIN_RIM_HEIGHT = 0.05  # wall height above the base.


def get_bin_spec(
  inner_half: float = BIN_INNER_HALF,
  wall_half: float = BIN_WALL_HALF,
  rim_height: float = BIN_RIM_HEIGHT,
  rgba: tuple[float, float, float, float] = (0.25, 0.45, 0.85, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="bin")  # no freejoint -> fixed base (mocap).

  outer = inner_half + wall_half
  wall_kwargs = dict(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    rgba=rgba,
    friction=(1.0, 0.02, 0.001),
  )

  # Base plate (bottom sits on the entity origin plane, z in [0, 2*wall_half]).
  body.add_geom(
    name="bin_base",
    size=(outer, outer, wall_half),
    pos=(0.0, 0.0, wall_half),
    **wall_kwargs,
  )
  # Four walls.
  hz = rim_height / 2.0
  cz = wall_half + hz
  body.add_geom(name="bin_wall_px", size=(wall_half, outer, hz),
                pos=(outer, 0.0, cz), **wall_kwargs)
  body.add_geom(name="bin_wall_nx", size=(wall_half, outer, hz),
                pos=(-outer, 0.0, cz), **wall_kwargs)
  body.add_geom(name="bin_wall_py", size=(outer, wall_half, hz),
                pos=(0.0, outer, cz), **wall_kwargs)
  body.add_geom(name="bin_wall_ny", size=(outer, wall_half, hz),
                pos=(0.0, -outer, cz), **wall_kwargs)
  return spec
