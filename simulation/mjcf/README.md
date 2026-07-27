# SO-101 on Linear Frame (MuJoCo / MJCF)

MuJoCo version of the combined model. **Load `scene.xml`** (it includes the model plus a
floor, lights, and skybox):

```bash
python -m mujoco.viewer --mjcf=simulation/mjcf/scene.xml
```

```
simulation/mjcf/
├── scene.xml            <- load this (model + floor/light/skybox)
├── so101_on_frame.xml   <- the model (bodies, joints, actuators, cameras)
└── sts3215.xml          <- Feetech STS3215 servo model (included by the model)
```

Meshes are shared with the URDF via `meshdir="../urdf"`, so this reuses
`../urdf/components/**/*.stl` (no duplicated files).

## Actuators

7 position actuators, `ctrlrange` set to each joint's limit:

`dof_slider` (the frame slider) + `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`, `gripper`.

The 6 arm joints use the **`sts3215` class** from `sts3215.xml` (`<include>`d by the model),
which carries the Feetech STS3215's identified parameters (BAM model): `armature`,
`frictionloss` (Coulomb), and `damping` = viscous friction + `kt²/R` back-EMF. Position-loop
`kp`/`forcerange` live in that class too and are tunable. The `dof_slider` (frame rack &
pinion) uses its own `slider` class in `so101_on_frame.xml`.

## Cameras

Two cameras, named to match the URDF camera frames:

| Camera | View |
|--------|------|
| `frame_wrist_camera`    | eye-in-hand, looks down the gripper approach at the grasp |
| `frame_overhead_camera` | on the frame's camera-holder, looks down at the workspace |

Render from one with `renderer.update_scene(data, camera="frame_wrist_camera")` (or
`"frame_overhead_camera"`). Note: MuJoCo cameras look down their local **−Z**, so the poses
here are the optical frames rotated 180° about X from the REP-103 (+Z view) convention used
in the URDF.

## Collision

The frame meshes are **visual only** (`contype=0`). Collision for the 2020 frame is provided
by 14 **box geoms** (one per extrusion, group 3) so the arm can't pass through the rails.

The lightbox's 4 panels (`Part_1`, `Part_1_1`, `Part_1_2` ×2, floor + 3 walls) are visual-only
meshes too, so each also gets an invisible box geom (`rgba="1 1 1 0"`, no group override) as a
collision pad. `Part_1_1` is the floor: without its pad, anything placed on the work surface
falls straight through to the ground plane. These pads used to be added at runtime by
`rl/environments/mjlab` (a `work_floor_collision` geom injected via `mujoco.MjSpec`); they're now baked into
this file instead, so every consumer of this model (mjlab, the Isaac Sim MJCF import in
`../usd/README.md`, etc.) gets a real work surface without needing its own patch.

The arm keeps its own mesh collisions. The frame is static (welded to the world); only the
slider and the arm move.

## Regenerating

Converted from `../urdf/so101_on_frame.urdf` with MuJoCo's URDF importer, then augmented with
the `<option>`/`<default>`, actuators, the two cameras, the extrusion collision boxes, and the
lightbox panel collision pads.
