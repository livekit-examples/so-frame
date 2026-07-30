# SO-101 on Linear Frame (Combined URDF)

`so101_on_frame.urdf` is a single, ready-to-use model that mounts the **SO-101 arm**
onto the **1-DOF linear frame** (`so_frame`). Load this one file; mesh paths are
relative to it, so it works out of the box in URDF viewers, PyBullet, Isaac, MuJoCo
loaders, etc. from the `simulation/urdf/` directory.

```
simulation/urdf/
├── so101_on_frame.urdf              <- load this
├── README.md
├── helper/
│   └── wrist_camera_aligner.html    <- interactive wrist-mount aligner (see Cameras)
└── components/
    ├── base_frame/                  <- the linear frame (so_frame)
    │   ├── urdf/so_frame.urdf       <- standalone frame, for reference
    │   └── meshes/*.stl
    ├── so101_arm/                   <- the SO-101 arm (new calibration)
    │   ├── so101_new_calib.urdf     <- standalone arm, for reference
    │   └── assets/*.stl, *.obj      <- *.obj = the two fixed-jaw collision hulls
    └── wrist_camera/                <- Hex-Nut MF wrist camera mount (32×32 UVC)
        └── SO-ARM101_camera_wrist_mount.stl
```

## Kinematics

Root link: `root` (the fixed frame structure).

| Joint          | Type       | Range                    | Notes                                  |
|----------------|------------|--------------------------|----------------------------------------|
| `dof_slider`   | prismatic  | −0.40286 … +0.41714 m    | The frame's linear axis (carriage travel) |
| `so101_mount`  | fixed      | n/a                      | Bolts the SO-101 base to the carriage plate |
| `shoulder_pan` | revolute   | −1.91986 … +1.91986 rad  | SO-101 joint 1                         |
| `shoulder_lift`| revolute   | −1.74533 … +1.74533 rad  | SO-101 joint 2                         |
| `elbow_flex`   | revolute   | −1.69 … +1.69 rad        | SO-101 joint 3                         |
| `wrist_flex`   | revolute   | −1.65806 … +1.65806 rad  | SO-101 joint 4                         |
| `wrist_roll`   | revolute   | −2.74385 … +2.84121 rad  | SO-101 joint 5                         |
| `gripper`      | revolute   | −0.174533 … +1.74533 rad | SO-101 gripper jaw                     |

Chain: `root → … → 100cm_4 →[dof_slider]→ 20mm_gantry_plate →[fixed]→ base_mount
→[so101_mount]→ base_link →[shoulder_pan]→ … → gripper_link →[fixed]→ frame_wrist_camera`.

The **new calibration** of the arm is used (each arm joint's zero is the middle of its
range).

## How the arm is mounted

The frame already carries an orange adapter plate (`base_mount`) fastened to the sliding
carriage (`20mm_gantry_plate`). That plate is machined with the SO-101 base footprint.
The `so101_mount` fixed joint seats the SO-101 `base_link` onto it:

- **Contact:** the SO-101 base's flat bottom rests flush on the plate's upper face;
  base bottom and plate top meet at the same plane (verified: both at world z = 0.075 m
  with the carriage at slider = 0, which includes the `fix_1` rig lift below).
- **Orientation:** `rpy = (π, 0, π/2)`, a 180° flip about X so the arm points *up*, away
  from the carriage, plus a 90° yaw so the base's screw pattern lines up with the plate.
- **Position:** `xyz = (0, −0.0410, −0.0124)` in the `base_mount` frame.

The pose was recovered by registering the SO-101 base's mounting-hole pattern against the
hole pattern milled into the `base_mount` plate (best fit: yaw ≈ 90°, ~1.2 mm RMS over 7
matched holes). If you later get the exact CAD mate, only the `so101_mount` `<origin>`
needs adjusting; nothing else changes.

## Cameras

The model has **two camera frames**. Both look along their link's **+X** axis: **+X = view
direction, −Y = image right, +Z = image up**, the SAPIEN convention that ManiSkill mounts a
camera with (`rl/environments/maniskill` mounts on these links directly, with no correction
pose). This is *not* the +Z-forward REP-103 optical convention; the remap between the two is
spelled out in the comments on the `frame_wrist_camera_joint` and `frame_overhead_camera` joints
in the URDF, and in the [MJCF README](../mjcf/README.md#cameras) for MuJoCo's −Z-forward cameras.

- **`frame_wrist_camera`**: eye-in-hand, on the wrist (see below).
- **`frame_overhead_camera`**: on the frame's `32mm_camera_holder`, looking down at the
  workspace. This one comes straight from the `so_frame` CAD (an Onshape mate connector) and
  is fixed to the frame.

### Wrist camera

The **Hex-Nut Recess Wrist Camera (MF)** mount (`components/wrist_camera/`) is attached to
the wrist-roll follower via the fixed joint `wrist_camera_mount_joint` (parent
`gripper_link`), so it rolls with `wrist_roll` but does not move when the gripper opens.

- **`frame_wrist_camera`** is the virtual camera's frame; attach your camera here. It is posed
  to look along the gripper's approach axis (gripper −Z), toward the grasp, matching the
  reference installation photos. On top of the +X convention above it carries a 180° roll
  about that view axis, so the render comes out the same way up as the real module, which is
  mounted upside down.
- Mount STL is in millimetres, so it carries `scale="0.001 0.001 0.001"`.

The mount pose was aligned interactively with `helper/wrist_camera_aligner.html`: open it
in a browser, drag the gizmo to line the mount's clamp holes up with the gripper's holes,
and it prints the `wrist_camera_mount_joint` / `frame_wrist_camera_joint` origins to paste
back into the URDF. To nudge it, edit those two joint origins (or re-run the helper). The
helper is self-contained (no assets to fetch), and it emits its camera joint with **+Z** as
the view direction, so that `rpy` needs the remap above before it goes into the URDF; the
`xyz` carries over unchanged.

## Collision

Most of the frame is **visual only**, with no `<collision>` element, since it was built for
rendering rather than physics. Three groups are the exception:

- **The lightbox's 4 panels** (`part_1`, `part_1_1`, `part_1_2`, `part_1_3`: floor + 3 walls,
  the "matte mica" side panels in the [USD](../usd/README.md) material scheme) each get a thin
  invisible box `<collision>` sized to its footprint. `part_1_1` is the floor; without its pad,
  anything resting on the work surface falls straight through to whatever ground plane a
  consuming sim provides.
- **The rail the gantry rides on** (`100cm_4`, the `dof_slider` joint's parent) gets a
  `0.02 × 0.02 × 1.0 m` box matching the 2020 profile, so the arm cannot sweep straight through
  it and learn motions that would hit the real aluminium. The other 13 extrusions stay visual
  only; they are out of the arm's reach.
- **The 7 arm links** keep the SO-101's own per-link mesh `<collision>` from
  `components/so101_arm/`, one per visual mesh, except on `gripper_link`, where the fixed jaw is
  split into the two convex hulls `fixed_jaw_base_convex.obj` and `fixed_jaw_finger_convex.obj`
  so each hull hugs the real surface at the finger.

Everything else (brackets, wheels, motors, screws, the wrist-camera mount) relies on the
consumer to add its own collision. See the [MJCF README](../mjcf/README.md#collision) for how
the MuJoCo model covers the rest of the frame with box geoms.

The root `fix_1` joint (`root → 100cm`) carries a small `+0.01 m` origin lift so the lightbox
floor's collision pad sits at world `z ≈ 0` rather than slightly underground.

## Visual geometry

Some links are **frames only**: they carry no `<visual>`, so they render as nothing while
keeping the kinematic chain intact. They were trimmed because the RL render stage is
geometry-bound (see `rl/environments/maniskill`), and these parts are either hidden inside an
enclosure or too small to read at the 128 px the policy cameras render at:

| Dropped | Why |
| --- | --- |
| `handle`, `handle_1` | frame handles, outside the workspace |
| `m5x25_low_profile_screw*` | fastener heads |
| `ge_27`, `motor_1723_3`, `pcb_chazuo_92`, `pinion`, the two metal servo horns | rail drive internals; the enclosure (`sg_ziji_15`, `xg_ziji_16`, `zk_122`) is kept |

That takes a rig from 756k to 567k triangles, about 25% off. The STLs stay on disk
(`Handle.stl`, `M5x25_low_profile_screw.stl`, `GE_27.stl`, `MOTOR_1723_3.stl`,
`PCB_CHAZUO_92.stl`, `Pinion.stl`) and are simply unreferenced, so re-adding a `<visual>`
block is all it takes to get one back.

**The MJCF has not been trimmed.** `../mjcf/so101_on_frame.xml` still renders all of these,
so the two models differ in visual geometry (only in visual geometry; kinematics, collision
and camera frames still match).

## Colors

Set in the combined URDF (edit the palette at the top of the build if you want to tweak):

- SO-101 arm links: **white**; gripper fingers (`wrist_roll_follower`, `moving_jaw`): **orange**
- Frame `base_mount`: **orange** (same as gripper), the only link that uses the
  `accent_orange` material. The `pinion` and the other rail-drive internals no longer carry a
  `<visual>` (see above), so nothing is drawn for them and no material is assigned to them
- Aluminium extrusions (`50cm*`, `100cm*`) and the 2020 brackets: **medium gray** (to separate
  from the light side panels)
- Frame `32mm_camera_holder`: **white**; motors stay black

These are the *baked* colours. `rl/environments/maniskill` recolours the rig at load time from
its own scheme in `config.py` (black frame and slider, white arm and gripper, motor hardware
left alone), so the sim renders will not match this palette.

## Regenerating

`so101_on_frame.urdf` is assembled from `components/base_frame/urdf/so_frame.urdf` and
`components/so101_arm/so101_new_calib.urdf`: mesh paths rewritten to be relative, the two
robots merged, the `so101_mount` joint added, the wrist camera (`wrist_camera_mount` +
`frame_wrist_camera`) attached, and the color scheme applied.
