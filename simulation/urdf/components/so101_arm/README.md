# SO101 Robot - URDF Description

> **Vendored component.** This directory carries the upstream SO-101 arm description, vendored
> from [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101),
> and the upstream README below, kept for provenance. What SO-Frame actually vendors is
> narrower than what that text describes:
>
> | Upstream mentions | Here |
> | --- | --- |
> | URDF **and** MuJoCo/MJCF files, `scene.xml` | `so101_new_calib.urdf` plus the meshes in `assets/`: upstream's STLs, and the two `*_convex.obj` fixed-jaw collision hulls this repo added. Upstream's MuJoCo files were not vendored; the MuJoCo model of the full rig is [`simulation/mjcf/`](../../../mjcf/README.md), specifically [`so101_on_frame.xml`](../../../mjcf/so101_on_frame.xml), which is this repo's own, not upstream's. |
> | New **and** old calibration (`so101_old_calib`) | New calibration only, so there is nothing to switch between. |
>
> The arm here is a **reference copy**. The model you load is the combined
> [`../../so101_on_frame.urdf`](../../so101_on_frame.urdf) (documented in
> [the URDF README](../../README.md)), which merges this arm onto the linear frame; edits made
> here do not propagate to it automatically.

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

Upstream ships two differently calibrated SO101 robot files:

- **New Calibration (Default)**: each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.urdf`.
- **Old Calibration**: each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.urdf`.

**SO-Frame vendors the new calibration only** (`so101_new_calib.urdf`, which is also what the
combined model uses). The old one is not here; see upstream if you need it. Every joint limit
quoted in [the URDF README](../../README.md#kinematics) and in
`rl/policy/src/soframe_policy/rig.py` assumes the new calibration. Swapping in the old one would
shift every joint's zero and invalidate both the trained policies and the deploy bridge's
sim-zero assumption.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini). This URDF carries only each joint's
`effort` (3 N.m) and `velocity` (10 rad/s) limits; the identified servo parameters (armature,
Coulomb friction, damping) live in
[`../../../mjcf/sts3215.xml`](../../../mjcf/sts3215.xml).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not** reflected in the URDF or in the MJCF: `gripper` is a revolute joint over
−0.174533 … +1.74533 rad. SO-Frame keeps the radian convention everywhere in sim and converts to
the wire's units at the edge, see `rl/deploy/utils/bridge.py`, which is the one place that
translation happens.

---

Feel free to open an issue or contribute improvements!
