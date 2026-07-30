# SO101 Robot - URDF Description

> **Vendored component.** This directory carries the upstream SO-101 arm description and the
> upstream README below, kept for provenance. What SO-Frame actually vendors is narrower than
> what that text describes:
>
> | Upstream mentions | Here |
> | --- | --- |
> | URDF **and** MuJoCo/MJCF files, `scene.xml` | `so101_new_calib.urdf` + `assets/*.stl` only. The MuJoCo model of the full rig is [`simulation/mjcf/`](../../../mjcf/README.md), which is this repo's own, not upstream's. |
> | New **and** old calibration (`so101_old_calib`) | New calibration only — there is nothing to switch between. |
>
> The arm here is a **reference copy**. The model you load is the combined
> [`simulation/urdf/so101_on_frame.urdf`](../../README.md), which merges this arm onto the
> frame; edits made here do not propagate to it automatically.

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

Upstream offers two calibrations:

- **New Calibration (Default)**: each joint's virtual zero is set to the **middle** of its joint range.
- **Old Calibration**: each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**.

**SO-Frame vendors the new calibration only** (`so101_new_calib.urdf`), and every joint limit
quoted in [the URDF README](../../README.md#kinematics) and in
`rl/policy/src/soframe_policy/rig.py` assumes it. Swapping in the old calibration would shift
every joint's zero and invalidate both the trained policies and the deploy bridge's
sim-zero assumption.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not** reflected in the URDF: `gripper` is a revolute joint over
−0.17453 … +1.74533 rad. SO-Frame keeps the radian convention everywhere in sim and converts
to the wire's units at the edge — see `rl/deploy/utils/bridge.py`, which is the one place that
translation happens.
