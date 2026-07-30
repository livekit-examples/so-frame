# SO101 Robot - URDF Description

This folder holds the SO-101 arm on its own: `so101_new_calib.urdf` plus the meshes in
`assets/`. It is vendored from
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101) and kept
for reference. The model this repo actually loads is the combined
[`../../so101_on_frame.urdf`](../../so101_on_frame.urdf), which mounts this arm on the linear
frame; upstream's MuJoCo files were not vendored, and the MJCF here is
[`../../../mjcf/so101_on_frame.xml`](../../../mjcf/so101_on_frame.xml).

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

Upstream ships two differently calibrated SO101 robot files. Only the new one is vendored here:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.urdf`. This is what the combined model uses.
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Not vendored; see upstream's `so101_old_calib.urdf`.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini). This URDF carries only each joint's
`effort` (3 N.m) and `velocity` (10 rad/s) limits; the identified servo parameters (armature,
Coulomb friction, damping) live in
[`../../../mjcf/sts3215.xml`](../../../mjcf/sts3215.xml).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files.

---

Feel free to open an issue or contribute improvements!
