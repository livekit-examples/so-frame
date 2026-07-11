# Simulation

The SO-Frame + SO-101 + two cameras, described two ways that share the same meshes: a
**URDF** and a **MuJoCo (MJCF)** model.

## URDF

`urdf/so101_on_frame.urdf` is the combined model for URDF viewers, PyBullet, Isaac, etc.
It includes the slider joint, the arm, and both camera frames (`frame_wrist_camera`,
`frame_overhead_camera`). See **[urdf/README.md](urdf/README.md)** for kinematics, joint
limits, the mounting details, and the interactive camera-alignment helper.

![SO-Frame setup](assets/setup.png)

## MJCF (MuJoCo)

`mjcf/scene.xml` is the MuJoCo model. On top of the URDF geometry it adds actuators, box
collisions for the frame, a floor/light/skybox, and the two cameras as real renderable
MuJoCo cameras. Load it with:

```bash
python -m mujoco.viewer --mjcf=simulation/mjcf/scene.xml
```

See **[mjcf/README.md](mjcf/README.md)** for actuators, collision, and camera details.

Same setup, plus what each camera sees:

| Setup | `frame_wrist_camera` | `frame_overhead_camera` |
|:---:|:---:|:---:|
| ![setup](assets/setup.png) | ![wrist camera](assets/cam_wrist.png) | ![overhead camera](assets/cam_overhead.png) |
