# SO-101 on Linear Frame (OpenUSD)

`so101_on_frame.usd` is a single self-contained USD scene (geometry baked in, Z-up, meters)
with physically-based materials and soft overhead lighting. Open it in usdview, Blender,
Omniverse, Houdini, etc.

![USD render](../assets/usd_render.png)

## Materials

Each part is bound to a `UsdPreviewSurface` chosen to match the real build:

| Material | Parts |
|----------|-------|
| **Aluminium** (metallic) | 2020 extrusions, corner/angle brackets, gantry plate |
| **Matte mica** (white, rough) | the light-diffusing side panels |
| **PLA — white** | SO-101 printed arm parts, camera holder, wrist-cam mount |
| **PLA — orange** | gripper fingers, `base_mount`, `pinion` |
| **Plastic — black** | servos/motors, PCBs, handles |
| **Steel** (metallic) | bearings, screws, nuts, V-wheels, servo horns |

Colors follow the same scheme as the URDF/MJCF.

## Lighting

Soft light from above: a large rectangular **area light** over the frame (like the softbox
in the real rig) plus a dim **dome** for ambient fill.

## Cameras

Three cameras: `/World/Camera` (the framing view above) plus the two on-robot cameras at the
same frames as the URDF/MJCF:

| `frame_wrist_camera` | `frame_overhead_camera` |
|:---:|:---:|
| ![wrist](../assets/usd_cam_wrist.png) | ![overhead](../assets/usd_cam_overhead.png) |

## Render

```bash
usdrecord --camera /World/Camera --imageWidth 1600 so101_on_frame.usd out.png
# or a robot camera:
usdrecord --camera /World/frame_wrist_camera --imageWidth 960 so101_on_frame.usd wrist.png
```

(Any USD-aware renderer works; the preview above is Hydra/Storm.)

## Regenerating

Baked from `../urdf/so101_on_frame.urdf` (default/calibrated pose) with material assignment
by part, vertex welding, and the light + camera rig added.
