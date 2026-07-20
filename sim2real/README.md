# sim2real

Deploy the sim-trained Squint RL policy onto the physical SO-101-on-frame rig,
over [`livekit-portal`](https://github.com/livekit/livekit-portal). Modeled on
the user's `livekit-actuate/vla-demo` bench: every process joins one LiveKit
room and speaks the wire contract in `portal.yaml`, so the robot and the policy
can run on the same machine or across the network.

The rig is an SO-101 arm on a 1-DOF linear frame (the `so_frame`) with two
cameras (arm-mounted + overhead) -- **seven** controllable joints, versus the
bare 6-motor SO-101 in vla-demo.

## Structure

```
sim2real/
├── portal.yaml          # shared wire contract (7 joints incl. the rail, 2 cameras)
├── common.py            # env / LiveKit token / fps pacer (trimmed from vla_demo.common)
├── pyproject.toml       # deploy deps + console scripts (own uv project)
├── robot/
│   ├── run.py           # robot runtime: SO-101 arm + rail, publishes state + RAW video
│   └── slider.py        # rail (dof_slider) actuator interface + stub  [NEEDS-HARDWARE]
└── policy/
    ├── run.py           # the Squint policy operator (camera-mapping -> inference -> action)
    ├── agent.py         # loads the trained encoder+actor from a checkpoint (sim-free)
    ├── bridge.py        # sim<->real unit/coordinate bridge          [NEEDS-CALIBRATION]
    ├── camera_mapping.py            # vendored apply_mapping/load_mapping
    └── overhead_camera_mapping.json # calibrated overhead rectification (from the sim work)
```

## The camera-mapping wiring (the point of this stack)

The policy was trained on a *narrow, undistorted, cropped* camera view (sim
overhead FOV 38 deg). The real cameras are wide-FOV (innoMaker 120 deg DFOV).
The **robot publishes RAW frames**; the **policy operator reconstructs the sim
view** before every inference by replaying the saved mapping
(`policy/*_camera_mapping.json`) through `apply_mapping`. So the policy always
sees exactly what it saw in sim, and the web-ui still gets the true wide stream.

Per tick, in `policy/run.py`:
raw frame -> `apply_mapping` (rectify to 128x128) -> stack wrist|overhead to
128x128x6 -> squint to 32x32 -> encoder+actor -> normalized delta action ->
integrate into a running joint target -> `bridge.sim_to_real` -> send.

The mappings are produced once per camera with
`rl/maniskill/examples/calibrate_real_camera.py`. The overhead mapping is
already calibrated and copied here; **the arm/wrist mapping still needs
calibrating** (drop `arm_camera_mapping.json` next to the overhead one).

## Quick start

```bash
cd sim2real
cp .env.example .env        # fill LIVEKIT_* and SO101_*
uv sync

uv run robot                # robot host: drives the arm + rail, streams cameras
uv run policy-squint        # policy: idle until it claims control
```

The robot ignores actions whose sender is not the active operator, so the
policy claims control via its `run_policy_squint` RPC (a web-ui button, or call
it directly) before the arm moves. `reset_to_zero_position` parks the rig.

## Before a real rollout (calibration gates)

1. **`policy/bridge.py`** -- the sim<->real per-joint offsets default to a pure
   unit conversion (rad<->deg, m<->mm) with zero offset, i.e. they assume the
   real joint zero == the sim joint zero. Measure each joint's real `.pos` at the
   sim zero pose and set `OFFSET_REAL` (and `SIGN` if a direction is reversed).
2. **`robot/slider.py`** -- swap `StubSlider` for the real rail driver.
3. **`robot/run.py` rest pose** -- set `REST_POSE_DEFAULTS` to the rig's real
   parked pose.
4. **arm/wrist camera mapping** -- calibrate and add `arm_camera_mapping.json`.

Until 1-3 are done the stack runs but must not command the real arm unsupervised;
bring it up first with the stub slider and a slow supervised check.
