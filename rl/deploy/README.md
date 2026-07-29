# deploy

Run a sim-trained policy on the physical rig, over
[`livekit-portal`](https://github.com/livekit/portal). Every process joins one LiveKit room and
speaks the wire contract in `portal.yaml`, so the robot and the policy can be on the same machine
or on opposite sides of a network.

```
robot/run.py     on the robot: publishes 7-DOF state + RAW camera frames, applies actions
policy/run.py    anywhere:     rectifies frames -> inference -> joint targets
utils/           the bridge, camera calibration, and debug tools
```

## Setup

`uv sync` + `cp .env.example .env` (LIVEKIT_URL, API key/secret, LIVEKIT_ROOM). See
[Commands](#commands).

`soframe-policy` is a path dependency at `../policy`, so this project only syncs next to a copy
of [`rl/policy/`](../policy/README.md), and that relative path has to resolve the same way on the
dev machine and on the robot host. `./scripts/deploy_to_robot.sh <host> --sync` arranges it:
`rl/deploy` to `~/so-frame-deploy`, `rl/policy` to `~/policy`. Copying only this directory gives
`Distribution not found at: file:///…/policy` on sync.

## Run

The policy claims control on startup and starts **paused**, holding the pose, so nothing moves
until you press `p`. `--no-start-paused` drives on launch; `--no-claim` sits idle until the web UI
claims it. Keys while running: `p` pause/resume, `r` reset to rest and hold, `0` ramp the rail
alone to wire 0 (end of travel, for re-zeroing the carriage), `q` quit. `--viz` shows what the
policy sees, and appears even with no frames arriving so you can tell connected from stalled.

There are no architecture flags. The checkpoint records its own encoder, input resolution,
camera count and proprio layout, see [policy/](../policy/README.md). On a Mac set
`POLICY_DEVICE=mps`: the default device pick is cuda-or-cpu, so a DINOv2 checkpoint would
otherwise run on the CPU.

## The camera mapping

The policy trained on a narrow, undistorted, cropped view (overhead FOV 38°). The real cameras
are wide-angle (120° DFOV). So **the robot publishes RAW frames** and the policy operator
reconstructs the sim view before every inference:

```
raw frame -> rectify (rotate, undistort, crop, resize) -> stack arm|overhead -> policy
```

Rectification replays `utils/camera_mappings/<camera>_camera_mapping.json` through the same
`apply_mapping` that fit it. `utils/calibrate_camera.py` fits one against a sim reference render.
Sliders: `rot90` / `angle` / `k1` / `k2` / `focal` straighten and scale the raw frame, then `zoom`
and `crop cx` / `cy` / `size` frame it. `[c]` re-centres the crop, `[s]` saves. Pass `--out-size`
to match the checkpoint's render size (128 for squint, 168 for `dino_patch`); it defaults to
whatever the existing mapping recorded. The sim FOV and camera pose are not sliders: the
reference render is fixed, and the URDF pose is ground truth.

`CAMERA_STACK` in `utils/camera_mapping.py` decides which real camera feeds which sim camera.
A missing mapping falls back to a plain resize, which is **out of distribution**: the loop says
so loudly at startup.

## Before the first real rollout

`utils/bridge.py` converts sim units (rad, m) to wire units (deg, rail 0..100). `OFFSET_REAL` is 0
for the arm and gripper because the follower's lerobot calibration already puts real `.pos` zero at
the URDF zero pose, so there is nothing to measure. Two assumptions ride on that, both visible in
`debug_policy.py --live`: that the calibrated zero is the URDF zero pose, and that `SIGN` is
identity, since a flipped joint drives the arm the wrong way and no offset would rescue it.
Re-check after re-homing a servo.

Check the rail first with `debug_policy.py --control` (`far`/`near`): it is the one axis calibrated
from geometry rather than from the follower's homing.

## utils

| | |
|---|---|
| `bridge.py` | sim ⇄ wire units. Joint order, limits, delta caps and rest pose come from `soframe_policy.rig`, shared with training. |
| `camera_mapping.py` | fit replay + the camera stack (which real camera feeds which sim camera) |
| `calibrate_camera.py` | fit a mapping against a sim reference render |
| `pull_frames.py` | grab raw frames off the live robot (passive; safe while a policy drives) |
| `debug_policy.py` | check the wiring with no policy: `--frame`, `--bridge`, `--live`, `--snapshot`, `--control` |
| `common.py` | env loading, LiveKit tokens, fps pacer |

## Commands

```bash
uv sync                                    # once, next to a copy of ../policy
cp .env.example .env                       # LIVEKIT_URL, API key/secret, LIVEKIT_ROOM

# on the robot: publish state + raw frames, apply actions
uv run robot

# wherever the GPU is: drive the arm from a checkpoint
POLICY_CHECKPOINT=/path/to/ckpt_best.pt uv run policy --viz

# ship rl/deploy + rl/policy to a robot host, then provision it
./scripts/deploy_to_robot.sh <host> --sync

# check the wiring, no motion
uv run python utils/debug_policy.py --bridge     # unit round-trip check, no robot needed
uv run python utils/debug_policy.py --live       # periodic rectified dumps, read-only
uv run python utils/debug_policy.py --snapshot   # one raw frame per camera, for calibration

# manual joint control (trackbars + live view). MOVES THE ROBOT
uv run python utils/debug_policy.py --control

# grab raw frames off a live robot (passive, safe while a policy drives)
uv run python utils/pull_frames.py
```

Only `debug_policy.py --control` and `policy/run.py` move the robot.

## Calibrating a camera

In order. Steps 1 and 2 are independent; step 3 needs both.

```bash
# 1. real frames, with the robot process running. Gripper CLOSED: the jaws are the wrist
#    camera's only landmark, so the real frame must match how the reference is rendered.
uv run python utils/pull_frames.py                 # -> utils/captures/real_{arm,overhead}_camera.png

# 2. sim references, in rl/environments/maniskill (needs the simulator; gripper closed by default).
#    --size must be the checkpoint's render size.
uv run python examples/dump_reference_views.py --out ../../deploy/utils/reference_views --size 168

# 3. fit each camera, here (no simulator needed). --out-size must match --size above.
#    The arm camera fits against the sim WRIST camera.
uv run python utils/calibrate_camera.py utils/captures/real_overhead_camera.png \
    --reference utils/reference_views/overhead_camera.png --camera overhead --out-size 168
uv run python utils/calibrate_camera.py utils/captures/real_arm_camera.png \
    --reference utils/reference_views/wrist_camera.png --camera arm --out-size 168
```

Straighten the rig's edges with `k1`/`k2`/`angle` first, then match scale with `focal`/`zoom`,
then frame with `crop cx`/`cy`/`size`, and read the blend last. `[s]` writes
`utils/camera_mappings/<camera>_camera_mapping.json` plus a `_fit.png` preview.

`--size` / `--out-size` are 128 for squint, 168 for `dino_patch`. A mismatch is not an error: the
encoder just upsamples, and the policy sees a blurrier view than it trained on. `--viz` shows it,
the lower tile row is the encoder's real input resolution.
