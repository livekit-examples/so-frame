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

```bash
uv sync
cp .env.example .env      # LIVEKIT_URL, API key/secret, LIVEKIT_ROOM
```

## Run

```bash
# on the robot
uv run robot

# wherever the GPU is
POLICY_CHECKPOINT=/path/to/ckpt_best.pt uv run policy --claim
```

`--claim` drives immediately; without it the policy waits to be claimed by the web UI. Keys while
running: `r` reset to rest and hold, `p` pause/resume, `q` quit. `--viz` shows what the policy
sees.

There are no architecture flags. The checkpoint records its own encoder, input resolution,
camera count and proprio layout — see [policy/](../policy/README.md).

## The camera mapping

The policy trained on a narrow, undistorted, cropped view (overhead FOV 38°). The real cameras
are wide-angle (120° DFOV). So **the robot publishes RAW frames** and the policy operator
reconstructs the sim view before every inference:

```
raw frame -> rectify (rotate, undistort, crop, resize) -> stack arm|overhead -> policy
```

Rectification replays `utils/camera_mappings/<camera>_camera_mapping.json`. To fit one:

```bash
# in rl/environments/maniskill (needs the simulator), dump what the sim camera sees
uv run python examples/dump_reference_views.py --out ../../rl/deploy/utils/reference_views

# here, drag sliders until the real frame matches it
uv run python utils/calibrate_camera.py utils/captures/real_overhead_camera.png \
    --reference utils/reference_views/overhead_camera.png --camera overhead
```

A missing mapping falls back to a plain resize, which is **out of distribution** — the loop says
so loudly at startup.

## Before the first real rollout

`utils/bridge.py` converts sim units (rad, m) to wire units (deg, rail 0..100). The rail mapping
is calibrated; **the six arm/gripper joints still assume real zero == sim zero.** Measure each
joint's real `.pos` at the sim-zero pose into `OFFSET_REAL` first. `uv run python
utils/debug_policy.py --bridge` prints the current state and flags this.

## utils

| | |
|---|---|
| `bridge.py` | sim ⇄ wire units. Joint order, limits, delta caps and rest pose come from `soframe_policy.rig`, shared with training. |
| `camera_mapping.py` | fit replay + the camera stack (which real camera feeds which sim camera) |
| `calibrate_camera.py` | fit a mapping against a sim reference render |
| `pull_frames.py` | grab raw frames off the live robot (passive; safe while a policy drives) |
| `debug_policy.py` | check the wiring with no policy: `--frame`, `--bridge`, `--live`, `--snapshot`, `--control`, `--cli` |
| `common.py` | env loading, LiveKit tokens, fps pacer |

Only `debug_policy.py --control` and `policy/run.py` move the robot.
