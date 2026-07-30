# deploy

Run a sim-trained policy on the physical rig, over
[`livekit-portal`](https://github.com/livekit/portal). Every process joins one LiveKit room and
speaks the wire contract in `portal.yaml`, so the robot and the policy can be on the same machine
or on opposite sides of a network.

```
robot/run.py     on the robot: publishes 7-DOF state + RAW camera frames, applies actions
policy/run.py    anywhere:     rectifies frames -> inference -> joint targets
utils/           the sim/wire bridge, the camera mapping, shared scaffolding
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
alone to wire 0 (end of travel, for re-zeroing the carriage), `q` quit. `--viz` opens a window
(`uv sync --group viz`) with the two rectified views the encoder is fed, a bar per joint for the
last action and how far it lags its target, the same keys as buttons, and live **arm lag**, **rail
lag** and **rail step** sliders. It appears even with no frames arriving, so you can tell connected
from stalled.

`--arch` picks a checkpoint from `checkpoints/` by name:

| `--arch` | file | encoder |
|---|---|---|
| `squint` | `squint_ckpt.pt` | CNN over a squinted stack, res 32 |
| `dino_patch` | `dino_patch_policy_ckpt.pt` | frozen DINOv2 patch tokens, res 168 |
| `dino_global_mean` | `dino_global_mean_ckpt.pt` | one mean-pooled vector per camera |
| `dino_cls` | `dino_cls_ckpt.pt` | one CLS vector per camera |

That selects a *file*, not an architecture: the checkpoint still declares its own encoder,
resolution, camera count and proprio layout (see [policy/](../policy/README.md)), and a file whose
contents disagree with the name it was loaded under is an error. That check earns its keep for the
two `dino_global` entries, whose pooling modes produce identical tensor shapes and so cannot be
told apart by loading alone. `--checkpoint <path>` still takes anything outside `checkpoints/`.

An action is a delta per control period, which is to say a velocity: sim integrated one every
100 ms step. Deploy does the same, so an action keeps being applied every tick until a new one
replaces it, and a decision replaces the velocity rather than being the only thing that produces
motion.

What bounds it is a per-group lag budget, `--max-lag-arm DELTAS` (default 1.0) and
`--max-lag-rail DELTAS` (default 2.0): a group's target advances only while it is within that many
action steps of its joints' measured pose. So the target can never run away from a slower arm, which
is what had the jaw closing after it had already passed the cube, and it makes the target a
velocity-clamped ramp instead. A new decision fires once every group is inside its budget.

Each group carries its own budget, which is the point: the rail keeps gliding at its commanded speed
while the arm is being waited on. Timing a 0.49 m traverse at 10 Hz with the arm as the slow
mechanism (arm joints close 15% of their gap per tick, rail 80%):

| target between decisions | decisions/s | rail speed | traverse |
|---|---|---|---|
| frozen | 2.7 | 1.85 cm/s | 26.7 s |
| velocity hold | 2.4 | 6.88 cm/s | 7.2 s |

Same decision rate, 3.7x the rail speed, and 6.88 cm/s is the 7.0 cm/s ceiling the rail was trained
at. A frozen target moved the carriage at `decisions/s x 7 mm`, so a slow *arm* joint throttled
the *rail*, a coupling nothing about the mechanism justifies.

When the rail is the slow mechanism its own budget is what matters, and it is a real bound rather
than advice. Same traverse with the rail also at 15%/tick:

| rail budget | rail speed | traverse | peak rail lag |
|---|---|---|---|
| 0.5 | 1.00 cm/s | 49.4 s | 1.25 |
| 1.0 | 1.55 cm/s | 31.8 s | 1.68 |
| 2.0 | 2.60 cm/s | 19.0 s | 2.54 |
| 4.0 | 4.53 cm/s | 10.9 s | 4.12 |

Peak lag never exceeds the budget by more than the one step an action adds, so nothing runs away.

A joint that physically cannot arrive (a stop, a dead servo) would leave its group's budget spent
forever, so decisions fall back to a 1 s timeout. Motion does not stop while waiting: that is what
the integration is. The per-second line reports `<group> lag L/MAX (joint)` per group plus
`N gate timeouts`; steady timeouts on one joint mean its budget is stricter than the hardware.

Two things are deliberately excluded. **The gripper gets no budget**: its whole range is 9.6 action
steps, so a jaw closed on the object sits several steps short of its command for as long as it
holds, and a position servo only pushes as hard as the distance it is asked to close, so holding its
target back would bleed grip force. Its action is applied once per decision rather than sustained,
since nothing bounds it. **Nothing advances without frames**: a lost camera must not mean the arm
keeps gliding blind on a stale command.

`--rail-step MM` (default 7.0, the trained value) is how far one full-command action moves the
carriage, and therefore its top speed per control period and the unit rail lag is counted in. It is
the one figure in the action contract that came off a control UI rather than a measurement, so the
viz exposes it as a slider: drag it while watching the rail to find what the carriage actually does
per step. Speed follows it linearly (3.5 mm gives 3.47 cm/s, 14 mm gives 13.53 cm/s). Changing it
runs the policy on a different action space than it trained on, which is the trade you are making.

With `--viz` open the window's sliders take over all three values, and each joint's tick turns red
once its group's budget is spent, so the bars show which joint the next decision is waiting on.

On a Mac set `POLICY_DEVICE=mps`: the default device pick is cuda-or-cpu, so a DINOv2 checkpoint
would otherwise run on the CPU.

## The camera mapping

The policy trained on a narrow, undistorted, cropped view (overhead FOV 38°). The real cameras
are wide-angle (120° DFOV). So **the robot publishes RAW frames** and the policy operator
reconstructs the sim view before every inference:

```
raw frame -> rectify (rotate, undistort+zoom+offset, centre-crop, resize) -> stack arm|overhead -> policy
```

Rectification replays `utils/camera_mappings/<camera>_camera_mapping.json` through the same
`apply_mapping` that fit it. Fitting a mapping, and every other calibration or debug tool, lives in
**[rl/calibrate](../calibrate/README.md)**: those need the simulator alongside the robot feed, which
the robot host must never have to install.

`CAMERA_STACK` in `utils/camera_mapping.py` decides which real camera feeds which sim camera.
A missing mapping falls back to a plain resize, which is **out of distribution**: the loop says
so loudly at startup.

## Before the first real rollout

`utils/bridge.py` converts sim units (rad, m) to wire units (deg, rail 0..100). `OFFSET_REAL` is 0
for most joints because the follower's lerobot calibration already puts real `.pos` zero at the URDF
zero pose. `wrist_roll` is the measured exception, at +90°.

Check both assumptions against a live arm from `rl/calibrate` with
`uv run calibrate`: that each joint's calibrated zero is the URDF zero pose (an
offset), and that `SIGN` is identity (a flipped joint drives the arm the wrong way, which no offset
would rescue). `uv run calibrate --bridge` lists what is currently applied. Re-check after re-homing
a servo.

Check the rail first: it is the one axis calibrated from geometry rather than from the follower's
homing.

## utils

| | |
|---|---|
| `bridge.py` | sim ⇄ wire units. Joint order, limits, delta caps and rest pose come from `soframe_policy.rig`, shared with training. |
| `camera_mapping.py` | fit replay + the camera stack (which real camera feeds which sim camera) |
| `pull_frames.py` | grab raw frames off the live robot (passive; safe while a policy drives) |
| `common.py` | env loading, LiveKit tokens, fps pacer, `DEPLOY_ROOT` |

`camera_mappings/` and `captures/` hold the calibration artifacts. The tool that produces them is
[rl/calibrate](../calibrate/README.md), which writes here.

## Commands

```bash
uv sync                                    # once, next to a copy of ../policy
cp .env.example .env                       # LIVEKIT_URL, API key/secret, LIVEKIT_ROOM

# on the robot: publish state + raw frames, apply actions
uv run robot

# wherever the GPU is: drive the arm from a checkpoint
POLICY_DEVICE=mps uv run policy --arch dino_patch --viz   # or squint | dino_global_mean | dino_cls
uv run policy --checkpoint /path/to/ckpt_best.pt --viz    # a file outside checkpoints/

# ship rl/deploy + rl/policy to a robot host, then provision it
./scripts/deploy_to_robot.sh <host> --sync

# grab raw frames off a live robot (passive, safe while a policy drives)
uv run python utils/pull_frames.py
```

`policy/run.py` is the only thing here that moves the robot.

Debug and calibration commands (bridge self-test, live rectified views, manual joint control,
camera fitting) are in [rl/calibrate](../calibrate/README.md), which reads this project's `.env`,
`portal.yaml` and mapping files.

A mapping's `out_size` should match the checkpoint's render size: 128 for squint, 168 for
`dino_patch` and `dino_global`. A mismatch is not an error, the encoder just upsamples and the
policy sees a blurrier view than it trained on. `--viz` reports it on the status line as
`stack <fitted> -> <encoder> px`.
