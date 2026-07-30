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
last action and how far the arm still lags its target, the same keys as buttons, and live **arm
lag** and **rail lag x** sliders. It appears even with no frames arriving, so you can tell connected
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

`--max-lag-arm DELTAS` (default 1.0) decides how long the target is held after each action: until
every arm joint is within that many action steps of where it was put. `--rail-lag-scale X` (default
3.0, minimum 1.0) is the rail's tolerance as a multiple of it, so the rail is always the lighter
gate and tightening the arm cannot make the carriage the joint being waited on.

The task is quasi-static, so waiting is free, and it means every inference runs on an observation
the arm has actually reached rather than one from two steps ago, which is the situation training
had.

The gate is the arm's measured lag, not a stopwatch, because a fixed wait is wrong in both
directions. Measured against a simulated arm at 10 Hz:

| case | fixed 0.3 s | lag gate 0.5 |
|---|---|---|
| arm closes 80% of the gap per tick, full-step actions | 3.3 dec/s | 10.0 dec/s |
| arm closes 20% per tick, full-step actions | 3.3 dec/s | 2.1 dec/s |
| arm closes 20% per tick, 0.05-step actions | 3.3 dec/s | 10.0 dec/s |

The last row is the case the stopwatch cannot express: the action is already finished, and waiting
is pure dead time. The middle row is the opposite, a move that needs longer than 0.3 s.

A joint that physically cannot arrive (a mechanical stop, a dead servo) would wedge the loop, so the
gate gives up after 1 s and decides anyway. The per-second line reports `<group> lag L/MAX (joint)`
for each group plus `N gate timeouts`; steady timeouts on one joint mean the gate is stricter than
the hardware, not that the policy is stuck.

Sensible values are roughly 0.1 to 2.0: one action moves a joint at most one step, so above ~2
nothing is ever gated, and below ~0.1 the joint may never get there and every decision waits out the
timeout. With `--viz` open the window's sliders take over the values and each joint's tick turns red
above its own gate, so the bars show which joint the next decision is waiting on. Tune them while
the arm moves rather than restarting the run per guess.

### Why the rail's tolerance is lighter, and what that can and cannot fix

One action step means a different physical thing on each mechanism: 0.85% of travel for the
belt-driven carriage, 2.86° for a direct position servo. The carriage crosses 117 steps end to end,
so its last fraction of a step is a millimetre no policy cares about, while the wrist's is the
difference between the jaw being where the frame says it is or not. Hence a multiplier rather than a
second independent number: whatever precision you demand of the arm, the rail is allowed three times
the slack, and the rail can never be what the arm is waiting on.

Rail speed is `decisions/s x 7 mm`, because one action advances the target by one step. The gate's
decision rate *is* the rail's speed, so where that rate comes from decides whether a lighter rail
tolerance helps at all. Simulated at 10 Hz, asking for full-speed rail throughout:

**When the rail is the slow mechanism** (closes 15% of its gap per tick, arm 80%) the multiplier is
the whole fix:

| arm gate | rail x | rail gate | decisions/s | rail speed |
|---|---|---|---|---|
| 1.0 | 1.0 | 1.00 | 2.3 | 1.6 cm/s |
| 1.0 | 3.0 | 3.00 | 5.1 | 3.5 cm/s |
| 1.0 | 6.0 | 6.00 | 10.0 | 6.9 cm/s |

At x6 the rail is never the blocker and reaches the 7 cm/s it was trained at.

**When the arm is the slow mechanism** (rail 80%, arm 15%) the multiplier does nothing at all:

| arm gate | rail x | rail gate | decisions/s | rail speed |
|---|---|---|---|---|
| 1.0 | 1.0 | 1.00 | 2.3 | 1.6 cm/s |
| 1.0 | 3.0 | 3.00 | 2.3 | 1.6 cm/s |
| 1.0 | 6.0 | 6.00 | 2.3 | 1.6 cm/s |

There is one decision for all seven joints, so while the arm is holding it the rail gets no new
target either, whatever its own tolerance says. The viz distinguishes the two cases directly: the
info line names the joint holding each group, and only red ticks are over their gate. If the red
ticks are arm joints, raising `rail lag x` will not help and the levers are `arm lag` or accepting
the rate.

**The gripper is not gated at all.** Its whole range is 9.6 action steps, so a jaw closed on the
object sits several steps short of its commanded position for as long as it holds, and no useful
tolerance would ever clear. Gating on it would drop every grasp to the timeout rate, during the part
of the task that matters most. Its lag still shows in the viz, just never in red.

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
