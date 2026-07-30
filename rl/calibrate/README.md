# calibrate

Calibration and debug tools for the physical rig: the joint bridge, the camera mapping, and
sim-vs-real comparison. Nothing here runs on the robot host.

```
debug_policy.py  the tool: drives the arm and fits the mappings
window.py        the window (PySide6, on the plain widgets in deploy's utils/qt.py)
sim_mirror.py    renders the sim cameras at a given qpos
```

## Why this is its own project

These tools need the simulator AND the live robot feed in one process. `livekit-portal` ships cp312
wheels only and `sapien` has cp312 wheels, so Python 3.12 is the one interpreter where both work.
They cannot live in `rl/deploy`, because putting `mani_skill` in that project's lockfile would force
the robot host to resolve `sapien` for linux aarch64 on every `uv sync`, for a dependency it never
installs.

It path-depends on `rl/deploy` rather than copying from it. `utils.bridge` (sim to wire units),
`utils.camera_mapping` (the mapping format, its replay and its constructor), `utils.common` (env
loading, tokens, the pacer) and `utils.qt` (the widgets both operator windows share) are imported,
so a mapping fitted here is replayed byte-for-byte by the deploy loop, and `.env` / `portal.yaml` /
`camera_mappings/` are read and written in the deploy tree.

## Calibrating the cameras

One command. **This moves the robot**, so keep an e-stop in reach. There is no
confirmation prompt: the joint fields open at the arm's measured pose, so nothing
moves until you change one.

```bash
uv sync
uv run calibrate
```

One window, resizable: **REAL | SIM | OVERLAY** for the camera being fitted on top, a strip for the
other camera under it, and the controls across the bottom. Both halves scroll rather than clip, and
the splitter between them is draggable.

Every value is a slider and a spin box over the same number: drag to search, type to land on an
exact value. The spin box decimals are what the mapping stores, so nothing you can dial in gets
rounded away on save.

The **arm** sliders drive the robot, one per joint, ranged to the sim joint limits so you cannot
command outside them. Targets ramp at the trained per-tick speed rather than snapping.

The **camera mapping** sliders fit the mapping: `rot90` / `angle` / `k1` / `k2` / `focal` / `zoom` /
`offset x` / `offset y`. They reseed from the saved mapping when you switch cameras, so you are
always nudging the fit that currently ships.

Colour is the same rail and the same file: `gain R` / `gain G` / `gain B` and `gamma`,
applied after the resize. **match colour to sim** sets the three gains so the real channel
means land on the sim render's, which is the right target: neutralising to grey would only make the
camera self-consistent, while the policy needs it to look like what it trained on. Hold the arm
still when you press it, since it measures the frame in front of you. It measures with the current
gains taken back out and resets `gamma` to 1, so pressing it twice does not compound.

Rectification is geometry and never touches pixel values, so a colour cast reaches the encoder
untouched unless corrected here. The policy trained under +-10% per-channel gain and 0.7-1.4 gamma
augmentation, so the goal is landing inside that envelope rather than perfect neutrality.

`zoom` is the only field of view control and `offset x`/`offset y` are the only framing controls,
both applied to the undistort output matrix. The kept square is always the centred largest one.
An earlier version exposed a movable crop window instead, which could not work: the crop was the
largest square that fits, so its centre was pinned in the narrow axis, and its size fought `zoom`
for the same job. A mapping still carrying those `crop_size`/`crop_cx` fields is refused on load,
here and in the deploy loop, and has to be refit.

The slider between the two image rows sets the overlay mix, real at one end and sim at the other.
Buttons:
**go to rest**, **go to park**, **hold here** (adopt the pose the arm is at), **clear speed peaks**,
**match colour to sim**, **reset zoom + offset**, **save mapping**. Close the window to release
control.

Order that works: straighten the rig's edges with `k1`/`k2`/`angle`, match scale with
`focal`/`zoom`, frame with the offsets, then read the blend.

The point of driving both at once is the wrist camera. Its view is almost entirely jaws, so a fit
checked at one arm pose tells you nothing about whether it holds as the arm moves. Sweep poses in
the control window and watch the blend hold, or not.

`--sim-size` (default 168) sets both the sim render resolution and the mapping's `out_size`. Use the
resolution your checkpoint trained at: 128 for squint, 168 for `dino_patch` and `dino_global`.

## Other modes

```bash
uv run calibrate                 # the tool. MOVES THE ROBOT.
uv run calibrate --bridge        # joint round-trip and the applied offsets, then exit. No robot, no sim.
uv run calibrate --ui-smoke 5    # the window alone on synthetic frames. No robot, no simulator.
uv run calibrate --sim-size 128  # match a squint checkpoint instead of dino's 168
```

There is no read-only mode and no way to fit a mapping without the simulator: driving the arm and
comparing against sim are the same activity, and separating them is what made the old two-step flow
able to validate a single arm pose only. The joint self-test prints on every start, before anything
moves, and the simulator loads before control is claimed so nothing is held while SAPIEN starts.

## Checking the joint mapping

Same window. The arm fields are sim units, and values leave through the same `bridge.sim_to_real`
the policy uses.

- press **go to rest**. If the real arm settles into the posture the SIM panel shows, that joint's
  calibrated zero is the URDF zero.
- drive one joint to each end. The slider ends are the URDF limits, so the real joint should hit
  its physical extreme at the same moment. Early stop or wrong direction is a `SIGN` problem, which
  no offset can fix.

`--bridge` lists the offsets and sign flips currently applied. `wrist_roll` carries a measured 90°
offset; everything else is identity.

The bottom right also shows achieved-vs-commanded joint speed. If the measured real maximum stays well under the
sim-commanded rate, the hardware cannot keep up with the trained speed.
