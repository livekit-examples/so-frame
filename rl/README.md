# rl

Train a policy in simulation, fit the real cameras to match what it trained on, then run it on
the real rig.

```
environments/
  maniskill/   vision-based RL (ManiSkill3 + SAPIEN). The one that feeds deploy.
  mjlab/       state-based RL (mjlab + MuJoCo-Warp). Same task, ground-truth poses.
policy/        the encoder, actor and checkpoint format -- everything IN a checkpoint
calibrate/     drive the real arm, fit the camera mappings against a live sim render
deploy/        run a trained checkpoint on the physical robot, over LiveKit Portal
```

Each directory is its own [uv](https://docs.astral.sh/uv/) project; `cd` into one and `uv sync`.
`policy/` is a path dependency of `environments/maniskill` and `deploy`, so syncing either one
installs it editable; you only sync `policy/` on its own to run its tests.

## Why policy/ is separate

`environments/maniskill` writes checkpoints and `deploy` reads them, so whatever is inside a
checkpoint has to be defined once, in a package both can depend on. Both take `policy/` as a path
dependency. It is pure `torch` + `numpy`, no simulator and no LiveKit, so it installs on the robot
as well as the GPU box.

That is also where the joint contract lives (`policy/src/soframe_policy/rig.py`): joint order,
limits, per-step motion caps, control rate, rest pose. Sim and the robot must agree on those, so
they have one home.

## Why calibrate/ is separate

Fitting a camera mapping needs the simulator and the live robot feed in one process, so
`calibrate/` path-depends on both `deploy/` (for the joint bridge and the mapping format, which
must be identical to what ships) and `environments/maniskill` (to render the sim side). It cannot
live inside `deploy/`, because putting `mani_skill` in that project's lockfile would make the
robot host resolve `sapien` on every `uv sync`, for something it never installs.

## Start here

```bash
cd environments/maniskill && uv sync
uv run python examples/visualize_sim.py   # look at the scene
uv run python train.py                    # train
```

Then [`calibrate/README.md`](calibrate/README.md) to fit the two cameras against the sim (this
moves the arm), and [`deploy/README.md`](deploy/README.md) to put a checkpoint on the robot.
