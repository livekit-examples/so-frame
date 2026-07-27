# rl

Train a policy in simulation, then run it on the real rig.

```
environments/
  maniskill/   vision-based RL (ManiSkill3 + SAPIEN). The one that feeds deploy.
  mjlab/       state-based RL (mjlab + MuJoCo-Warp). Same task, ground-truth poses.
policy/        the encoder, actor and checkpoint format -- everything IN a checkpoint
deploy/        run a trained checkpoint on the physical robot, over LiveKit Portal
```

Each directory is its own [uv](https://docs.astral.sh/uv/) project; `cd` into one and `uv sync`.

## Why policy/ is separate

`environments/maniskill` writes checkpoints and `deploy` reads them, so whatever is inside a
checkpoint has to be defined once, in a package both can depend on. Both take `policy/` as a path
dependency. It is pure `torch` + `numpy` — no simulator, no LiveKit — so it installs on the robot
as well as the GPU box.

That is also where the joint contract lives (`policy/src/soframe_policy/rig.py`): joint order,
limits, per-step motion caps, control rate, rest pose. Sim and the robot must agree on those, so
they have one home.

## Start here

```bash
cd environments/maniskill && uv sync
uv run python examples/visualize_sim.py   # look at the scene
uv run python train.py                    # train
```

Then [`deploy/README.md`](deploy/README.md) to put a checkpoint on the robot.
