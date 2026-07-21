# policy

The trained Squint RL policy as a Portal `operator`. Idle until it claims
control; then, each tick, it turns the robot's observation into the exact obs it
saw in sim, runs one inference, and sends joint targets back.

```bash
# from sim2real/
uv run policy-squint --checkpoint /path/to/ckpt_best.pt
# default checkpoint: ../rl/maniskill/runs/v31_clean_recipe/ckpt_best.pt
```

## Per-tick pipeline

```
robot obs: 7-DOF state (real units) + 2 RAW camera frames (640x480, wide FOV)
  state:  bridge.real_to_sim        # deg/mm -> rad/m (sim units)
  frames: camera_mapping.apply_mapping(per camera)   # rectify to the 128x128 sim view
          stack wrist(0:3)|overhead(3:6)              # -> 128x128x6 (sim RGB obs)
  agent:  SquintPolicy.act(rgb6, state14)             # squint to 32, encoder+actor
          -> normalized delta action a in [-1, 1]^7
  step:   sim_target += a * DELTA_LIMIT ; clamp to joint limits    # the sim delta controller
          bridge.sim_to_real(sim_target)              # rad/m -> deg/mm
  send:   op.send_action(real_target)
```

Three things must match the training setup exactly, or the policy sees
out-of-distribution input:

1. **camera mapping** -- `apply_mapping` reconstructs the sim FOV/undistortion.
   `camera_mappings/overhead_camera_mapping.json` is calibrated; add
   `camera_mappings/arm_camera_mapping.json` for the wrist camera (calibrate with
   `rl/maniskill/examples/calibrate_real_camera.py`). A missing mapping falls
   back to a plain resize and is logged loudly.
2. **channel order** -- wrist first, overhead second (`CAMERA_STACK`), matching
   the sim's sensor order.
3. **proprio state** -- `qpos(7)` then the controller `target(7)` = 14 dims, in
   sim units. `target` is this loop's running integrated target (seeded from the
   measured pose on claim), which is what `pd_joint_target_delta_pos` tracks in
   sim.

## Files

- `agent.py` -- loads `CNNEncoder`+`Actor` from the checkpoint. Those two
  classes are vendored (plain torch, no mani_skill) into `nets.py`, copied from
  `rl/maniskill/train_squint.py`; re-copy them if the training nets change.
- `nets.py` -- the vendored `CNNEncoder`/`Actor` network definitions.
- `bridge.py` -- sim<->real unit/coordinate bridge. **`NEEDS-CALIBRATION`**: the
  offsets default to zero (real zero assumed == sim zero). Measure per joint.
- `camera_mapping.py` -- vendored `apply_mapping`/`load_mapping` (kept in sync
  with the calibration tool).

## Notes vs the vla-demo policies

This policy is **per-tick continuous** (it replans every tick), so there is no
chunk queue, no settle gate, and no language conditioning -- simpler than the
chunked lerobot VLAs in `livekit-actuate/vla-demo`. It also emits *normalized
delta* actions that this loop integrates, rather than absolute targets straight
from the model.
