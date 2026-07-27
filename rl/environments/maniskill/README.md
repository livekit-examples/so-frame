# SO-101-on-frame RL: pick a cube, place it in a bin

Vision-based RL on [ManiSkill3](https://github.com/haosulab/ManiSkill). The policy sees only the
frame's wrist and overhead cameras plus proprioception — no ground-truth object poses. Cube and
bin spawn anywhere in the workspace each episode.

Implements [Squint: Fast Visual Reinforcement Learning for Sim-to-Real
Robotics](https://arxiv.org/abs/2602.21203) (Almuzairee & Christensen, 2026), a visual SAC with a
distributional C51 critic, massively parallel envs, and low-resolution "squinted" observations.

## Requirements

Python `>=3.10,<3.13` (pinned to 3.10 via `.python-version`), and **a Linux box with an NVIDIA
GPU** for training. macOS can run the CPU backend for editing and smoke tests
(`--sim_backend cpu --num_envs 1 --num_eval_envs 1`) but not real training.

## Run

```bash
uv sync
uv run python examples/visualize_sim.py        # look at the scene before spending GPU time
uv run python train.py                         # squint CNN
uv run python train.py --encoder dino_patch    # frozen DINOv2 + patch head
uv run python train.py --help                  # every flag, documented
```

Checkpoints land in `runs/<exp_name>/`: `ckpt.pt` (latest) and `ckpt_best.pt` (best eval). Both
record their own architecture, so deploying needs no flags.

Useful flags: `--num_envs`, `--total_timesteps`, `--exp_name`, `--track` (wandb),
`--checkpoint <path>` to warm-start. Sim2real knobs: `--overhead_camera_fov` /
`--wrist_camera_fov` and `--overhead_camera_pos_offset` / `--overhead_camera_rot_offset` to
re-calibrate a camera without editing `config.py`, `--arm_speed_scale` for arm-speed ablations,
`--binary_gripper` to force the jaw fully open/closed each step.

## Where things are

```
train.py                     entry point
config.py                    task, robot, reward, cost and colour constants  <- edit here
wrappers.py                  observation pipeline (downsample, jitter, sensor aug, DINOv2 tokens)
envs/base_random_env.py      domain randomization, lighting, URDF-mounted cameras
envs/pick_place.py           the task: scene, spawn, success, reward
robot/so101_on_frame.py      the agent (arm + rail as one 7-DOF robot), materials/colour pass
sac/                         Args, env construction, critic, logging, training loop
examples/                    visualize_sim, dump_reference_views, render_realistic,
                             render_chained_eval
```

The encoder, actor and checkpoint format live in [`policy/`](../../policy/README.md), shared with
deploy so the network that runs on the robot is the one that trained.

## Task

Reward is a monotonic ladder — each stage sits above the previous stage's maximum, so progress is
monotone and a regression falls to a lower rung on its own:

```
reach [0,1] < grasped [2,3] < holding [4,5] < released 6 < success 10
```

The `holding` rung rises with how far the jaw is opened over the bin (`SHAPE_HOLD_OPEN`), so
releasing is a continuous climb rather than a blind jump to the next rung. That ramp is why the
gripper action stays continuous by default; `--binary_gripper` reproduces the older binary-jaw
runs.

No penalty terms. Motion limits are structural instead: the delta action space caps per-step speed
at the measured real servo rate (0.05 rad/step arm, 0.007 m/step rail at 10 Hz) and the force
limits cap torque at the STS3215's 3 N·m stall. Nothing to trade off against the task reward.

Cube (20 mm) and bin (100 mm, 96 mm opening) come from CAD in
`simulation/assets/objects/`. Both spawn in one zone — bin first, then the cube at least 5 cm
clear of it. The zone is the measured intersection of top-down graspable reach, the overhead
camera's footprint, and where objects physically fit.

## Encoders

| `--encoder` | vision | default res |
|---|---|---|
| `squint` | CNN over a squinted image stack | 32 px |
| `dino_patch` | self-attention over frozen DINOv2 patch tokens | 168 px |

Resolution, replay size and update ratio default per encoder; see `sac/args.py`.

## Sim2real notes

- **10 Hz control**, matching the deploy loop. A policy trained at one rate and driven at another
  sees a different amount of world motion per decision.
- **Camera FOVs are calibrated** against the real rig (overhead 38°, wrist 58°). Both cameras are
  SAPIEN-mounted on the URDF's own camera links, so their poses follow the model rather than being
  rewritten per step. Deploy rectifies the real cameras to match; dump the reference renders with
  `examples/dump_reference_views.py`.
- **Domain randomization** covers arm and gripper PD gains, lighting, qpos noise, colour jitter and
  sensor-realism augmentation per episode, plus camera pose/FOV jitter drawn once per scene build
  (pass `--reconfiguration_freq` to resample it during a run). On by default.

## Cost knobs

The per-step cost lives in three places, all documented with their measurements in `config.py`:

- **Physics** — `SOLVER_POSITION_ITERATIONS` (8) / `SOLVER_VELOCITY_ITERATIONS` (0) /
  `FRICTION_EVERY_ITERATION`. Below 8 position iterations a resting cube starts to buzz.
- **Render** — `RASTER_SHADOWS` is **off**: the sim's key light cast a large directional shadow
  the real lightbox does not produce, so dropping it both narrows the sim2real gap and cuts ~30%
  off the render stage. With it off, `--visual_fidelity flat` is no longer a meaningful speed
  option (87.0 vs 87.3 ms/step against `raster`), so prefer `raster` and keep the PBR materials.
  `--visual_fidelity raytraced` is for one-off renders only and forces the CPU backend.
- **Observations** — the default `--obs_mode rgb` runs no segmentation pass. `SENSOR_FAR` (3 m)
  culls the ground plane outright, leaving the renderer's black clear colour, which is what the
  greenscreen overlay used to paint. Turn `apply_overlay` back on only to composite a real
  background photo, and then also pass `--obs_mode rgb+segmentation` to feed its mask.
