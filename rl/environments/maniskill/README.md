# SO-101-on-frame RL: pick a cube, place it in a bin

Vision-based RL on [ManiSkill3](https://github.com/haosulab/ManiSkill). The policy sees only the
frame's wrist and overhead cameras plus proprioception — no ground-truth object poses. Cube and
bin spawn anywhere in the workspace each episode.

Implements [Squint: Fast Visual Reinforcement Learning for Sim-to-Real
Robotics](https://arxiv.org/abs/2602.21203) (Almuzairee & Christensen, 2026), a visual SAC with a
distributional C51 critic, massively parallel envs, and low-resolution "squinted" observations.

## Requirements

Python `>=3.10,<3.13`, and **a Linux box with an NVIDIA GPU** for training. macOS can run the CPU
backend for editing and smoke tests (`--sim_backend cpu --num_envs 1`) but not real training.

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
`--checkpoint <path>` to warm-start, `--visual_fidelity flat` for a faster shadowless render.

## Where things are

```
train.py                     entry point
config.py                    task, robot, reward and colour constants  <- edit here
wrappers.py                  observation pipeline (downsample, jitter, sensor aug, DINOv2 tokens)
envs/base_random_env.py       domain randomization, greenscreen, camera mounts
envs/pick_place.py            the task: scene, spawn, success, reward
robot/so101_on_frame.py       the agent (arm + rail as one 7-DOF robot)
sac/                          Args, env construction, critic, logging, training loop
examples/                     scene check, reference renders for calibration, rollout videos
```

The encoder, actor and checkpoint format live in [`policy/`](../../policy/README.md), shared with
deploy so the network that runs on the robot is the one that trained.

## Task

Reward is a monotonic ladder — each stage sits above the previous stage's maximum, so progress is
monotone and a regression falls to a lower rung on its own:

```
reach [0,1] < grasped [2,3] < holding [4,5] < released 6 < success 10
```

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
- **Camera FOVs are calibrated** against the real rig (overhead 38°, wrist 58°). Deploy rectifies
  the real cameras to match; dump the reference renders with
  `examples/dump_reference_views.py`.
- **Domain randomization** covers camera pose/FOV jitter, arm and gripper PD gains, lighting,
  qpos noise, colour jitter and sensor-realism augmentation. On by default.
