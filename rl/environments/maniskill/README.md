# SO-101-on-frame RL: pick a cube, place it in a bin

Vision-based RL on [ManiSkill3](https://github.com/haosulab/ManiSkill). The policy sees only the
frame's wrist and overhead cameras plus proprioception, with no ground-truth object poses. Cube and
bin spawn anywhere in the workspace each episode.

Implements [Squint: Fast Visual Reinforcement Learning for Sim-to-Real
Robotics](https://arxiv.org/abs/2602.21203) (Almuzairee & Christensen, 2026), a visual SAC with a
distributional C51 critic, massively parallel envs, and low-resolution "squinted" observations.

## Requirements

Python `>=3.10,<3.13`, and **a Linux box with an NVIDIA GPU** for training. macOS can run the CPU
backend for editing and smoke tests (`--sim_backend cpu --num_envs 1 --num_eval_envs 1`, the only
count that backend supports) but not real training.

## Run

```bash
uv sync
uv run python train.py                         # squint CNN
uv run python train.py --encoder dino_patch    # frozen DINOv2 + patch head
uv run python train.py --encoder dino_global   # same backbone, collapsed to one vector per camera
uv run python train.py --help                  # every flag, documented
```

Checkpoints land in `runs/<exp_name>/`: `ckpt.pt` (latest) and `ckpt_best.pt` (best eval). Both
record their own architecture, so deploying needs no flags.

Useful flags: `--num_envs` (1024), `--total_timesteps` (12M), `--exp_name`, `--track` (wandb).
`--checkpoint <path>` warm-starts, resetting the entropy temperature so fine-tuning keeps
exploring; `--no-reset_alpha` inherits the checkpoint's instead. `--visual_fidelity flat` is a
faster shadowless render, and `raytraced` forces the cpu backend, so it is eval-only.
`--object_colors distinct` paints the cube blue and the bin yellow, which the res-32 CNN needs and
sim2real fidelity pays for. `--encoder_lr` defaults to `--q_lr` and is worth lowering for the
transformer head; `--grad_clip` is off by default. `--arm_speed_scale` is for arm-speed ablations
only.

## Commands

All from `rl/environments/maniskill/`.

```bash
# look at the scene, cameras and randomization before spending GPU time
uv run python examples/visualize_sim.py
uv run python examples/visualize_sim.py --headless --out /tmp/frames   # no display (ssh)

# one-off pretty render, to look at the sim only
uv run python examples/render_realistic.py --shader rt-fast --out /tmp/realistic.png

# a trained checkpoint as one continuous video, episodes back to back
uv run python examples/render_chained_eval.py \
    --checkpoint runs/<exp>/ckpt_best.pt --num_episodes 10 --out /tmp/chained.mp4

# replay-buffer index arithmetic (CPU, seconds); host-RAM + prefetch path (needs CUDA)
uv run --with pytest pytest tests/test_replay.py -q
uv run --with pytest pytest tests/test_replay_cuda.py -q
```

## Where things are

```
train.py                      entry point
examples/                     scene check, one-off pretty render, rollout videos
tests/                        replay index arithmetic, and its CUDA-only host-RAM path
src/soframe_rl_maniskill/
  config.py                   task, robot, reward and colour constants  <- edit here
  wrappers.py                 observation pipeline (downsample, jitter, sensor aug, DINOv2 features)
  envs/base_random_env.py     domain randomization, greenscreen, camera mounts
  envs/pick_place.py          the task: scene, spawn, success, reward
  robot/so101_on_frame.py     the agent (arm + rail as one 7-DOF robot)
  sac/                        Args, env construction, replay, critic, logging, training loop
```

The encoders, actor and checkpoint format live in [`policy/`](../../policy/README.md), shared with
deploy so the network that runs on the robot is the one that trained.

## Task

Reward is a monotonic ladder: each stage sits above the previous stage's maximum, so progress is
monotone and a regression falls to a lower rung on its own:

```
reach [0,1.5] < grasped [2,3] < holding [4,5] < released 6 < success 10
```

Both jaw motions are shaped, so neither is a blind jump off a plateau: closing pays while the tool
is on the cube (top of the reach stage), opening pays while holding over the bin (top of the
holding stage). Each stays below the next rung, so a jaw that shuts without catching the cube is
worth less than a grasp, and the most-open still-holding pose is worth less than a real release.

No penalty terms. Motion limits are structural instead: the delta action space caps per-step speed
at the measured real servo rate (0.05 rad/step arm, 0.007 m/step rail at 10 Hz) and the force
limits cap torque at the STS3215's 3 N·m stall. Nothing to trade off against the task reward.

Cube (20 mm) and bin (100 mm, 96 mm opening) come from CAD in
`simulation/assets/objects/`. Both spawn in one zone: bin first, then the cube at least 5 cm
clear of it. The zone is the measured intersection of top-down graspable reach, the overhead
camera's footprint, and where objects physically fit.

## Encoders

| `--encoder` | vision | default res | renders at | updates/step |
|---|---|---|---|---|
| `squint` | CNN over a squinted image stack | 32 px | 128 px | 64 |
| `dino_patch` | self-attention over frozen DINOv2 patch tokens | 168 px | 168 px | 32 |
| `dino_global` | MLP over one frozen DINOv2 vector per camera (`--dino_pool cls\|mean\|cls_mean`) | 168 px | 168 px | 32 |

`dino_global` is the collapsed-pooling control for `dino_patch`: same frozen ViT-S/14, same
resolution and the same update ratio, so the only difference is whether the patch grid survives.
At two cameras and 168 px that is 288 tokens against 2 vectors, or 4 under `cls_mean`.

The observation pipeline follows from the encoder (`sac/build.py`):

```
squint       render at render_size -> downsample to res -> jitter -> sensor aug
dino_patch   render at render_size -> jitter -> sensor aug -> tokenize at res
dino_global  render at render_size -> jitter -> sensor aug -> pool at res
```

The frozen ViT runs in that last wrapper, once per env step rather than once per minibatch, so
what lands in the replay buffer is already encoder-ready. It has to be last: everything above it
needs pixels, and it emits features.

Resolution and update ratio default per encoder; see `sac/args.py`. The updates/step defaults are
tuned against `--batch_size 512` and 1024 envs, so scaling `--num_envs` without scaling them
changes the update-to-data ratio. Replay retention does NOT vary by encoder: it is
`--replay_episodes * config.EPISODE_HORIZON * --num_envs` for all three, so a comparison between
them is not confounded by how much history each one keeps.

## Replay

`sac/replay.py` stores each observation once and recovers `next_obs` by an index offset: the
successor of a slot is the same env one iteration later, so nothing is kept twice. That matters
most for the patch head, whose 2-camera 168 px token grid is 216 KB per observation. Slots whose
successor crosses a reset are dropped rather than bootstrapped across the seam, as is the newest
iteration, which has no successor yet.

`--replay_episodes 2` on the default 1024 envs is 2 * 200 * 1024 = 409,600 transitions. Capacity
rounds down to a whole number of iterations, and `--buffer_size` overrides the derivation.

`--replay_storage cpu` keeps the buffer in pinned host RAM, for when it will not fit in VRAM, with
`--replay_prefetch` batches in flight so the gather and the PCIe copy overlap training. It needs
`--obs_only_replay` (the default): host storage exists only in this buffer, and
`--no-obs_only_replay` falls back to the torchrl two-copy buffer, which stays in VRAM.

## Sim2real notes

- **10 Hz control**, matching the deploy loop. A policy trained at one rate and driven at another
  sees a different amount of world motion per decision.
- **The overhead camera's FOV is calibrated** against the real rig, at 38°. The wrist's 58° comes
  from the MJCF twin's `fovy` rather than a fit against the real camera. Deploy rectifies the real
  cameras to match; fit the mapping in [rl/calibrate](../../calibrate/README.md), which renders
  these cameras live beside the real ones.
- **Domain randomization** covers camera pose/FOV jitter, arm and gripper PD gains, lighting,
  qpos noise, colour jitter and sensor-realism augmentation. On by default.
