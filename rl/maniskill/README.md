# SO-101-on-frame RL: pick a bar, place it in a bin (vision, Squint)

A [ManiSkill3](https://github.com/haosulab/ManiSkill) reinforcement-learning task where the
SO-101-on-frame arm picks up a flat bar and drops it into a bin, trained **from the frame's
own wrist and overhead cameras**. There's no ground-truth bar/bin pose anywhere in the
observation, just RGB pixels and proprioception. Bar and bin start at randomized positions
each episode.

It implements

> **[Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics](https://arxiv.org/abs/2602.21203)**
> by Abdulaziz Almuzairee & Henrik I. Christensen (UC San Diego), 2026.

Squint is a visual Soft Actor-Critic that combines massively parallel GPU simulation, a
distributional (C51) critic, low-resolution "squinted" observations, layer norm throughout,
and a tuned update-to-data ratio to train vision policies in **minutes**, not hours. The
paper's own task set (the "SO-101 Task Set") already targets an SO-101 arm in ManiSkill3, so
this folder is a direct port of the authors' own
[reference implementation](https://github.com/aalmuzairee/squint). The SAC agent is vendored
untouched; the training script carries a few local additions (warm-start handling, color and
visual-fidelity and smoothness flags, best-checkpoint tracking), listed in its module
docstring.

## Requirements

- Python `>=3.10,<3.13` (pinned to 3.10 via `.python-version`, matching Squint's own tested
  environment).
- **A Linux box with an NVIDIA GPU.** ManiSkill3 runs on SAPIEN/PhysX with CUDA, and the
  visual SAC agent leans on `torch.compile` and CUDA graphs. macOS is fine for reading and
  editing code but not for running the sim or training.

## Setup (uv)

This is a [uv](https://docs.astral.sh/uv/) project. From `rl/maniskill/`:

```bash
uv sync                                 # resolves + installs mani_skill3, torch, this package
```

## Visualize / train

```bash
uv run python examples/visualize_sim.py                       # sanity-check scene + camera
uv run python train_squint.py --env_id=SOFramePickPlaceBin-v1
```

`examples/visualize_sim.py` steps a batch of environments through a scripted rail + gripper
motion and shows both camera observations next to a third-person render (pass `--headless`
to write PNG frames instead of opening a window, and read the printed item/bin/tcp
positions for a numeric reach check). It's worth running before spending GPU time on
training, since it's the fastest way to catch a broken camera mount, a bar or bin spawning
out of reach, or the arm loading into a bad pose.

`train_squint.py` and `utils.py` are vendored from the paper authors' own code
([github.com/aalmuzairee/squint](https://github.com/aalmuzairee/squint)). The agent itself
(CNN encoder, distributional critic ensemble, resolution downsampling, layer norm,
`torch.compile` and CUDA graphs) is untouched, and all of its flags apply; the local
additions on top are listed in `train_squint.py`'s module docstring. The useful flags:

```bash
uv run python train_squint.py \
    --env_id=SOFramePickPlaceBin-v1 \
    --num_envs=1024 \
    --total_timesteps=1_500_000 \
    --track --wandb_entity=YOUR_WANDB_USERNAME
```

At the paper's own settings (1024 parallel envs, 1.5M timesteps) their pick-and-place task
converges in roughly 15 minutes on a single RTX 3090. Expect a similar order of magnitude
here, adjustable via `--num_envs`/`--total_timesteps`.

## Rendering a trained checkpoint

`examples/render_chained_eval.py` renders a checkpoint's rollouts as one continuous video:
episodes play back-to-back with no gaps (single env, wrist + overhead cameras only, no
third-person view, no multi-env tiling). An episode cuts to a fresh one immediately on
success; one that hasn't succeeded gets at least `--fail_seconds` of runway before cutting,
so failed attempts aren't shown too briefly.

```bash
# Fast rasterizer (matches what the policy actually saw during training)
uv run python examples/render_chained_eval.py \
    --checkpoint checkpoints/model_best.pt \
    --num_episodes 10 --fail_seconds 5 --out /tmp/rollout.mp4

# Ray-traced shading, real PBR textures, overhead softbox lighting -- for a nicer-looking
# render, never used for training. Single-env only: ray tracing isn't supported on the
# gpu-parallelized sensor camera path, so this drops the sim backend to cpu.
uv run python examples/render_chained_eval.py \
    --checkpoint checkpoints/model_best.pt \
    --raytraced --render_size 512 \
    --num_episodes 5 --fail_seconds 5 --out /tmp/rollout_realistic.mp4
```

**`--target_image_size` must match whatever `--image_size` the checkpoint was actually
trained with** (`checkpoints/model_best.pt` was trained at the default 32, so the default
works for it). The shipped checkpoint reaches **0.94 success_at_end** on the strict metric
(bar settled inside the bin, robot clear and static) with fixed colors and full
lighting/gripper/camera randomization. The render resolution
(`--render_size`) is independent of this -- `DeployAgent` downsamples internally before
feeding the policy, so you can render at high resolution for visual quality while the policy
still sees exactly what it was trained on.

For a single high-fidelity still image (not a rollout) with a scripted open/close gripper
motion instead of a trained policy, see `examples/render_realistic.py`.

## Layout

```
rl/maniskill/
├── pyproject.toml                     uv project; ManiSkill3 + torch/torchrl/tensordict deps
├── train_squint.py                    Squint's SAC training script (vendored, local additions listed in its docstring)
├── utils.py                           Squint's obs wrappers (resolution downsample/"squint", color jitter)
├── examples/
│   ├── visualize_sim.py               scripted sanity check (windowed or --headless PNG frames)
│   ├── render_chained_eval.py         continuous rollout video from a checkpoint (see below)
│   └── render_realistic.py            single ray-traced high-fidelity still
└── src/soframe_rl_maniskill/
    ├── robot/so101_on_frame.py        ManiSkill agent: this repo's frame-mounted SO-101
    └── envs/
        ├── base_random_env.py         domain randomization + dual-camera (wrist/overhead) base
        ├── black_overlay.png          plain black sim-to-real background overlay
        └── pick_place.py              SOFramePickPlaceBin-v1 task (bar + bin, vision obs)
```

## How it works

### Scene

A plain `_load_scene` method builds the scene directly: a ground plane (`build_ground`) plus
the `so101_on_frame` agent, which loads this repo's own
`simulation/urdf/so101_on_frame.urdf` **unmodified** as a ManiSkill `BaseAgent`
(`robot/so101_on_frame.py`). Since the frame, rail, and arm are already one URDF tree, there's
no separate "table" to build; the frame *is* the scene. The flat bar (7.5 × 2.5 × 1.5 cm) and
bin (8.5 cm footprint) are added as plain per-env actors in `pick_place.py`. The work-surface
height and spawn regions were measured directly from the loaded articulation plus the panel
mesh's STL bounds (measurement script in git history).

The agent adds a rail joint (`dof_slider`) as a 7th controllable DOF on top of Squint's
original 6 (5 arm joints plus gripper), and computes the end-effector (`tcp`) point from a
fixed offset off `gripper_link`, since this URDF has no separate fingertip links the way
Squint's own robot description does.

### Observations: vision, not state

| Component | Source |
|---|---|
| `rgb` | The URDF's already-calibrated `frame_wrist_camera` and `frame_overhead_camera` mounts (see `simulation/urdf/README.md`), rendered at `--render_size` (default 128×128) then downsampled to `--image_size` (default 32×32): Squint's "resolution squinting". The two cameras' images are stacked along the channel axis (H×W×6). |
| `state` | Proprioception only: joint positions (`noisy_qpos`, with sim2real noise) plus controller state. **No ground-truth bar/bin poses.** |

This is Squint's own default too. Its `obs_mode="rgb+segmentation"` has no `state` component
in ManiSkill's sense, so `pick_place.py`'s `_get_obs_extra` (which *would* add ground-truth
item/bin poses, item dimensions, friction, and so on) only turns on if a caller explicitly
requests an obs_mode ending in `+state`. That's useful for offline eval and debugging, never
for training: the policy never sees privileged object state, only what a real camera and the
robot's own joint encoders would give it.

### Actions

`pd_joint_target_delta_pos` control (from `so101_on_frame.py`'s controller configs): the
policy outputs a per-joint delta target added to the current joint targets, clipped to
per-joint step limits (0.02 m for the rail, 0.1 rad for the arm joints, 0.2 rad for the
gripper). It's the same "fast movement" delta scheme Squint uses for its own SO-101, just
sized for this repo's slower rail joint too.

### Reward (dense, staged)

Ported from Squint's `envs/place.py` with the same structure: a staged reach, grasp, place,
success decomposition, gated by contact and pose checks in `evaluate()`.

| Stage (gate) | Reward | Purpose |
|---|---|---|
| Always | `2·(1 − tanh(5·d(tcp, bar)))` | Reach toward the bar. |
| `is_item_grasped` | `3 + place_reward` | Once grasped, reward closing the bar-to-bin distance (`place_reward` combines an overall distance term with a separate XY/Z-gated term that rewards lifting above the bin rim before descending into it; the carry target sits just above the rim, since the jaw can't open inside the bin). |
| `is_item_above_bin` | `4 + place_reward + dropped + gripper_openness + static_bonus` | Once positioned over the bin, reward releasing the bar and coming to rest. The `static_bonus` only pays out after the bar is actually released, and success pays over twice this stage's ceiling: both are deliberate, since an earlier reward where holding-forever nearly matched success got exploited exactly that way. |
| `success` | `20` | Bar settled at rest inside the bin, ungrasped, robot static and clear of the bin. |
| Penalties | `−1` hovering over the bin while still gripping, `−0.5` touching the bin, `−1` while the bar sits on the work surface | Make releasing (not hovering) the only way to earn more, keep wall contact affordable (the fingers must enter the opening to place), and encourage picking up quickly. |

`compute_normalized_dense_reward` divides by 20 (the success reward) for scale-invariant
logging, matching Squint's convention.

### Domain randomization

`envs/base_random_env.py` plus `pick_place.py`'s `PickPlaceRandomizationConfig`, off by
default (`domain_randomization=False`; `train_squint.py` turns it on via
`--env_domain_randomization`):

| Randomized | Range / behavior |
|---|---|
| Gripper stiffness / damping | Per-episode, `(500, 2000)` / `(50, 200)` |
| Ambient lighting | Per-env ambient color in `(0.2, 0.5)` |
| Bar / bin color | Off by default (fixed blue bar, dark-yellow bin); `--randomize_colors` opts in to per-scene-build random RGB with a visibility floor against the near-white work surface |
| Wrist + overhead camera pose | Small per-step jitter (±2 mm, ±1°) on top of the URDF's calibrated mount poses |
| Wrist + overhead camera FOV | ±1° per episode |
| Initial joint pose | Gaussian noise, configurable std (`initial_qpos_noise_scale`, `robot_qpos_noise_std`) |
| Bar/bin friction, density | Per-episode uniform ranges |
| Background | Greenscreen overlay compositing the sim background out (`black_overlay.png` by default; swap in a photo of your own table for a closer sim-to-real match) |

### Training algorithm (Squint)

| Component | Setting |
|---|---|
| Algorithm | Soft Actor-Critic, off-policy, `num_envs` parallel collection plus `num_updates` gradient steps per env step |
| Critic | Distributional C51 ensemble (`num_q=2`, `num_atoms=101`, support `[-20, 20]`), layer-normed MLP over a shared image+state projection |
| Actor | Tanh-squashed Gaussian over the same projection, layer-normed MLP |
| Vision | Small CNN encoder (kernel/stride tuned to `--image_size`) over the downsampled ("squinted") wrist + overhead images |
| Speed | `torch.compile` and CUDA graphs on the update/rollout functions, automatic entropy tuning |
| Defaults | 1024 parallel envs, 1.5M total timesteps, batch size 512, 256 updates per env step |

## Caveats / TODO

- **Joint order assumption.** `SO101OnFrame.keyframes` and the reward's gripper-openness
  lookup assume SAPIEN's URDF loader orders `robot.active_joints` as
  `[dof_slider, shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
  (rail, then arm, then gripper, following the kinematic tree). The controller configs
  themselves don't depend on this order (they're built by joint name), but the keyframe
  `qpos` arrays do. Verify the arm spawns in a sane pose with `examples/visualize_sim.py`
  before trusting it.
- **Workspace spawn region** (`WORK_SURFACE_Z`, `ITEM_SPAWN_CENTER`/`BIN_SPAWN_CENTER`,
  `SPAWN_HALF_SIZE` in `pick_place.py`) is tuned for this physical rig's lightbox surface.
  Worth reverifying the bar and bin actually land on that surface, in reach, in the viewer,
  if the URDF geometry ever changes.
- **No real-robot deployment script yet.** Squint's own `deploy.py` and `deploy_utils/`
  (camera calibration, hardware interface) weren't ported; this folder covers simulation
  training only so far.

## References

- [Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics](https://arxiv.org/abs/2602.21203)
  (Almuzairee & Christensen, 2026): the method implemented here.
- [github.com/aalmuzairee/squint](https://github.com/aalmuzairee/squint): the reference
  implementation `train_squint.py`/`utils.py` are vendored from.
- [ManiSkill3](https://github.com/haosulab/ManiSkill): the simulator.
