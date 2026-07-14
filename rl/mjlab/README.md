# SO-101-on-frame RL: pick a cube, place it in a bin

A [mjlab](https://github.com/mujocolab/mjlab) reinforcement-learning task where the
SO-101-on-frame arm picks up a cube and drops it into a bin. Cube and bin start at
randomized positions on the workspace each episode.

The robot model comes from `../../simulation/mjcf/so101_on_frame.xml`, imported with two
small edits: a `grasp_site` on the gripper, and collision boxes on the lightbox's panels
(otherwise visual-only).

## Requirements

- Python `>=3.10,<3.14` (pinned to 3.13 via `.python-version`).
- For real training: a **Linux box with an NVIDIA GPU** (mjlab runs on `mujoco-warp`,
  CUDA). macOS is fine for editing and light CPU-only checks, but not for full
  GPU-parallel training.

## Setup (uv)

This is a [uv](https://docs.astral.sh/uv/) project. From `rl/mjlab/`:

```bash
uv sync                 # resolves + locks (uv.lock) + installs into rl/mjlab/.venv
```

mjlab is pulled from a pinned git commit (see `[tool.uv.sources]` in `pyproject.toml`); it
is not yet on PyPI.

## Train / play / render

```bash
uv run soframe-train  Mjlab-Pick-Place-Bin-SO101
uv run soframe-play   Mjlab-Pick-Place-Bin-SO101 --checkpoint-file <path>
uv run soframe-render --checkpoint-file <path> --out fleet.mp4
```

The train/play wrappers `import soframe_rl` first (registering the task), then delegate to
mjlab's own `train`/`play` CLIs, so all of mjlab's flags apply.

Training parameters (parallel envs, iterations, PPO hyperparameters, network size) live in
**[`train.toml`](train.toml)**. Edit that file rather than the Python. Any value can still
be overridden per run on the CLI (e.g. `--env.scene.num-envs 2048`).

### Pretrained checkpoint

A trained policy ships in **[`checkpoints/model_best.pt`](checkpoints/model_best.pt)**
(~4 MB): **55.6% place-success at 94% workspace randomization**, trained with the full
recipe described below. Try it without training anything:

```bash
# interactive viewer
uv run soframe-play Mjlab-Pick-Place-Bin-SO101 --checkpoint-file checkpoints/model_best.pt

# fleet video (one arm -> endless field); vary --seed for different rollouts
uv run soframe-render --checkpoint-file checkpoints/model_best.pt --seed 0 --out fleet.mp4
```

Both run best on GPU but also work on CPU (slowly; pass `--device cpu` to
`soframe-render`).

### Fleet render

`soframe-render` rolls out a trained checkpoint across many environments (default 400) and
renders a single camera move: it starts behind one arm at its lightbox, then eases back to a
low, wide shot where the grid of rigs overflows the frame and reads as an endless field.
Episodes auto-reset continuously so every arm stays active for the whole clip. Framing is
tunable via `--azimuth`, `--elevation-close/-wide`, `--wide-dist-frac`; runs on CPU or GPU
(`--device`).

## Layout

```
rl/mjlab/
├── pyproject.toml            uv project; mjlab dep + console scripts
├── .python-version           pinned to 3.13
├── train.toml                training params (edit this to tune runs)
├── checkpoints/model_best.pt pretrained policy (55.6% success @ 94% randomization)
└── src/soframe_rl/
    ├── __init__.py           imports config -> registers the task
    ├── train.py / play.py    thin wrappers around mjlab's CLIs
    ├── render.py             fleet render: one arm -> endless field (soframe-render)
    ├── train_params.py       loads train.toml
    ├── so101_constants.py    robot EntityCfg: loads the imported XML, re-declares
    │                         actuators/gains, home keyframe, action scale
    ├── assets.py             get_cube_spec (free box) + get_bin_spec (fixed tray)
    ├── pick_place_env_cfg.py robot-agnostic manager wiring (obs/act/reward/etc.)
    ├── mdp/
    │   ├── commands.py       PlaceInBinCommand: spread-scaled cube+bin placement,
    │   │                     goal inside the bin, carry-progress tracking
    │   ├── rewards.py        grasp_lift, potential-based transport, in_bin_bonus
    │   ├── curriculums.py    command_spread_curriculum (ADR: success-gated spread)
    │   └── observations.py   cube_to_goal_distance (command-agnostic)
    └── config/
        ├── env_cfg.py        SO-101 specialization (entities, scale, grasp site)
        ├── rl_cfg.py         PPO (rsl-rl) runner cfg, built from train.toml
        └── __init__.py       register_mjlab_task(...)
```

## How it works

mjlab uses Isaac Lab's **manager-based** design: an environment is a **Scene** (the
physical entities) plus a set of **Managers** that read observations, apply actions,
compute rewards, fire resets, and advance a curriculum every step. All of it runs batched
over `num_envs` on the GPU via `mujoco-warp`.

The scene has three entities:

- **`robot`**: the MJCF model, loaded with its own XML actuators stripped and
  re-declared in Python as `BuiltinPositionActuatorCfg`s, so stiffness/damping/effort/
  armature are first-class, tunable, sim-to-real knobs rather than buried in XML. It's a
  fixed-base entity, so mjlab wraps it in a mocap body that also carries the frame's
  loose geoms, keeping the whole rig positioned correctly in every parallel env.
- **`cube`**: a free (`freejoint`) 2.5 cm, 30 g box with high tangential friction so the
  gripper can hold it.
- **`bin`**: a base plate plus four walls. No freejoint, so it's teleported to a new spot
  each episode rather than physically knocked around.

The rig sits at a height where the frame's legs rest on the ground plane and the lightbox's
bottom panel sits at the workspace surface height, with the cube and bin placed on that
panel in the arm's reach. Physics runs at `timestep=0.005` with `decimation=4` (policy acts
at 50 Hz).

### Managers

| Manager | What ours does |
|---|---|
| **Observation** | Two groups: `actor` (27-dim, with input noise) and `critic` (same, no noise). Joint pos (7) + joint vel (7) + `ee_to_cube` (3) + `cube_to_goal` (3) + last action (7). |
| **Action** | One `JointPositionActionCfg` over all 7 actuators (slider + 5 arm + gripper): delta position targets, per-actuator scale, added to the current pose and clipped to joint limits. |
| **Command** | `PlaceInBinCommand`. On each episode reset it samples a bin and cube position (keeping them apart), teleports both, and publishes a target point just above the bin opening. |
| **Event** | Resets the robot to its home pose plus a small jitter. Domain-randomization events are a TODO. |
| **Reward** | The staged terms below. |
| **Termination** | Time-out only (episode length cap); no early failure termination. |
| **Curriculum** | Ramps the placement spread and the joint-velocity penalty over training. |

### Reward

A **staged decomposition** (reach → grasp/lift → carry → place → in-bin), each stage a
dense term plus a milestone bonus:

| Term | Purpose |
|---|---|
| `reach_and_bring` | Reach the cube; once reached, also rewards closing on the target. |
| `grasp_lift` | Rewards lifting the cube only while the gripper is actually on it, so it shapes a real grasp. |
| `transport` | Potential-based carry: rewards only the per-step reduction in the lifted cube's distance to the bin, so it can't be farmed by hovering. |
| `place_precise` | Sharpens placement once the cube is near the target. |
| `in_bin_bonus` | Milestone bonus for landing the cube in the bin. |
| `action_rate_l2` | Penalizes jerky action changes. |
| `joint_pos_limits` | Keeps the arm off its joint end-stops. |
| `joint_vel_hinge` | Penalizes joint speeds above a threshold; curriculum-ramped. |

**Success** (a metric, not a reward) means the cube's center is inside the bin footprint,
below the rim.

### Curriculum

Two terms ramp with training progress:

- **Placement spread**: the key one. The cube/bin layout scales from fixed and easy
  (a short hop between them) to full workspace randomization. It's performance-gated: a
  smoothed success rate drives it, so it holds the easy layout until the policy can place,
  then widens randomization as fast as the policy keeps up, and backs off if it starts
  failing.
- **Smoothness**: the joint-velocity penalty ramps up linearly over training, so the
  policy can explore freely early on and is pushed toward smooth, hardware-safe motion
  later. The ramp is gradual and capped, since too sharp or too large a penalty can make
  the policy stop moving altogether.

### Configs

- **`train.toml`**: the values you tune per run: `[runner]` (parallel envs, iterations,
  rollout length, save interval, experiment name), `[algorithm]` (PPO hyperparameters),
  `[policy]` (MLP width/activation/init std).
- **Python cfg modules**: `pick_place_env_cfg.py` wires the managers; `config/env_cfg.py`
  fills in the SO-101 specifics; `so101_constants.py` and `assets.py` define the entities;
  `config/__init__.py` registers the task.

## TODO

Deliberately not in this scaffold yet:

- [ ] Name the two gripper collision geoms in the XML → enables fingertip-friction
      domain randomization and contact-based grasp detection/reward.
- [ ] Add domain randomization events (friction, mass, actuator gains, latency,
      observation noise) for sim-to-real.
- [ ] Add a vision observation group using the existing wrist/overhead cameras.
