# SO-101-on-frame RL: pick a cube, place it in a bin

A [mjlab](https://github.com/mujocolab/mjlab) reinforcement-learning task where the
SO-101-on-frame arm picks up a cube and drops it into a bin. Cube and bin are repositioned
every episode, starting from a fixed nominal layout and widening toward full workspace
randomization as the curriculum advances.

The robot model comes from `../../../simulation/mjcf/so101_on_frame.xml`, imported with two
small edits: a `grasp_site` on the gripper, and collision boxes on the lightbox's panels
(otherwise visual-only).

## Requirements

- Python `>=3.10,<3.14` (pinned to 3.13 via `.python-version`).
- For real training: a **Linux box with an NVIDIA GPU** (mjlab runs on `mujoco-warp`,
  CUDA). macOS is fine for editing and light CPU-only checks, but not for full
  GPU-parallel training.

## Setup (uv)

This is a [uv](https://docs.astral.sh/uv/) project. From `rl/environments/mjlab/`:

```bash
uv sync                 # resolves + locks (uv.lock) + installs into rl/environments/mjlab/.venv
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

No pretrained checkpoint ships with this task: training starts from scratch. `soframe-play`
and `soframe-render` both need a `--checkpoint-file` from a run of your own.

### Fleet render

`soframe-render` rolls out a trained checkpoint across many environments (default 400) and
renders a single camera move: it starts behind one arm at its lightbox, then eases back to a
low, wide shot where the grid of rigs overflows the frame and reads as an endless field.
Episodes auto-reset continuously (`--episode-seconds`, default 6) so every arm stays active
for the whole clip. Framing is tunable via `--azimuth`, `--elevation-close/-wide`,
`--wide-dist-frac`; `--seed` varies the layouts and rollouts. It defaults to `cuda:0` and
also runs on CPU, slowly, via `--device cpu`.

## Layout

```
rl/environments/mjlab/
├── pyproject.toml            uv project; mjlab dep + console scripts
├── .python-version           pinned to 3.13
├── train.toml                training params (edit this to tune runs)
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
    │   ├── actions.py        TargetRelativeJointPositionAction: delta from the
    │   │                     previous target, clamped to the soft joint limits
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
panel in the arm's reach. Physics runs at `timestep=0.005` with `decimation=20`, so the
policy acts at 10 Hz, matching `so101_constants.CONTROL_HZ`, the maniskill twin, and the
deploy loop's 10 Hz tick. The action cadence is part of the sim-to-real contract, so it is
not a free knob. Episodes are `episode_length_s=30.0`, i.e. 300 decisions, which is the
runway the real-servo speeds need to reach, carry and place.

### Managers

| Manager | What ours does |
|---|---|
| **Observation** | Two groups: `actor` (27-dim, with input noise) and `critic` (same, no noise). Joint pos (7) + joint vel (7) + `ee_to_cube` (3) + `cube_to_goal` (3) + last action (7). |
| **Action** | One `TargetRelativeJointPositionActionCfg` (ours, `mdp/actions.py`) over all 7 actuators (slider + 5 arm + gripper): `target = previous_target + action * scale`, clamped to the soft joint limits. Integrating from the previous target rather than the measured pose is what the maniskill twin and the deploy loop do; delta-from-current stalls under load. |
| **Command** | `PlaceInBinCommand`. On each episode reset it samples a bin and cube position (keeping them apart), teleports both, and publishes a target at the cube's resting place on the bin floor, so putting the cube in *increases* the place reward. |
| **Event** | Resets the robot to its home pose plus a small joint jitter (+/- 0.05). Domain-randomization events are a TODO. |
| **Reward** | The staged terms below. |
| **Termination** | Time-out only (episode length cap); no early failure termination. |
| **Curriculum** | One term: it widens the placement spread as success allows. |

The action `scale` is per actuator and is the only thing that enforces real speed in sim: it
is `SO101_ACTION_SCALE`, the joint speeds measured on the real rig divided by `CONTROL_HZ`
(arm 0.5 rad/s, rail 0.07 m/s, gripper 2.0 rad/s, deliberately quicker so the jaw can open
and close within a reach). Effort limits bound torque, not velocity, so they cannot do this
job. Because `scale` is a hard per-step cap, the rate limit is structural.

### Reward

A **staged decomposition** (reach → grasp/lift → carry → place → in-bin), each stage a
dense term plus a milestone bonus:

| Term | Weight | Purpose |
|---|---|---|
| `reach_and_bring` | 1.0 | Reach the cube; once reached, also rewards closing on the target. |
| `grasp_lift` | 15.0 | Rewards lifting the cube only while the gripper is actually on it, so it shapes a real grasp. |
| `transport` | 40.0 | Potential-based carry: rewards only the per-step reduction in the lifted cube's distance to the bin, so it can't be farmed by hovering. Gated on the cube clearing ~rim height. |
| `place_precise` | 5.0 | Sharpens placement once the cube is near the target. |
| `in_bin_bonus` | 10.0 | Milestone bonus for landing the cube in the bin. |
| `joint_pos_limits` | -10.0 | Keeps the arm off its joint end-stops. |

`joint_pos_limits` is the only penalty. There is no action-rate or joint-velocity term: the
action space's per-step cap already limits speed and jerk structurally, so there is no
penalty weight to tune against the task rewards. The weights above have not been swept, they
are the values the current wiring uses.

**Success** (a metric, not a reward) means the cube's center is inside the bin footprint,
below the rim.

### Curriculum

One term, `placement_spread`, ramps with training progress. The cube/bin layout scales from
fixed and easy (a short hop between them) to full workspace randomization. It's
performance-gated: a smoothed success rate (EMA, `alpha=0.01`) drives it, raising spread
above 0.4 success and lowering it below 0.2, by `1e-4` per env step. So it holds the easy
layout until the policy can place, then widens randomization as fast as the policy keeps up,
and backs off if it starts failing. `soframe-play` disables the curriculum and pins
`initial_spread = 1.0`, i.e. it always shows the hard, fully randomized case.

### Configs

- **`train.toml`**: the values you tune per run: `[runner]` (parallel envs, iterations,
  rollout length, save interval, experiment name), `[algorithm]` (PPO hyperparameters),
  `[policy]` (MLP width/activation/init std).
- **Python cfg modules**: `pick_place_env_cfg.py` wires the managers; `config/env_cfg.py`
  fills in the SO-101 specifics; `so101_constants.py` and `assets.py` define the entities;
  `config/__init__.py` registers the task.

### What is measured and what is a guess

Be honest about which numbers you can lean on:

- **Measured**: the per-joint speeds behind `SO101_ACTION_SCALE` and `CONTROL_HZ`. These are a
  hand-maintained copy of `soframe_policy.rig.REAL_JOINT_SPEED` / `rig.CONTROL_HZ`, which is
  authoritative because deploy shares it. This project has its own lockfile and cannot import
  `soframe_policy`, so edit `rig.py` first and mirror the change here.
- **Functional but not calibrated**: the PD gains in `so101_constants.py`. They mirror the
  generic values in the XML's `<default>` classes, not measured STS3215 (arm/gripper) or
  rail-drive (slider) parameters. The effort limits do track `rig.JOINT_FORCE_LIMITS` (3 N.m
  arm, 100 N rail), but the rail figure is itself functional rather than measured.
- **Estimates**: the workspace bounds. `workspace_x`/`workspace_y` on
  `PlaceInBinCommandCfg` (`mdp/commands.py`) carry an explicit warning: they have not been
  checked for reachability in the viewer. The same goes for the nominals and `spread_xy` next
  to them, and for the gripper's open/closed sign in `HOME_KEYFRAME`.

There is no domain randomization yet, so nothing here has been stress-tested for
sim-to-real transfer.

## TODO

Deliberately not in this scaffold yet:

- [ ] Name the two gripper collision geoms in the XML → enables fingertip-friction
      domain randomization and contact-based grasp detection/reward.
- [ ] Add domain randomization events (friction, mass, actuator gains, latency,
      observation noise) for sim-to-real.
- [ ] Add a vision observation group using the existing wrist/overhead cameras.
