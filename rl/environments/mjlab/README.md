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

No trained checkpoint ships with this directory — `soframe-play` and `soframe-render` both
need one you trained yourself. `soframe-train` writes to
`logs/rsl_rl/<experiment_name>/<timestamp>/` (`experiment_name` comes from `train.toml`).

Note that the action space and control rate were reworked after the last policy was trained
here: 10 Hz instead of 50, and delta-from-target with a real-speed per-step cap instead of an
absolute target from the home pose. Observation and action *dimensions* are unchanged, so an
older checkpoint still loads — it just means something different by every action it emits.
Retrain rather than trusting one.

### Fleet render

`soframe-render` rolls out a trained checkpoint across many environments (default 400) and
renders a single camera move: it starts behind one arm at its lightbox, then eases back to a
low, wide shot where the grid of rigs overflows the frame and reads as an endless field.
Episodes auto-reset continuously so every arm stays active for the whole clip. Framing is
tunable via `--azimuth`, `--elevation-close/-wide`, `--wide-dist-frac`; runs on CPU or GPU
(`--device`).

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
    │   ├── actions.py        TargetRelativeJointPositionAction: delta from the running
    │   │                     target, clamped to the soft joint limits
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
  loose geoms, keeping the whole rig positioned correctly in every parallel env. The arm
  and gripper are capped at the STS3215's ~3 N·m stall torque, matching the XML's
  `actuatorfrcrange` and the maniskill twin.
- **`cube`**: a free (`freejoint`) 2.5 cm, 30 g box with high tangential friction so the
  gripper can hold it. (The maniskill twin uses a 20 mm CAD cube instead; the two task
  objects are not identical.)
- **`bin`**: a base plate plus four walls, 10 cm interior. No freejoint, so it's teleported
  to a new spot each episode rather than physically knocked around.

The rig sits at a height where the frame's legs rest on the ground plane and the lightbox's
bottom panel sits at the workspace surface height, with the cube and bin placed on that
panel in the arm's reach. Physics runs at `timestep=0.005` with `decimation=20`, so the
policy acts at **10 Hz** — the same rate as `so101_constants.CONTROL_HZ`, the maniskill twin
and the deploy loop's portal tick. Episodes are 30 s (300 steps), which is the runway the
real-servo speeds need to reach, carry and place.

### Managers

| Manager | What ours does |
|---|---|
| **Observation** | Two groups: `actor` (27-dim, with input noise) and `critic` (same, no noise). Joint pos (7) + joint vel (7) + `ee_to_cube` (3) + `cube_to_goal` (3) + last action (7). |
| **Action** | One `TargetRelativeJointPositionActionCfg` (ours, `mdp/actions.py`) over all 7 actuators (slider + 5 arm + gripper): `target = clamp(previous_target + action * scale)`. Integrating from the previous *target* rather than the measured pose is what the maniskill twin and the deploy loop both do — delta-from-current can't get ahead of a lagging joint, so the target collapses onto the measured pose and the arm stalls under load. `scale` is the measured real joint speed divided by `CONTROL_HZ`, making it a hard per-step motion cap. |
| **Command** | `PlaceInBinCommand`. On each episode reset it samples a bin and cube position (keeping them apart), teleports both, and publishes a target point just above the bin opening. |
| **Event** | Resets the robot to its home pose plus a small jitter. Domain-randomization events are a TODO. |
| **Reward** | The staged terms below. |
| **Termination** | Time-out only (episode length cap); no early failure termination. |
| **Curriculum** | Ramps the placement spread over training. |

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
| `joint_pos_limits` | Keeps the arm off its joint end-stops. |

**No speed or smoothness penalties.** `action_rate_l2` and `joint_vel_hinge` are gone: they
were standing in for a rate limit the action space did not have (the hinge's `max_vel` was
0.5, literally the real arm's 0.5 rad/s). The relative action space's per-step cap enforces
the same thing structurally, with no weight to tune against the task rewards.

**Success** (a metric, not a reward) means the cube's center is inside the bin footprint,
below the rim.

### Curriculum

One term ramps with training progress:

- **Placement spread**: the cube/bin layout scales from fixed and easy (a short hop between
  them) to full workspace randomization. It's performance-gated: a smoothed success rate
  drives it, so it holds the easy layout until the policy can place, then widens
  randomization as fast as the policy keeps up, and backs off if it starts failing.

The smoothness-penalty ramp that used to sit alongside it went with the penalty it tuned.
Its own rationale was the argument against it: the weight had to be ramped linearly because
a step change could collapse a trained policy, and had to stay under a threshold or the
penalty outbid the task rewards. A per-step cap in the action space has no weight and no
window to fall out of.

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
