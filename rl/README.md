# SO-101-on-frame — RL: pick a cube, place it in a bin

A [mjlab](https://github.com/mujocolab/mjlab) reinforcement-learning task where the
SO-101-on-frame arm picks up a cube and drops it into a bin. Cube **and** bin start
at randomized positions on the workspace plane each episode.

Everything RL-specific lives in this `rl/` folder. The robot model in
`../simulation/` is **imported unmodified** — the only edit to it is a single
`grasp_site` added to `gripper_link` (the end-effector reference point).

## Requirements

- Python `>=3.10,<3.14` (mjlab constraint; pinned to 3.13 via `.python-version`).
- For real training: a **Linux box with an NVIDIA GPU** — mjlab runs on
  `mujoco-warp` (CUDA). macOS is fine for editing and light CPU-only checks, but
  not for full GPU-parallel training.

## Setup (uv)

This is a [uv](https://docs.astral.sh/uv/) project. From `rl/`:

```bash
uv sync                 # resolves + locks (uv.lock) + installs into rl/.venv
```

mjlab is pulled from a pinned git commit (see `[tool.uv.sources]` in
`pyproject.toml`); it is not yet on PyPI.

## Train / play

```bash
uv run soframe-train Mjlab-Pick-Place-Bin-SO101
uv run soframe-play  Mjlab-Pick-Place-Bin-SO101 --checkpoint-file <path>
```

Both wrappers `import soframe_rl` first (registering the task) and then delegate to
mjlab's own `train`/`play` CLIs, so all of mjlab's flags apply.

Training parameters (parallel envs, iterations, PPO hyperparameters, network size)
live in **[`train.toml`](train.toml)** — edit that file rather than the Python. Any value
can still be overridden per-run on the CLI (e.g. `--env.scene.num-envs 2048`).

## Layout

```
rl/
├── pyproject.toml            uv project; mjlab dep + console scripts
├── .python-version           pinned to 3.13
├── train.toml                training params (edit this to tune runs)
└── src/soframe_rl/
    ├── __init__.py           imports config -> registers the task
    ├── train.py / play.py    thin wrappers around mjlab's CLIs
    ├── train_params.py       loads train.toml
    ├── so101_constants.py    robot EntityCfg: loads the imported XML, strips its
    │                         actuators, re-declares actuators/gains in Python,
    │                         home keyframe, action scale, workspace ranges
    ├── assets.py             get_cube_spec (free box) + get_bin_spec (fixed tray)
    ├── pick_place_env_cfg.py robot-agnostic manager wiring (obs/act/reward/etc.)
    ├── mdp/
    │   ├── commands.py       PlaceInBinCommand: randomizes cube + bin, goal = bin
    │   └── observations.py   cube_to_goal_distance (command-agnostic)
    └── config/
        ├── env_cfg.py        SO-101 specialization (entities, scale, grasp site)
        ├── rl_cfg.py         PPO (rsl-rl) runner cfg, built from train.toml
        └── __init__.py       register_mjlab_task(...)
```

## How it works

mjlab uses Isaac Lab's **manager-based** design: an environment is a **Scene** (the
physical entities) plus a set of **Managers** that read observations, apply actions,
compute rewards, fire resets, and advance a curriculum every step. All of it runs
batched over `num_envs` on the GPU via `mujoco-warp` — the term functions below operate
on tensors of shape `(num_envs, …)`, never one env at a time.

### How the environment is built

The scene composes **three separate entities** (`config/env_cfg.py`); the repo's MuJoCo
model is imported, never edited (bar the one `grasp_site`):

- **`robot`** — `so101_constants.get_so101_robot_cfg()` loads
  `simulation/mjcf/so101_on_frame.xml` with `MjSpec.from_file`, **strips the XML
  actuators**, and re-declares them in Python (see below). It's a *fixed-base* articulated
  entity, so mjlab's `auto_wrap_fixed_base_mocap` wraps it in a **mocap body**. That wrap
  also pulls in the frame's 45 loose `worldbody` geoms, so the whole rig (frame + arm)
  duplicates and positions correctly in every parallel env. `get_spec` also adds an
  **invisible collision pad** at the lightbox's bottom panel: that panel (`Part_1_1`) is
  visual-only in the model, so without the pad objects fall through it to the ground plane
  below the frame. The pad gives them a surface to rest on at `WORK_SURFACE_Z`.
- **`cube`** — `assets.get_cube_spec()`: a free (`freejoint`) 2.5 cm, 30 g box with high
  tangential friction so the gripper can hold it.
- **`bin`** — `assets.get_bin_spec()`: a base plate + four walls. No freejoint → fixed
  base → mjlab wraps it as a **mocap** body, so it never gets knocked over but can be
  teleported to a new spot each episode.

mjlab attaches each entity under a per-env frame and adds a ground **plane**
(`TerrainEntityCfg`). The rig sits at `HOME_KEYFRAME.pos.z = 0.09` so the frame's legs rest
on the plane and the lightbox bottom panel is at `WORK_SURFACE_Z`; the cube and bin sit on
that panel, in the arm's reach. Physics runs at `timestep=0.005` with `decimation=4`
(policy acts at 50 Hz).

### Managers

| Manager | What ours does |
|---|---|
| **Observation** | Two groups. `actor` (27-dim, with input noise) and `critic` (same, no noise → asymmetric actor-critic): joint pos (7) + joint vel (7) + `ee_to_cube` (3) + `cube_to_goal` (3) + last action (7). |
| **Action** | One `JointPositionActionCfg` over all 7 actuators (slider + 5 arm + gripper): **delta** position targets, per-actuator scale (`SO101_ACTION_SCALE`), added to the current pose and clipped to joint limits. |
| **Command** | `PlaceInBinCommand` (`mdp/commands.py`). On each episode reset it samples a bin (x,y) and a cube (x,y) on the plane (rejecting cube spawns within `min_separation` of the bin), teleports both, and publishes `target_pos` = a point just above the bin opening. Resampling is set to ~never, so randomization happens only at reset. |
| **Event** | Reset the robot base to its env origin and the joints to the home pose ± a small jitter. (Domain-randomization events are a TODO.) |
| **Reward** | The five terms below. |
| **Termination** | `time_out` only (episode length cap); no early failure termination. |
| **Curriculum** | Ramps the joint-velocity penalty over training (below). |

Actuators are declared in Python (`so101_constants.py`) as `BuiltinPositionActuatorCfg`s
— arm/gripper and the rail slider each get their own stiffness / damping / effort /
armature — so gains are first-class, tunable, sim-to-real knobs rather than buried in XML.

### Reward

Per-env, per-step weighted sum (`pick_place_env_cfg.py`). Distances use the `grasp_site`
and the cube/target positions; `d` is Euclidean distance:

| Term | Weight | Form | Purpose |
|---|---|---|---|
| `reach_and_bring` | +1.0 | `reach · (1 + bring)`, `reach = exp(−d(ee,cube)²/0.2²)`, `bring = exp(−d(cube,target)²/0.3²)` | The `(1 + bring)` gate means moving the cube toward the bin only pays **once the gripper has reached the cube** — so the policy learns to reach first, then transport. |
| `place_precise` | +1.0 | `exp(−d(cube,target)²/0.05²)` | Tight Gaussian that sharpens the final placement over the bin. |
| `grasp_lift` | +15.0 | `exp(−d(ee,cube)²/0.05²) · clamp(cube_z − surface, 0, 0.15)` | Pays for lifting the cube **only while the gripper is on it** — the signal that was missing when the first run stalled at reaching. |
| `action_rate_l2` | −0.01 | `‖aₜ − aₜ₋₁‖²` | Discourages jerky action changes. |
| `joint_pos_limits` | −10.0 | penalty for joints at their limits | Keeps the arm off its end-stops. |
| `joint_vel_hinge` | −0.01→−1.0 | `Σ max(|v|−0.5, 0)²` | Penalizes joint speeds above 0.5 rad/s; weight is curriculum-ramped. |

`reach_and_bring` and `place_precise` are reused verbatim from mjlab's manipulation task
(`staged_position_reward`, `bring_object_reward`). **Success** (tracked as a metric, not a
reward) = cube center horizontally inside the bin footprint and below the rim.

### Curriculum

One term ramps the `joint_vel_hinge` weight by training progress: `−0.01` at step 0 →
`−0.1` at 500 iterations → `−1.0` at 1000 iterations (steps counted as
`iterations × num_steps_per_env`). Early training barely penalizes fast motion so the arm
can freely explore reaching and grasping; later it's pushed hard toward smooth,
hardware-safe trajectories.

### Configs

- **`train.toml`** — the values you tune per run: `[runner]` (parallel envs, iterations,
  rollout length, save interval, experiment name), `[algorithm]` (PPO hyperparameters),
  `[policy]` (MLP width/activation/init std). `train_params.py` loads it; `config/rl_cfg.py`
  builds the rsl-rl runner from it; `config/env_cfg.py` reads `num_envs`. CLI flags still
  override any of it per run.
- **Python cfg modules** — `pick_place_env_cfg.py` wires the managers (robot-agnostic);
  `config/env_cfg.py` fills in the SO-101 specifics (entities, action scale, grasp site);
  `so101_constants.py` and `assets.py` define the entities. `config/__init__.py` registers
  the task so mjlab's CLI can find it.

## ⚠️ Must validate on the GPU box before trusting results

These were set from geometry/estimates and need confirmation in the viewer:

1. **Workspace ranges** (`so101_constants.WORKSPACE_*`, `PlaceInBinCommandCfg`) —
   confirm the sampled cube/bin region is actually reachable.
2. **Home pose + vertical offset** (`HOME_KEYFRAME`) — `pos.z=+0.09` aligns the
   original floor with mjlab's plane; joint signs (esp. gripper open/close) are
   unconfirmed.
3. **Actuator gains** — starting values mirror the generic XML gains, **not**
   calibrated STS3215 / rail-drive params. Tune + randomize before sim-to-real.

## TODO

Deliberately not in this scaffold yet:

- [ ] Name the two gripper collision geoms in the XML → enables fingertip-friction
      domain randomization and contact-based grasp detection/reward.
- [ ] Add domain randomization events (friction, mass, actuator gains, latency,
      observation noise) for sim-to-real.
- [ ] Add a vision observation group using the existing wrist/overhead cameras.
