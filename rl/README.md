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

## Train / play / render

```bash
uv run soframe-train  Mjlab-Pick-Place-Bin-SO101
uv run soframe-play   Mjlab-Pick-Place-Bin-SO101 --checkpoint-file <path>
uv run soframe-render --checkpoint-file <path> --out fleet.mp4
```

The train/play wrappers `import soframe_rl` first (registering the task) and then delegate
to mjlab's own `train`/`play` CLIs, so all of mjlab's flags apply.

Training parameters (parallel envs, iterations, PPO hyperparameters, network size)
live in **[`train.toml`](train.toml)** — edit that file rather than the Python. Any value
can still be overridden per-run on the CLI (e.g. `--env.scene.num-envs 2048`).

### Pretrained checkpoint

A trained policy ships in **[`checkpoints/model_3300.pt`](checkpoints/model_3300.pt)**
(~4 MB): **55.6% place-success at 94% workspace randomization**, trained with the full
recipe below (staged rewards, ADR spread curriculum, linear smoothness ramp). Try it
without training anything:

```bash
# interactive viewer
uv run soframe-play Mjlab-Pick-Place-Bin-SO101 --checkpoint-file checkpoints/model_3300.pt

# fleet video (one arm -> endless field); vary --seed for different rollouts
uv run soframe-render --checkpoint-file checkpoints/model_3300.pt --seed 0 --out fleet.mp4
```

Both need a GPU for best results but run on CPU (slowly; pass `--device cpu` to
`soframe-render`).

### Fleet render

`soframe-render` rolls out a trained checkpoint across many environments (default 400)
and renders a single camera move: it starts behind one arm at its lightbox, then eases
straight back (no panning) to a low, wide shot where the grid of rigs overflows the frame
and reads as an endless field. Useful both as a hero shot and to debug what a policy
actually does.

Details worth knowing:

- **Episodes are finite and phase-staggered** (`--episode-seconds`, default 6). Play mode
  never times out, so an arm would finish one attempt and freeze; here envs continuously
  auto-reset to fresh cube/bin layouts, keeping every arm active for the whole clip.
- The focus env is the one nearest the grid center and renders in full; all other envs'
  moving parts are stamped in with `mjv_addGeoms`.
- Camera framing is tunable: `--azimuth` (180 = behind the arm), `--elevation-close/-wide`,
  `--wide-dist-frac` (wide-shot distance as a fraction of grid radius; <1 keeps robots
  running past the frame edges). A gradient skybox is added (the plane scene has none).
- Runs on CPU or GPU (`--device`). On a shared box, prefer `nice -n 19` + CPU if a
  training run owns the GPU.

## Layout

```
rl/
├── pyproject.toml            uv project; mjlab dep + console scripts
├── .python-version           pinned to 3.13
├── train.toml                training params (edit this to tune runs)
├── checkpoints/model_3300.pt pretrained policy (55.6% success @ 94% randomization)
└── src/soframe_rl/
    ├── __init__.py           imports config -> registers the task
    ├── train.py / play.py    thin wrappers around mjlab's CLIs
    ├── render.py             fleet render: one arm -> endless field (soframe-render)
    ├── train_params.py       loads train.toml
    ├── so101_constants.py    robot EntityCfg: loads the imported XML, strips its
    │                         actuators, re-declares actuators/gains in Python,
    │                         home keyframe, action scale, work-floor collision pad
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

The reward is a **staged decomposition** (reach → grasp/lift → carry → place → in-bin),
each stage a dense term plus a milestone bonus — the standard recipe for pick-and-place RL
(see [references](#approach--references)):

| Term | Weight | Form | Purpose |
|---|---|---|---|
| `reach_and_bring` | +1.0 | `reach · (1 + bring)`, `reach = exp(−d(ee,cube)²/0.2²)`, `bring = exp(−d(cube,target)²/0.3²)` | Reach the cube (bring only pays once reached). |
| `grasp_lift` | +15.0 | `exp(−d(ee,cube)²/0.05²) · clamp(cube_z − surface, 0, 0.15)` | Pays for lifting the cube **only while the gripper is on it** — forms the grasp. |
| `transport` | +40.0 | `clamp((cube_z − surface)/0.06, 0, 1) · Δ(−d_xy(cube,bin))` | **Potential-based carry**: pays only for the per-step *reduction* in the lifted cube's horizontal distance to the bin, so it can't be farmed by hovering. Lift gate ≈ rim height, so the cube must clear the rim first. |
| `place_precise` | +5.0 | `exp(−d(cube,target)²/0.05²)` | Sharpens placement; `target` is the cube's resting spot **inside** the bin, so lowering the cube in increases the reward. |
| `in_bin_bonus` | +10.0 | `1` while cube is in the bin | Milestone: reward the actual placement. |
| `action_rate_l2` | −0.01 | `‖aₜ − aₜ₋₁‖²` | Discourages jerky action changes. |
| `joint_pos_limits` | −10.0 | penalty for joints at their limits | Keeps the arm off its end-stops. |
| `joint_vel_hinge` | −0.01→−1.0 | `Σ max(|v|−0.5, 0)²` | Penalizes joint speeds above 0.5 rad/s; curriculum-ramped. |

`reach_and_bring`/`place_precise` are reused from mjlab (`staged_position_reward`,
`bring_object_reward`); `grasp_lift`, `transport`, `in_bin_bonus` are in `mdp/rewards.py`.
**Success** (a metric, not a reward) = cube center inside the bin footprint, below the rim.

### Curriculum

Two curriculum terms ramp by training progress (`common_step_counter`, i.e.
`iterations × num_steps_per_env`):

- **Placement spread** (`mdp/curriculums.py`) — the key one. `PlaceInBinCommand.spread`
  scales the layout from **0** (fixed: cube a short ~16 cm hop from the bin) to **1** (full
  workspace randomization of cube and bin). It's **performance-gated (ADR-style)**: a
  smoothed success rate drives it — spread rises only while success ≥ 0.4 and falls below
  0.2, so it holds the easy fixed layout until the policy can place, then widens
  randomization exactly as fast as the policy keeps up (backing off if it starts failing).
  Self-pacing to competence avoids a fixed schedule outrunning the policy and eroding
  success — the automatic-domain-randomization pattern from the literature below.
- **Smoothness** — the `joint_vel_hinge` weight ramps **linearly** from `−0.01` (flat
  until ~iter 3000) to `−0.05` by ~iter 7000 (`reward_weight_ramp_curriculum`): explore
  freely early, then push toward smooth, hardware-safe motion. Two properties matter:
  the change must be *gradual* (a discrete several-fold jump in a penalty shifts the
  reward landscape faster than PPO can adapt and can collapse a trained policy), and
  *capped* (if the penalty outweighs the task reward, carrying the cube costs more than
  placing earns and the policy stops moving — measured on this task, success holds to
  ~−0.05 and erodes fast beyond ~−0.06).

### Approach & references

The reward decomposition and curriculum follow common practice for pick-and-place RL:

- **Staged, decomposed rewards with milestone bonuses** (reach → grasp → lift → carry →
  place): [Pick-and-place RL survey](https://www.mdpi.com/2218-6581/10/3/105),
  [Task decomposition + dedicated reward system](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10296071/),
  [Reward engineering for object pick-and-place](https://arxiv.org/pdf/2001.03792).
- **Curriculum: start fixed/easy, then scale up randomization by a 0→1 coefficient gated on
  progress** — [Dynamic reward curriculum for loco-manipulation](https://arxiv.org/pdf/2509.13239),
  [Domain randomization / ADR](https://www.emergentmind.com/topics/domain-randomization),
  [Asymmetric self-play with ADR](https://arxiv.org/pdf/2101.04882).
- **Start the object within easy reach early**, then expand: [SO-100 cube-lift in Isaac Lab](https://medium.com/@kabilankb2003/training-so-100-robot-for-cube-lifting-in-isaac-lab-from-simulation-to-intelligent-control-with-9e81f94c6d6e).

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
