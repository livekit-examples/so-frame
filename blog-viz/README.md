# blog-viz

Figures for the sim2real write-up in [`../draft.md`](../draft.md). One script per figure, each
writing into `out/`.

[`../draft-latex.md`](../draft-latex.md) is the X Articles version: no inline math, since X renders
only block LaTeX, and `fig9` in place of the mermaid diagram, which X does not render either.

Nothing here re-declares a number that lives somewhere else. The reward constants come from the
training config, the camera mappings from the deploy tree, the sim frames from the training env
itself, and the training curves from a cached Weights & Biases pull. A figure cannot drift from
the code without the code changing under it.

## Running

Everything runs through **rl/calibrate's** project. That is the one environment where the
simulator and the live robot feed coexist (Python 3.12, `sapien` + `mani_skill` + `livekit-portal`
together), which is exactly what these figures need.

```bash
cd blog-viz
uv run --project ../rl/calibrate python fig3_calibration.py
```

Sim rendering on this Mac goes through MoltenVK and SAPIEN's cpu backend. `sim_common.py` sets
`VK_ICD_FILENAMES` on import, so there is no env-var prefix to remember. If it complains, the fix
is `brew install molten-vk`.

The two figures that need no simulator (`fig1`, `fig2`) still import through the same project for
`style.py` and the training constants.

## The reference capture

Several figures compare simulation against the real rig. That comparison is only worth anything if
both sides are at the same arm pose, so `pull_reference.py` records the joint state **alongside**
the frames:

```bash
# with the robot online. Passive: never claims control, safe while a policy is driving.
uv run --project ../rl/calibrate python pull_reference.py
```

It writes `raw/real_arm_camera.png`, `raw/real_overhead_camera.png` and `raw/reference.json` (sim
qpos, wire state, timestamp). `rl/deploy/utils/pull_frames.py` saves the frames alone, which is
all the calibration tool needs because it drives the arm itself and therefore already knows the
pose; here there is nothing driving, so the pose has to be recorded.

Re-pull after **any** camera recalibration. The mappings in `rl/deploy/utils/camera_mappings/` are
replayed by these figures exactly as the deploy loop replays them, so a refit changes what
`fig3` and `fig7` show.

Capture the reference with the **jaw closed** where you can. The wrist camera sees almost nothing
except the gripper, and only closed do both jaws meet in frame; at the rest pose the moving jaw
swings out and the view degrades to a featureless wedge.

## The training curves

`fig1` reads `raw/wandb_runs.json` rather than the network, so it redraws offline and cannot
silently change when a run is renamed upstream. Refresh it with:

```bash
uv run --with wandb python fetch_runs.py
```

Credentials come from `~/.netrc`. The runs it pulls are named in `fetch_runs.py`: the v4/v5 family
is the current recipe, and v1 to v3 are the same code before the reward ladder gained its
jaw-closing ramp and the horizon came down to 200 steps.

## The figures

| # | figure | script | needs |
|---|---|---|---|
| 1 | four encoders, success vs steps | `fig1_encoder_curves.py` | the wandb cache |
| 2 | the reward ladder | `fig2_reward_ladder.py` | training constants only |
| 3 | sim / raw / rectified / blended, both cameras | `fig3_calibration.py` | sim + a reference capture |
| 4 | what each encoder is handed | `fig4_what_the_encoder_sees.py` | sim |
| 5 | the spawn zone, and episodes drawn from it | `fig5_spawn_zone.py` | sim |
| 6 | domain-randomization draws | `fig6_domain_randomization.py` | sim |
| 7 | DINOv2 features, overhead camera | `fig7_dino_features.py` | sim + reference + torch.hub |
| 8 | DINOv2 features, wrist camera | `fig8_dino_features_wrist.py` | same, via `fig7.main("arm_camera")` |
| 9 | the deploy loop | `fig9_deploy_loop.py` | nothing; it is drawn, not measured |

Still missing, because they need capture the repo cannot synthesize:

- **the hero clip**, a matched sim and real rollout of the cube task. The sim half wants
  `rl/environments/maniskill/examples/render_chained_eval.py` against `dino_patch_policy_ckpt.pt`,
  which is a GPU-box job; the real half is a screen capture of a live session.
- **the deploy `--viz` screenshot**, the two rectified views the encoder is fed next to the human's
  wide-angle web view. Grab it during a rollout with `uv run policy --arch dino_patch --viz`.
- **the closer clip**, a long real rollout with no human in the loop.

The old `assets/*.gif` are not stand-ins for these: they are the previous rig, with an orange arm
and coloured jenga blocks, on the previous task.

## Design system

`style.py` holds it: a warm cream page, one type scale, and a left rail the title and the plotting
area share. Edit it to restyle every figure at once. Figures carry a title, panel headings and the
labels needed to read the marks, and nothing else: the prose lives in the post.

Titles are placed by `place_title()`, which draws the figure, measures it, and hangs the title off
the left edge of the leftmost panel at a fixed gap above the tallest thing on the page. Nothing
about the title band is hand-tuned per figure, which is what used to put titles on top of panels.
Regular weight throughout, no bold.

Series colours are the validated four-slot categorical order (blue, orange, aqua, yellow), checked
against this page colour rather than a generic white one: worst adjacent CVD ΔE 9.1, worst adjacent
normal-vision ΔE 22.9. Aqua and yellow fall under 3:1 contrast on cream, so any figure using them
labels its series directly instead of relying on a legend swatch, and both carry a hairline of
relief. One encoder keeps one colour across every figure in the post.

The rig hues (`CUBE`, `BIN`) are separate from the series colours on purpose: they are annotations
that mirror physical objects, not chart accents, and they are converted from the linear base
colours the renderer is given.
