"""The headline result: what the vision encoder is worth, and what the recipe was worth.

Left panel, the four encoders under the current recipe, all 12M steps, same task, same reward,
same replay retention. Right panel, the same squint CNN before the reward ladder gained its
jaw-closing ramp and the horizon came down to 200 steps: four runs, three of them separate seeds,
none of which ever placed the cube.

Reads raw/wandb_runs.json (see fetch_runs.py). Needs no simulator.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import style

HERE = pathlib.Path(__file__).resolve().parent
RUNS = json.loads((HERE / "raw" / "wandb_runs.json").read_text())

# Eval is 35 parallel episodes, so a reading is a multiple of 1/35 and single points jump around.
# A 3-point rolling mean shows the level without inventing one; the raw points stay visible under
# it so nothing is hidden by the smoothing.
SMOOTH = 3

CURRENT = (
    ("v4-dino", "dino_patch", "dino_patch"),
    ("v5-dino-global-mean", "dino_global_mean", "dino_global (mean)"),
    ("v5-dino-global-cls", "dino_global_cls", "dino_global (cls)"),
    ("v4-squint", "squint", "squint CNN"),
)
PREVIOUS = ("v1-squint", "v2-squint-s1", "v2-squint-s2", "v2-squint-s3", "v3-squint")


def series(name):
    ev = RUNS[name]["eval"]
    return (np.array([e["step"] for e in ev], dtype=float) / 1e6,
            np.array([e["success_at_end"] for e in ev], dtype=float))


def rolling(y, n=SMOOTH):
    if len(y) < n:
        return y
    pad = np.concatenate([np.full(n - 1, y[0]), y])
    return np.convolve(pad, np.ones(n) / n, mode="valid")


def main():
    style.apply()
    fig = plt.figure(figsize=(12.6, 5.5))
    left, right = 0.075, 0.985
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.16,
                          left=left, right=right, top=0.76, bottom=0.13)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    style.title_block(
        fig,
        "The dense patch grid is what learns the task",
        "Evaluation success over training, 35 held-out episodes per point. Same task, reward, "
        "replay retention and step budget throughout.",
        left=left,
    )

    # ---- Panel A: the four encoders, current recipe --------------------------------------
    ax_a.set_title("Four encoders, current recipe", color=style.INK, pad=10)
    ends = []
    for run, key, label in CURRENT:
        x, y = series(run)
        color = style.ENCODER_COLOR[key]
        ax_a.plot(x, y, color=color, lw=1.0, alpha=0.28, zorder=2)
        sm = rolling(y)
        ax_a.plot(x, sm, color=color, lw=2.0, zorder=3,
                  path_effects=style.RELIEF.get(color))
        ends.append((sm[-1], x[-1], color, label))

        # Mark where the run first places a cube at all: half the story is when it blooms.
        first = next((i for i, v in enumerate(y) if v > 0), None)
        if first is not None:
            ax_a.plot([x[first]], [y[first]], "o", ms=6.5, color=color, mec=style.BG, mew=1.6,
                      zorder=4)

    # Direct labels, required: aqua and gold both sit under 3:1 on the cream page, so identity
    # cannot rest on a legend swatch alone. Two runs finish within a few points of each other, so
    # the label anchors are pushed apart top-down and a leader line keeps each tied to its curve.
    ends.sort(reverse=True)
    min_gap = 0.075
    anchors = []
    for i, (value, *_rest) in enumerate(ends):
        y = value if not anchors else min(value, anchors[-1] - min_gap)
        anchors.append(y)
    for (value, xe, color, label), y in zip(ends, anchors):
        if abs(y - value) > 1e-3:
            ax_a.plot([xe, xe + 0.18], [value, y], color=color, lw=0.9, alpha=0.7, zorder=3)
        ax_a.text(xe + 0.28, y, label, color=color, fontsize=style.T_TICK,
                  va="center", ha="left", fontweight="bold")

    # ---- Panel B: the CNN either side of the recipe change --------------------------------
    ax_b.set_title("The same CNN, before and after the recipe change", color=style.INK, pad=10)
    for run in PREVIOUS:
        x, y = series(run)
        ax_b.plot(x, y, color=style.MUTED, lw=1.8, alpha=0.6, zorder=2)
    x, y = series("v4-squint")
    ax_b.plot(x, rolling(y), color=style.ENCODER_COLOR["squint"], lw=2.0, zorder=3,
              path_effects=style.RELIEF[style.GOLD])
    ax_b.text(12.2, rolling(y)[-1], "after", color=style.GOLD, fontsize=style.T_TICK,
              va="center", ha="left", fontweight="bold")
    ax_b.annotate(
        "before: five runs, three of them\nseparate seeds, never one placement",
        xy=(9.4, 0.0), xytext=(9.0, 0.13), color=style.MUTED, fontsize=style.T_TICK,
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.9, alpha=0.8,
                        shrinkA=2, shrinkB=3),
    )

    for ax, xmax in ((ax_a, 15.0), (ax_b, 14.2)):
        style.clean_axes(ax)
        ax.set_xlim(-0.3, xmax)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xticks([0, 3, 6, 9, 12])
        ax.set_xlabel("environment steps (millions)")
    ax_a.set_ylabel("success rate")
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax_b.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_yticklabels([])

    style.footnote(
        fig,
        "Faint line: raw evaluation, a multiple of 1/35. Bold: 3-point rolling mean. "
        "Dot: the first evaluation that placed a cube.  "
        "Left runs: v4-dino, v5-dino-global-mean, v5-dino-global-cls, v4-squint. "
        "Right: v1-squint, v2-squint seeds 1-3, v3-squint.",
        left=left,
    )
    return fig


if __name__ == "__main__":
    import sim_common
    sim_common.save(main(), "fig1_encoder_curves.png")
