"""What the vision encoder is worth: four of them under one recipe, 12M steps each.

Same task, same reward, same replay retention, same step budget, so the only thing varying is
what the policy is handed instead of pixels.

The squint CNN's five zero-success runs under the previous recipe used to be a second panel here.
Dropped: that is a different claim, and five flat lines at zero is weaker evidence than the one
sentence of prose that states it.

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
    fig = plt.figure(figsize=(9.6, 5.0))
    left, right = 0.085, 0.985
    gs = fig.add_gridspec(1, 1, left=left, right=right, bottom=0.14, top=0.97)
    ax_a = fig.add_subplot(gs[0])

    # ---- Panel A: the four encoders, current recipe --------------------------------------
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
                  va="center", ha="left")

    style.clean_axes(ax_a)
    ax_a.set_xlim(-0.3, 15.6)
    ax_a.set_ylim(-0.03, 1.06)
    ax_a.set_xticks([0, 3, 6, 9, 12])
    ax_a.set_xlabel("environment steps (millions)")
    ax_a.set_ylabel("success rate")
    ax_a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_a.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])

    style.place_title(fig, "Evaluation success rate vs environment steps, by vision encoder")
    return fig


if __name__ == "__main__":
    import sim_common
    sim_common.save(main(), "fig1_encoder_curves.png")
