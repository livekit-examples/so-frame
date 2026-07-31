"""The deploy loop, as a figure rather than a mermaid block.

X Articles renders neither mermaid nor inline math, so the diagram in draft-latex.md is this. Drawn
by hand in the same style as every other figure rather than exported from a diagram tool, so the
page, the type and the palette match.

Node text is kept word for word with the mermaid source in draft.md, so the two cannot drift into
saying different things.

    uv run --project ../rl/calibrate python fig9_deploy_loop.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sim_common as sc
import style

# Two columns: the robot on the left, the policy operator on the right, LiveKit between them.
ROBOT_X, POLICY_X = 0.165, 0.605
ROBOT_W, POLICY_W = 0.26, 0.32
NODE_H = 0.075

# The policy chain runs top to bottom; the robot's three nodes sit on the rows they talk to.
ROWS = dict(top=0.855, stack=0.705, tok=0.555, actor=0.405, integrate=0.255, bridge=0.105)

ROBOT = [
    (ROWS["top"], "two raw frames\n640×480, 120° DFOV"),
    (ROWS["actor"], "measured joint state"),
    (ROWS["bridge"], "servos track the target"),
]
POLICY = [
    (ROWS["top"], "rectify per camera\nrotate, undistort, zoom, crop, colour"),
    (ROWS["stack"], "stack wrist + overhead\n168×168×6"),
    (ROWS["tok"], "frozen DINOv2\n288 patch tokens"),
    (ROWS["actor"], "actor mean\ndelta action in [-1,1]$^7$"),
    (ROWS["integrate"], "integrate running target"),
    (ROWS["bridge"], "sim units to wire units"),
]


def node(ax, x, y, w, text):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - NODE_H / 2), w, NODE_H,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor=style.BG, edgecolor=style.AXIS, lw=1.0, zorder=3,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=style.T_TICK,
            color=style.INK, linespacing=1.45, zorder=4)


def arrow(ax, p0, p1, color, *, dashed=False, rad=0.0, label=None, label_pos=None):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=2,
    ))
    if label:
        lx, ly = label_pos or ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=style.T_FOOT,
                color=color, bbox=dict(facecolor=style.BG, edgecolor="none", pad=1.6), zorder=5)


def group(ax, x0, x1, label, tint):
    ax.add_patch(FancyBboxPatch(
        (x0, 0.035), x1 - x0, 0.945 - 0.035,
        boxstyle="round,pad=0,rounding_size=0.02",
        facecolor=tint, edgecolor=style.AXIS, lw=1.0, alpha=0.6, zorder=1,
    ))
    ax.text((x0 + x1) / 2, 0.962, label, ha="center", va="bottom",
            fontsize=style.T_LABEL, color=style.INK)


def main():
    style.apply()
    fig = plt.figure(figsize=(9.2, 7.0))
    ax = fig.add_axes([0.02, 0.01, 0.96, 0.88])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    group(ax, 0.025, 0.305, "robot runtime", "#F1EEE4")
    group(ax, 0.435, 0.775, "policy operator", "#E8EEF8")

    for y, text in ROBOT:
        node(ax, ROBOT_X, y, ROBOT_W, text)
    for y, text in POLICY:
        node(ax, POLICY_X, y, POLICY_W, text)

    # The policy chain, top to bottom.
    order = [ROWS[k] for k in ("top", "stack", "tok", "actor", "integrate", "bridge")]
    for a, b in zip(order, order[1:]):
        arrow(ax, (POLICY_X, a - NODE_H / 2), (POLICY_X, b + NODE_H / 2), style.BLUE)

    # Across the room, in both directions.
    for y, label in ((ROWS["top"], "livekit room"), (ROWS["actor"], "livekit room")):
        arrow(ax, (ROBOT_X + ROBOT_W / 2, y), (POLICY_X - POLICY_W / 2, y),
              style.ORANGE, label=label)
    arrow(ax, (POLICY_X - POLICY_W / 2, ROWS["bridge"]), (ROBOT_X + ROBOT_W / 2, ROWS["bridge"]),
          style.ORANGE, label="joint targets")

    # The target carried into the next tick, the one edge that is not a hand-off. Routed as an
    # explicit lane outside the panel: an arc between two adjacent rows gets hidden behind the
    # nodes it passes, which made the only feedback in the diagram the hardest edge to see.
    lane = 0.815
    right = POLICY_X + POLICY_W / 2
    dash = dict(color=style.MUTED, lw=1.4, linestyle=(0, (4, 3)), zorder=5)
    ax.plot([right, lane], [ROWS["integrate"]] * 2, **dash)
    ax.plot([lane, lane], [ROWS["integrate"], ROWS["actor"]], **dash)
    ax.add_patch(FancyArrowPatch(
        (lane, ROWS["actor"]), (right, ROWS["actor"]), arrowstyle="-|>", mutation_scale=13,
        lw=1.4, color=style.MUTED, linestyle=(0, (4, 3)), shrinkA=0, shrinkB=0, zorder=5,
    ))
    ax.text(lane + 0.018, (ROWS["integrate"] + ROWS["actor"]) / 2, "next tick's\nstate",
            ha="left", va="center", fontsize=style.T_FOOT, color=style.MUTED, linespacing=1.4)

    style.place_title(fig, "One deploy tick, from raw frames to joint targets")
    return fig


if __name__ == "__main__":
    sc.save(main(), "fig9_deploy_loop.png")
