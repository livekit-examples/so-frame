"""The reward ladder: five stages, each one's ceiling strictly below the next one's floor.

Every number is imported from the training config, so the figure cannot drift from the reward the
policy actually trained on. Needs no simulator, but imports the training package for the constants.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import style
from sim_common import config

cfg = config()

# (label, floor, shaping headroom, what the shaping pays for)
STAGES = [
    ("reach", 0.0, 1.0 + cfg.SHAPE_REACH_CLOSE,
     "align over the cube,\ndescend once aligned,\nthen close the jaw on it"),
    ("grasped", cfg.RUNG_GRASPED, 1.0,
     "carry toward the drop point,\n5 cm above the bin rim"),
    ("holding\nover the bin", cfg.RUNG_HOLDING, cfg.SHAPE_HOLD_OPEN,
     "open the jaw"),
    ("released\nover the bin", cfg.RUNG_RELEASED, 0.0,
     "flat: letting go is\nalready the reward"),
    ("success", cfg.REWARD_SUCCESS, 0.0,
     "cube settled in the bin,\narm and cube static,\njaw clear"),
]

BAR_W = 0.52


def main():
    style.apply()
    fig = plt.figure(figsize=(9.0, 5.4))
    left = 0.075
    ax = fig.add_axes([left, 0.10, 0.965 - left, 0.76])


    for i, (label, floor, head, note) in enumerate(STAGES):
        top = floor + head
        # The stages are exclusive, not cumulative, so each is drawn only over the band it can
        # occupy rather than as a column standing on zero.
        if head:
            ax.add_patch(FancyBboxPatch(
                (i - BAR_W / 2, floor), BAR_W, head,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=style.BLUE, edgecolor="none", zorder=3,
            ))
            ax.text(i, top + 0.22, f"{top:g}", ha="center", va="bottom",
                    fontsize=style.T_TICK, color=style.MUTED)
        else:
            # A flat rung is a single value, so it gets a rule rather than a band.
            ax.plot([i - BAR_W / 2, i + BAR_W / 2], [floor, floor],
                    color=style.ORANGE, lw=5.0, solid_capstyle="round", zorder=3)
        ax.text(i, floor - 0.30, f"{floor:g}", ha="center", va="top",
                fontsize=style.T_TICK, color=style.INK)
        ax.text(i, -1.15, label, ha="center", va="top", linespacing=1.45,
                fontsize=style.T_LABEL, color=style.INK)

    # The invariant, drawn: the gap between one stage's ceiling and the next stage's floor.
    for i in range(len(STAGES) - 1):
        top = STAGES[i][1] + STAGES[i][2]
        nxt = STAGES[i + 1][1]
        ax.annotate("", xy=(i + 0.5, nxt), xytext=(i + 0.5, top),
                    arrowprops=dict(arrowstyle="-|>", color=style.FAINT, lw=1.0,
                                    shrinkA=0, shrinkB=0))
        ax.text(i + 0.58, (top + nxt) / 2, f"+{nxt - top:g}", ha="left", va="center",
                fontsize=style.T_FOOT, color=style.FAINT)

    ax.set_xlim(-0.62, len(STAGES) - 0.30)
    ax.set_ylim(-2.6, 11.2)
    ax.set_xticks([])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("reward")
    style.clean_axes(ax)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_bounds(0, 10)

    # Legend: two mark meanings, so the colour is never carrying identity on its own.
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=9, color=style.BLUE, label="rung + shaping"),
        plt.Line2D([], [], marker="s", ls="", ms=9, color=style.ORANGE, label="flat rung"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.90), frameon=False, fontsize=style.T_TICK,
              labelcolor=style.MUTED, handletextpad=0.5, borderaxespad=0.2)

    style.place_title(fig, "Reward value by task stage, with each stage's shaping range")
    return fig


if __name__ == "__main__":
    import sim_common
    sim_common.save(main(), "fig2_reward_ladder.png")
