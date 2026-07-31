"""One spawn zone, and what the episodes drawn from it actually look like.

Left: the zone projected onto the overhead camera's own view, which is where two of the three
limits that define it come from. Right: the bin and cube positions of many episodes, sampled by
the env's own reset rather than re-implemented here, so the gap rule and the per-object edge inset
are the ones the task really uses.

    uv run --project ../rl/calibrate python fig5_spawn_zone.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import sim_common as sc
import style

EPISODES = 220


def main():
    cfg = sc.config()
    cx, cy = cfg.WORKSPACE_CENTER
    hx, hy = cfg.WORKSPACE_HALF

    env, u = sc.build_env(res=336)   # rendered large, purely so the figure panel is crisp
    sc.park_objects(u)
    view = sc.sim_views(u)["overhead_camera"]
    K, ext = sc.camera_projection(u, "overhead_camera")

    # The zone's four corners on the work surface, projected through the camera that sees them.
    corners = [(cx - hx, cy - hy), (cx + hx, cy - hy), (cx + hx, cy + hy), (cx - hx, cy + hy)]
    uv = sc.project([[x, y, cfg.WORK_SURFACE_Z] for x, y in corners], K, ext)
    scale = view.shape[0] / 336.0
    uv = uv * scale

    # Sample real episodes: reset and read where the env put things.
    cubes, bins_ = [], []
    for i in range(EPISODES):
        env.reset(seed=1000 + i)
        cubes.append(u.item.pose.p[0, :2].cpu().numpy().copy())
        bins_.append(u.bin.pose.p[0, :2].cpu().numpy().copy())
    cubes, bins_ = np.array(cubes), np.array(bins_)
    env.close()

    gaps = np.linalg.norm(cubes - bins_, axis=1)

    style.apply()
    fig = plt.figure(figsize=(10.6, 4.9))
    left = 0.06
    gs = fig.add_gridspec(1, 2, left=left, right=0.975, bottom=0.16,
                          width_ratios=[1.0, 1.30], wspace=0.14,
                          top=style.content_top(fig, headings=True))

    style.title_block(fig, "Spawn zone in the overhead view, and sampled object positions", left=left)


    # ---- The zone, seen by the camera that constrains it ---------------------------------
    ax = fig.add_subplot(gs[0])
    ax.imshow(view)
    ax.add_patch(plt.Polygon(uv, closed=True, fill=False, ec=style.ORANGE, lw=2.0, zorder=3))
    ax.fill(uv[:, 0], uv[:, 1], color=style.ORANGE, alpha=0.10, zorder=2)
    # The zone runs past the bottom of the frame, and drawing it autoscales the axes to follow.
    # Clamp back to the image so the panel stays a picture rather than a picture plus empty space.
    ax.set_xlim(0, view.shape[1])
    ax.set_ylim(view.shape[0], 0)
    style.frame_image(ax)
    style.panel_title(ax, "overhead view", color=style.INK)

    # ---- The spawns themselves -----------------------------------------------------------
    # Plotted with the rail axis horizontal: the zone is 728 mm along the rail against 458 mm
    # across it, so laid out this way it fills the panel instead of a tall sliver of it.
    ax2 = fig.add_subplot(gs[1])
    ax2.add_patch(Rectangle((cy - hy, cx - hx), 2 * hy, 2 * hx, fill=True,
                            facecolor=style.ORANGE, alpha=0.09, ec=style.ORANGE, lw=1.6, zorder=1))
    ax2.scatter(bins_[:, 1], bins_[:, 0], s=30, marker="s", facecolor=style.BIN,
                edgecolor=style.MUTED, linewidth=0.4, alpha=0.75, zorder=2, label="bin")
    ax2.scatter(cubes[:, 1], cubes[:, 0], s=11, marker="o", facecolor=style.CUBE,
                edgecolor="none", alpha=0.85, zorder=3, label="cube")
    # Equal aspect, but with the limits pinned to the zone plus a small margin: left to itself it
    # pads whichever axis the panel shape leaves slack in, which reads as zone that isn't there.
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("along the rail (m)")
    ax2.set_ylabel("across it (m)")
    style.clean_axes(ax2, grid_axis=None)
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, -0.34), ncol=2, frameon=False,
               fontsize=style.T_TICK, labelcolor=style.MUTED, handletextpad=0.4, borderaxespad=0.0)
    style.panel_title(ax2, f"{EPISODES} episodes", color=style.INK)
    # Limits last: equal aspect will otherwise pad whichever axis the panel shape leaves slack in,
    # which draws zone that is not there. Both axes are metres, so the aspect has to stay honest.
    ax2.set_xlim(cy - hy - 0.02, cy + hy + 0.02)
    ax2.set_ylim(cx - hx - 0.02, cx + hx + 0.02)

    return fig


if __name__ == "__main__":
    sc.save(main(), "fig5_spawn_zone.png")
