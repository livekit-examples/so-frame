"""What each encoder is actually handed.

The squint CNN renders at 128 px and area-averages down to 32, where a 20 mm cube is about one
pixel and hue is the only thing left of it. The DINOv2 heads take 168 px, which is 12 patches a
side, so the cube spans a patch or two and survives as a shape.

    uv run --project ../rl/calibrate python fig4_what_the_encoder_sees.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import sim_common as sc
import style

# Where to stand the objects for the shot: both in the overhead camera's footprint, far enough
# apart to read, and inside the spawn zone the task samples from.
BIN_XY = (-0.120, -0.650)
CUBE_XY = (-0.165, -0.395)

DISPLAY = 460   # every panel is blown up to this many pixels, so the pixel sizes compare honestly


def patch_grid(ax, n, size, color):
    """Draw DINOv2's patch boundaries over an image panel."""
    step = size / n
    for i in range(1, n):
        ax.axvline(i * step - 0.5, color=color, lw=0.6, alpha=0.55)
        ax.axhline(i * step - 0.5, color=color, lw=0.6, alpha=0.55)


def main():
    cfg = sc.config()

    # 168 first: the resolution the deployed checkpoint's cameras rendered at.
    env, u = sc.build_env(res=168)
    sc.place_objects(u, CUBE_XY, BIN_XY)
    v168 = sc.sim_views(u)["overhead_camera"]
    # The cube's pixel, taken from the camera itself rather than eyeballed, so the ring lands on it
    # in every panel regardless of resolution.
    K, ext = sc.camera_projection(u, "overhead_camera")
    cube_uv = sc.project([[CUBE_XY[0], CUBE_XY[1], cfg.WORK_SURFACE_Z + cfg.CUBE_HALF]], K, ext)[0]
    cube_frac = cube_uv / 168.0
    env.close()

    # ... then 128, the squint CNN's render size. Rebuilt rather than resized, because the whole
    # point of squinting is that the 32 px frame is an AVERAGE of a real render, not a resample of
    # a different one.
    env, u = sc.build_env(res=128)
    sc.place_objects(u, CUBE_XY, BIN_XY)
    v128 = sc.sim_views(u)["overhead_camera"]
    env.close()

    v32 = sc.squint(v128, k=4)

    style.apply()
    # Three stacked bands of text above the panels: the title, then the encoder each pair of
    # panels belongs to, then each panel's own resolution. They need room, hence the low `top`.
    fig = plt.figure(figsize=(12.4, 5.6))
    left = 0.055
    gs = fig.add_gridspec(1, 4, left=left, right=0.985, bottom=0.03, wspace=0.055,
                          top=0.97)

    panels = [
        (sc.blow_up(v128, DISPLAY), "128 px render", None),
        (sc.blow_up(v32, DISPLAY), "32 px, area-averaged", None),
        (sc.blow_up(v168, DISPLAY), "168 px render", None),
        (sc.blow_up(v168, DISPLAY), "168 px, 12 × 12 patches", 12),
    ]

    axes = []
    for i, (img, heading, grid) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img, interpolation="nearest")
        if grid:
            patch_grid(ax, grid, DISPLAY, style.BLUE)
        style.frame_image(ax)
        style.panel_title(ax, heading, color=style.INK)

        # Ring the cube in every panel, so the eye can follow the one object that disappears.
        ax.add_patch(Circle((cube_frac[0] * DISPLAY, cube_frac[1] * DISPLAY),
                            DISPLAY * 0.055, fill=False, ec=style.ORANGE, lw=1.6))
        axes.append(ax)

    # Name the encoder over the panels it consumes, so a panel is never just a resolution.
    style.span_label(fig, axes[0:2], "squint CNN")
    style.span_label(fig, axes[2:4], "DINOv2 heads")

    style.place_title(fig, "Overhead camera view at each encoder's input resolution")
    return fig


if __name__ == "__main__":
    sc.save(main(), "fig4_what_the_encoder_sees.png")
