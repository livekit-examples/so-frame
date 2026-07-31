"""Rectifying reality into the simulator's geometry, for both cameras.

Per camera: the sim render, the raw wide-angle frame the robot publishes, that frame put through
the fitted mapping, and the two blended so the alignment is checkable rather than asserted.

The sim is rendered at the ARM POSE the real frames were captured at, which is why
pull_reference.py records the joint state alongside the pixels. Without it the wrist panel would
be comparing two different poses, and the wrist camera sees little except the jaws.

The sim's cube and bin are stood where the REAL ones are, recovered from the rectified overhead
frame rather than measured by hand: once a mapping is fitted, a pixel in a real frame can be
un-projected onto the work surface through the sim camera that the mapping was fitted against.
That the two then land on top of each other is the check, not the assumption.

    uv run --project ../rl/calibrate python fig3_calibration.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import sim_common as sc
import style

COLS = ("sim render", "raw frame, as published", "rectified", "blended")


def mapping_note(m):
    """The mapping's fitted values, as one line."""
    if not m:
        return "no mapping fitted: plain resize (out of distribution)"
    bits = [f"rot90 {m['rot90']}", f"angle {m['angle_deg']:g}°",
            f"k1 {m['k1']:g}", f"k2 {m['k2']:g}", f"focal {m['focal_px']:g}px",
            f"zoom {m['zoom']:g}", f"offset {m['offset_x']:g}, {m['offset_y']:g}",
            f"gamma {m['gamma']:g}"]
    return "   ".join(bits)


def main():
    ref = sc.reference()
    maps = sc.mappings()

    env, u = sc.build_env(res=sc.RES)
    sc.park_objects(u)
    sc.set_qpos(u, ref["sim_qpos"])

    # Read the real objects' positions off the rectified overhead frame and stand the sim's own
    # cube and bin there. The bin's centroid is taken a centimetre up, on the rim rather than the
    # floor, because that is the height the yellow the camera sees actually sits at.
    cfg = sc.config()
    K, ext = sc.camera_projection(u, "overhead_camera")
    over = sc.rectify_real("overhead_camera", sc.RES, maps=maps)
    found = {}
    for want, height in (("blue", cfg.CUBE_HALF), ("yellow", 0.010)):
        uv = sc.colour_centroid(over, want)
        if uv is not None:
            found[want] = sc.unproject_to_plane(uv, K, ext, height)[0][:2]
    if len(found) == 2:
        sc.place_objects(u, found["blue"], found["yellow"])
        sc.set_qpos(u, ref["sim_qpos"])
    else:
        print("[viz] could not locate both objects in the real frame; leaving them parked")

    sim = sc.sim_views(u)

    style.apply()
    fig = plt.figure(figsize=(12.2, 7.4))
    left = 0.055
    gs = fig.add_gridspec(2, 4, left=left, right=0.985, top=0.775, bottom=0.10,
                          wspace=0.06, hspace=0.30)

    style.title_block(
        fig,
        "Reality is rectified into the simulator, not the other way round",
        "The real cameras are 120° wide-angle modules. Rather than teach the renderer to fake a "
        "cheap lens, every real frame is undistorted,\nrotated and cropped to exactly the view the "
        "policy trained on. The same mapping file is replayed on every deploy tick.",
        left=left, gap=0.045,
    )

    for row, cam in enumerate(sc.CAMERA_ORDER):
        raw = sc.real_frame(cam)
        rect = sc.rectify_real(cam, sc.RES, maps=maps)
        s = sim[sc.REAL_TO_SIM_CAM[cam]]
        blend = (0.5 * s.astype(np.float32) + 0.5 * rect.astype(np.float32)).astype(np.uint8)

        for col, img in enumerate((s, raw, rect, blend)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            style.frame_image(ax)
            if row == 0:
                style.panel_title(ax, COLS[col])
            if col == 0:
                ax.text(-0.06, 0.5, sc.CAMERA_TITLE[cam], transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=style.T_LABEL,
                        color=style.INK)

        # The fit, under the row it produced.
        ax_note = fig.add_subplot(gs[row, 1:])
        ax_note.axis("off")
        ax_note.text(0.0, -0.10, mapping_note(maps.get(cam)), transform=ax_note.transAxes,
                     ha="left", va="top", fontsize=style.T_FOOT, color=style.FAINT)

    pose = ", ".join(f"{k} {v:+.2f}" for k, v in ref["sim_qpos"].items())
    style.footnote(
        fig,
        f"Captured off the live rig on {ref['captured_at'][:10]}, sim rendered at the same joint "
        f"pose ({pose}),\nwith the cube and bin stood where the rectified overhead frame says the "
        "real ones are. The overhead camera's FOV was fitted against the rig; the wrist's is "
        "inherited from the MJCF twin,\nwhich is why its objects sit a little large and its blend "
        "is the looser of the two.",
        left=left,
    )
    env.close()
    return fig


if __name__ == "__main__":
    sc.save(main(), "fig3_calibration.png")
