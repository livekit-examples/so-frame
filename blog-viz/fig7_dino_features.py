"""What the frozen backbone sees in a render and in a photograph.

The same DINOv2 ViT-S/14 that the deployed policy runs on, applied to a sim frame and to the
matching rectified real frame, with the patch tokens of BOTH painted by one shared PCA. A shared
projection is the whole point: fitting a separate PCA per image would guarantee they look alike.
Each frame is centred first, which removes the constant domain offset and leaves the structure.

The bottom row is the same tokens read the way the collapsed control reads them: one vector per
camera, which is what the figure is really about, since that is the variant that fails on the real
robot.

    uv run --project ../rl/calibrate python fig7_dino_features.py

`main` takes the real camera track, so fig8 renders the wrist view through this same code.
Pulls the backbone from torch.hub on first use, so the first run needs network.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import sim_common as sc
import style

RES = 168
GRID = RES // 14   # 12


def tokens(rgb, torch):
    """(res, res, 3) uint8 -> (GRID*GRID, 384) float32 patch tokens from the frozen backbone."""
    import sys, pathlib
    sys.path.insert(0, str(sc.REPO / "rl" / "policy" / "src"))
    from soframe_policy.encoders import dino_patch

    stack = torch.as_tensor(np.concatenate([rgb, rgb], axis=-1)).unsqueeze(0)   # 1 camera, twice
    tok = dino_patch.tokenize(stack, RES, num_cams=2, device="cpu")
    return tok[0, : GRID * GRID].float().numpy()


def main(real_cam="overhead_camera", title=None):
    import torch

    ref = sc.reference()
    maps = sc.mappings()

    env, u = sc.build_env(res=RES)
    sc.park_objects(u)
    sc.set_qpos(u, ref["sim_qpos"])
    cfg = sc.config()
    K, ext = sc.camera_projection(u, "overhead_camera")
    over_real = sc.rectify_real("overhead_camera", RES, maps=maps)
    found = {}
    for want, height in (("blue", cfg.CUBE_HALF), ("yellow", 0.010)):
        uv = sc.colour_centroid(over_real, want)
        if uv is not None:
            found[want] = sc.unproject_to_plane(uv, K, ext, height)[0][:2]
    if len(found) == 2:
        sc.place_objects(u, found["blue"], found["yellow"])
        sc.set_qpos(u, ref["sim_qpos"])
    views = sc.sim_views(u)
    sim = views[sc.REAL_TO_SIM_CAM[real_cam]]
    real = sc.rectify_real(real_cam, RES, maps=maps)
    env.close()

    t_sim, t_real = tokens(sim, torch), tokens(real, torch)

    # One PCA over both, each frame mean-centred first so the constant sim-vs-real offset does not
    # eat the leading component. What is left is structure the two share, or do not.
    both = np.concatenate([t_sim - t_sim.mean(0), t_real - t_real.mean(0)], axis=0)
    # float64 for the decomposition, and floating-point flags ignored across it: the tokens arrive
    # via bf16 and leave a stale FPU flag behind that numpy reports at whichever operation checks
    # next, which is this matmul. The inputs and the result are finite; asserted below.
    both = (both - both.mean(0)).astype(np.float64)
    assert np.isfinite(both).all(), "non-finite DINOv2 tokens"
    with np.errstate(all="ignore"):
        _, _, vt = np.linalg.svd(both, full_matrices=False)
        proj = both @ vt[:3].T
    assert np.isfinite(proj).all(), "PCA projection went non-finite"
    lo, hi = np.percentile(proj, 2), np.percentile(proj, 98)
    proj = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1)
    pca_sim, pca_real = proj[: GRID * GRID], proj[GRID * GRID:]

    style.apply()
    fig = plt.figure(figsize=(10.4, 5.4))
    left = 0.065
    gs = fig.add_gridspec(2, 3, left=left, right=0.90, bottom=0.03,
                          wspace=0.07, hspace=0.16, width_ratios=[1, 1, 1.15],
                          top=0.97)


    rows = [("simulation", sim, pca_sim), ("real, rectified", real, pca_real)]
    for r, (label, img, pca) in enumerate(rows):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(img)
        style.frame_image(ax)
        ax.text(-0.07, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center",
                fontsize=style.T_LABEL, color=style.INK)
        if r == 0:
            style.panel_title(ax, "frame", color=style.INK)

        ax = fig.add_subplot(gs[r, 1])
        ax.imshow(pca.reshape(GRID, GRID, 3), interpolation="nearest")
        style.frame_image(ax)
        if r == 0:
            style.panel_title(ax, f"{GRID} × {GRID} patch tokens", color=style.INK)

        # The collapsed control's view: the same tokens averaged into one 384-dim vector. Drawn as
        # a strip of its 384 values rather than as a patch image, because that is the honest shape.
        # It is not empty, it simply has no positions left in it.
        ax = fig.add_subplot(gs[r, 2])
        pooled = (t_sim if r == 0 else t_real).mean(0)
        ax.imshow(pooled.reshape(1, -1), aspect="auto", cmap=style.sequential_blue(),
                  vmin=-2.0, vmax=2.0, interpolation="nearest")
        style.frame_image(ax)
        if r == 0:
            style.panel_title(ax, "mean-pooled to one vector", color=style.INK)

    style.place_title(fig, title or
                      f"DINOv2 patch tokens, {sc.CAMERA_TITLE[real_cam]}, shared PCA projection")
    return fig


if __name__ == "__main__":
    sc.save(main(), "fig7_dino_features.png")
