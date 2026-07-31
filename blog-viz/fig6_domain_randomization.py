"""What the policy actually trains on: the same scene, resampled.

The leftmost column is the scene with randomization off. Everything to its right is one draw from
the training distribution, through the same wrappers the training pipeline uses, in the same
order: lighting and gains and joint noise inside the env, then colour jitter, then the
sensor-realism pass that stands in for a cheap USB camera.

Camera pose and FOV jitter are drawn per scene build, not per episode, so each column is a full
reconfigure rather than a reset.

    uv run --project ../rl/calibrate python fig6_domain_randomization.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import sim_common as sc
import style

DRAWS = 7
RES = 168
BIN_XY = (-0.120, -0.620)
CUBE_XY = (-0.165, -0.380)


def main():
    import torch

    sc.config()
    import soframe_rl_maniskill.wrappers as w

    cols = []
    for i in range(DRAWS + 1):
        # A fresh env per column, not a reconfigure of one: camera pose and FOV jitter are drawn
        # when the scene is BUILT, and repeatedly rebuilding one scene exhausts MoltenVK's
        # descriptor pool on this machine.
        env, u = sc.build_env(dr=bool(i), seed=200 + i, res=RES)
        sc.place_objects(u, CUBE_XY, BIN_XY)
        # Jaw shut: the wrist camera sees little except the gripper, and only closed do both jaws
        # meet in frame. At the rest pose the moving jaw swings out and the view is a blank wedge.
        sc.set_qpos(u, {"gripper": 0.0})
        raw = u.get_obs()["sensor_data"]
        rgb = np.concatenate(
            [raw[c]["rgb"][0].cpu().numpy() for c in ("wrist_camera", "overhead_camera")], axis=-1,
        )
        if i:   # column 0 is the same scene with randomization off, as the reference
            # Applied by hand rather than by wrapping the env, so one code path produces both the
            # clean column and the randomized ones. Order is sac/build.py's: jitter, then sensor.
            jitter, sensor = w.ColorJitterWrapper(env), w.SensorAugWrapper(env)
            obs = sensor.observation(jitter.observation({"rgb": torch.as_tensor(rgb)}))
            rgb = obs["rgb"].cpu().numpy().astype(np.uint8)
        cols.append((rgb[..., :3], rgb[..., 3:6]))
        env.close()

    style.apply()
    fig = plt.figure(figsize=(13.0, 4.6))
    left = 0.045
    gs = fig.add_gridspec(2, DRAWS + 1, left=left, right=0.99, top=0.70, bottom=0.10,
                          wspace=0.035, hspace=0.06)

    style.title_block(
        fig,
        "Reality should be one more draw from the training distribution",
        "Lighting, PD gains, joint-read noise, camera pose and FOV, colour, gamma, white balance, "
        "sensor noise and a compression proxy, all resampled.",
        left=left,
    )

    for col, (wrist, overhead) in enumerate(cols):
        for row, img in enumerate((wrist, overhead)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            style.frame_image(ax)
            if col == 0:
                ax.text(-0.09, 0.5, ("wrist", "overhead")[row], transform=ax.transAxes,
                        rotation=90, ha="right", va="center", fontsize=style.T_FOOT,
                        color=style.MUTED)
            if row == 0:
                style.panel_title(ax, "off" if col == 0 else f"draw {col}",
                                  color=style.INK if col == 0 else style.FAINT)

    style.footnote(
        fig,
        "Colour randomization of the objects themselves is deliberately off: there is one real rig "
        "and its cube and bin are a known blue and yellow, so those are matched rather than "
        "randomized,\nand the policy gets to spend its capacity elsewhere. Ranges live in "
        "envs/base_random_env.py and envs/pick_place.py.",
        left=left,
    )
    return fig


if __name__ == "__main__":
    sc.save(main(), "fig6_domain_randomization.png")
