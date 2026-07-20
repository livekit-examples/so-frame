"""Interactively move the SIM camera (pose + FOV) against a rectified real photo.

Renders the sim live while you drag sliders, blended over the real frame, so you can
see directly which parameter is wrong instead of guessing. The two panels are
independent: SIM sliders only change the sim render (FOV by default; position/rotation
too with --unlock-pose, otherwise the URDF pose is treated as ground truth), REAL
sliders only change how the raw photo is rectified (rotation, distortion, focal). The
real panel is a full-width center crop at the REAL focal's own scale, so its angular
span is fixed by the REAL knobs alone; drag SIM fov until the contents match scale.
Workflow: straighten the rig's edges with REAL k1/focal first, then match scale with
SIM fov, then read the blend.

    VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
    uv run python examples/calibrate_camera.py path/to/real_overhead.png --camera overhead

Keys:
    [s] print train_squint.py flags and write the deploy mapping JSON for this state
    [q]/[esc] quit
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

import soframe_rl_maniskill.envs  # noqa: F401
from soframe_rl_maniskill.envs.base_random_env import RandomizationConfig

WINDOW = "calibrate_camera"
RENDER_SIZE = 256
SQUINT = 32


def rectify_real(image_bgr, rot90, k1, k2, focal_px, fov_rad, out_size):
    """Rotate + undistort the raw frame, then center-crop exactly `fov_rad` of it."""
    if rot90:
        image_bgr = np.rot90(image_bgr, k=rot90).copy()
    h, w = image_bgr.shape[:2]
    k = np.array([[focal_px, 0, w / 2], [0, focal_px, h / 2], [0, 0, 1]], dtype=np.float64)
    # Zoom out just enough that the fov-sized crop fits the narrow axis.
    crop = min(h, w)
    f_eff = crop / (2 * np.tan(fov_rad / 2))
    new_k = np.array([[f_eff, 0, w / 2], [0, f_eff, h / 2], [0, 0, 1]], dtype=np.float64)
    und = cv2.undistort(image_bgr, k, np.array([k1, k2, 0, 0, 0]), None, new_k)
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    return cv2.resize(und[y0:y0 + crop, x0:x0 + crop], (out_size, out_size), interpolation=cv2.INTER_AREA)


def squint_tile(img):
    small = cv2.resize(img, (SQUINT, SQUINT), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_NEAREST)


class SimView:
    def __init__(self, camera):
        self.camera = camera
        self.env = gym.make(
            "SOFramePickPlaceBin-v1", num_envs=1, obs_mode="rgb+segmentation",
            render_mode="sensors", sim_backend="cpu", domain_randomization=False,
            sensor_configs=dict(width=RENDER_SIZE, height=RENDER_SIZE),
        )
        self.env.reset(seed=0)
        u = self.env.unwrapped
        self.u = u
        self.mount = u.overhead_camera_mount if camera == "overhead" else u.wrist_camera_mount
        self.link = u.agent.overhead_camera_link if camera == "overhead" else u.agent.wrist_camera_link
        self.render_cams = u.scene.sensors[f"{camera}_camera"].camera._render_cameras
        self.rest_qpos = u.agent.robot.get_qpos().clone()
        self.slider_limits = u.agent.robot.get_qlimits()[0, u.agent.joint_names.index("dof_slider")]
        self.item_home = u.item.pose
        self.bin_home = u.bin.pose
        self._last = None
        self._cached = None

    def render(self, fov_deg, pos, rpy_deg, rail_frac, hide_objects):
        key = (fov_deg, tuple(pos), tuple(rpy_deg), rail_frac, hide_objects)
        if key == self._last and self._cached is not None:
            return self._cached
        u = self.u

        if hide_objects:
            u.item.set_pose(sapien.Pose(p=[5, 5, 0.1]))
            u.bin.set_pose(sapien.Pose(p=[5, 6, 0.1]))
        else:
            u.item.set_pose(self.item_home)
            u.bin.set_pose(self.bin_home)

        qpos = self.rest_qpos.clone()
        lo, hi = self.slider_limits
        qpos[0, u.agent.joint_names.index("dof_slider")] = lo + rail_frac * (hi - lo)
        u.agent.robot.set_qpos(qpos)

        for rc in self.render_cams:
            rc.set_fovy(np.deg2rad(fov_deg), compute_x=True)
        offset = sapien.Pose(p=list(pos), q=euler2quat(*np.deg2rad(rpy_deg)))
        self.mount.set_pose(self.link.pose * offset)

        obs = u.get_obs()
        rgb = obs["sensor_data"][f"{self.camera}_camera"]["rgb"][0].cpu().numpy()
        self._cached = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._last = key
        return self._cached


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="raw real camera frame")
    parser.add_argument("--camera", choices=["overhead", "wrist"], default="overhead")
    parser.add_argument("--unlock-pose", action="store_true",
                        help="also expose sim camera position/rotation sliders. By default the "
                             "URDF pose is treated as ground truth and only intrinsics vary.")
    parser.add_argument("--out-prefix", default=None)
    args = parser.parse_args()
    out_prefix = args.out_prefix or f"{args.camera}_aligned"

    raw = cv2.imread(args.image)
    if raw is None:
        raise SystemExit(f"could not read {args.image}")

    sim = SimView(args.camera)

    default_cfg = RandomizationConfig()
    default_fov = round(float(np.rad2deg(getattr(default_cfg, f"{args.camera}_camera_fov"))))

    cv2.namedWindow(WINDOW)
    tb = cv2.createTrackbar
    # SIM side: what the simulated camera does.
    tb("SIM fov deg", WINDOW, default_fov, 95, lambda v: None)
    if args.unlock_pose:
        tb("SIM fwd cm", WINDOW, 10, 70, lambda v: None)      # -10..+60
        tb("SIM lat cm", WINDOW, 15, 30, lambda v: None)      # -15..+15
        tb("SIM vert cm", WINDOW, 15, 30, lambda v: None)     # -15..+15
        tb("SIM roll deg", WINDOW, 20, 40, lambda v: None)    # -20..+20
        tb("SIM pitch deg", WINDOW, 30, 60, lambda v: None)   # -30..+30
        tb("SIM yaw deg", WINDOW, 20, 40, lambda v: None)     # -20..+20
    tb("SIM rail %", WINDOW, 50, 100, lambda v: None)
    tb("SIM hide obj", WINDOW, 0, 1, lambda v: None)
    # REAL side: how the raw photo is rectified before comparison.
    tb("REAL rot90", WINDOW, 1, 3, lambda v: None)
    tb("REAL k1 x1000", WINDOW, 270, 1200, lambda v: None)   # -0.6..+0.6, start -0.33
    tb("REAL k2 x1000", WINDOW, 420, 600, lambda v: None)    # -0.3..+0.3, start +0.12
    tb("REAL focal px", WINDOW, 465, 1200, lambda v: None)
    # Display only.
    tb("blend %", WINDOW, 50, 100, lambda v: None)

    print("[calibrate_camera] drag sliders; [s] save, [q] quit")
    if not args.unlock_pose:
        print("[calibrate_camera] sim camera pose locked to the URDF (ground truth); "
              "pass --unlock-pose to expose position/rotation sliders")

    while True:
        g = lambda n: cv2.getTrackbarPos(n, WINDOW)
        fov = max(g("SIM fov deg"), 30)
        if args.unlock_pose:
            pos = ((g("SIM fwd cm") - 10) / 100.0, (g("SIM lat cm") - 15) / 100.0, (g("SIM vert cm") - 15) / 100.0)
            rpy = (g("SIM roll deg") - 20.0, g("SIM pitch deg") - 30.0, g("SIM yaw deg") - 20.0)
        else:
            pos, rpy = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        k1 = (g("REAL k1 x1000") - 600) / 1000.0
        k2 = (g("REAL k2 x1000") - 300) / 1000.0
        focal = max(g("REAL focal px"), 100)
        rot90 = g("REAL rot90")

        sim_bgr = sim.render(fov, pos, rpy, g("SIM rail %") / 100.0, bool(g("SIM hide obj")))
        # Real panel is independent of the SIM side: full-width center crop of the
        # rectified frame at the REAL focal's own scale (zoom = 1). Its angular span is
        # therefore 2*atan(crop / (2*focal)); shown in the info panel. Alignment means
        # dragging SIM fov until the two contents match scale, at which point the sim
        # fov equals the real crop's true angular span.
        real_fov = 2 * np.arctan(min(raw.shape[:2]) / (2 * focal))
        real = rectify_real(raw, rot90, k1, k2, focal, real_fov, RENDER_SIZE)

        alpha = g("blend %") / 100.0
        blend = cv2.addWeighted(real, 1 - alpha, sim_bgr, alpha, 0)
        top = np.hstack([real, sim_bgr, blend])

        info = np.zeros((RENDER_SIZE, RENDER_SIZE, 3), dtype=np.uint8)
        lines = [
            f"SIM  fov {fov} deg",
            f"REAL crop spans {np.rad2deg(real_fov):.1f} deg",
            "     (aligned when these match)",
            f"SIM  pos {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f} m",
            f"SIM  rpy {rpy[0]:+.1f} {rpy[1]:+.1f} {rpy[2]:+.1f} deg",
            f"REAL rot90={rot90} k1={k1:+.2f} k2={k2:+.2f}",
            f"REAL focal {focal}px",
            "",
            "[s] save  [q] quit",
        ]
        for i, line in enumerate(lines):
            cv2.putText(info, line, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        bottom = np.hstack([squint_tile(real), squint_tile(sim_bgr), info])

        cv2.imshow(WINDOW, np.vstack([top, bottom]))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            cv2.imwrite(f"{out_prefix}_preview.png", top)
            # Deploy mapping consumed by sim2real/policy/camera_mapping.apply_mapping. It
            # must reproduce the REAL panel exactly as displayed here (full-width crop
            # at the REAL focal's own scale, zoom = 1), because that is the view the
            # sim was aligned against. fov_deg records that crop's angular span; note
            # it intentionally differs from the SIM fov when the URDF pose and the
            # physical camera disagree: the alignment reconciles the two.
            h, w = (raw.shape[1], raw.shape[0]) if rot90 % 2 else (raw.shape[0], raw.shape[1])
            crop = min(h, w)
            mapping = dict(
                camera=args.camera, source_image_size=[raw.shape[1], raw.shape[0]],
                fov_deg=round(float(np.rad2deg(real_fov)), 2), sim_fov_deg=float(fov),
                rot90=rot90, angle_deg=0.0, k1=k1, k2=k2,
                focal_px=focal, zoom=1.0,
                crop_cx=w // 2, crop_cy=h // 2, crop_size=crop, out_size=128,
            )
            with open(f"{args.camera}_camera_mapping.json", "w") as f:
                json.dump(mapping, f, indent=2)
            print(f"\nsaved {out_prefix}_preview.png and {args.camera}_camera_mapping.json")
            print("train with:")
            flags = ""
            if args.unlock_pose:
                flags += f" --{args.camera}_camera_pos_offset {pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}"
                flags += f" --{args.camera}_camera_rot_offset {rpy[0]:.2f} {rpy[1]:.2f} {rpy[2]:.2f}"
            if abs(fov - default_fov) > 0.5:
                flags += f" --{args.camera}_camera_fov {fov}"
            print(" " + (flags.strip() or "(no flags needed: everything at defaults)"))

    cv2.destroyAllWindows()
    sim.env.close()


if __name__ == "__main__":
    main()
