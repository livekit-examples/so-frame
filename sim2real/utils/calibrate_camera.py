"""Fit a real camera's rectification against a sim reference render.

Drag sliders to rectify a raw photo (rotate -> undistort -> crop) until it matches a reference
render of the same sim camera, then save the mapping the deploy loop replays every tick.

    # once, in rl/maniskill (needs the simulator):
    uv run python examples/dump_reference_views.py

    # here, per camera (no simulator needed):
    uv run python utils/calibrate_camera.py captures/real_overhead_camera.png \\
        --reference reference_views/overhead_camera.png --camera overhead

Pure OpenCV + numpy on purpose. The previous version rendered the sim live, which meant the
deploy tree had to install mani_skill and SAPIEN just to calibrate a camera. The sim side is
now a PNG, so this runs anywhere -- including on the robot host.

The trade: the sim's FOV is no longer a slider, because the reference render is fixed. If the
scale cannot be made to match, the sim FOV itself is wrong -- re-dump the reference with
`--fov` and re-fit. That is rare; the FOVs in config.py are already calibrated.

Keys: [s] save, [q] quit.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import cv2
import numpy as np

from camera_mapping import SIM_SENSOR_SIZE, apply_mapping

WINDOW = "calibrate_camera"
VIEW = 256   # display size per panel
SQUINT = 32  # the squint encoder's resolution, shown so you can judge what survives it


def squint_tile(img):
    """What the policy's low-res encoder actually sees, upscaled to compare."""
    small = cv2.resize(img, (SQUINT, SQUINT), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (VIEW, VIEW), interpolation=cv2.INTER_NEAREST)


def build_mapping(camera, raw_shape, rot90, angle, k1, k2, focal, out_size=SIM_SENSOR_SIZE):
    """The mapping dict, exactly as apply_mapping will replay it at deploy time."""
    h, w = (raw_shape[1], raw_shape[0]) if rot90 % 2 else (raw_shape[0], raw_shape[1])
    crop = min(h, w)
    return dict(
        camera=camera,
        source_image_size=[int(raw_shape[1]), int(raw_shape[0])],
        # Angular span of the crop at this focal length. Recorded for reference: it is the real
        # camera's span, and it can legitimately differ from the sim FOV, since the fit is what
        # reconciles the two.
        fov_deg=round(float(np.rad2deg(2 * np.arctan(crop / (2 * focal)))), 2),
        rot90=int(rot90), angle_deg=float(angle), k1=float(k1), k2=float(k2),
        focal_px=int(focal), zoom=1.0,
        crop_cx=int(w // 2), crop_cy=int(h // 2), crop_size=int(crop),
        out_size=int(out_size),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="raw frame off the real camera (see utils/pull_frames.py)")
    parser.add_argument("--reference", required=True,
                        help="sim reference render for the same camera, from "
                             "rl/maniskill/examples/dump_reference_views.py")
    parser.add_argument("--camera", choices=["overhead", "arm"], default="overhead",
                        help="which mapping to write: <camera>_camera_mapping.json")
    parser.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent / "camera_mappings"))
    args = parser.parse_args()

    raw = cv2.imread(args.image)
    if raw is None:
        raise SystemExit(f"could not read {args.image}")
    ref = cv2.imread(args.reference)
    if ref is None:
        raise SystemExit(f"could not read {args.reference} "
                         "(dump it with rl/maniskill/examples/dump_reference_views.py)")
    ref_view = cv2.resize(ref, (VIEW, VIEW), interpolation=cv2.INTER_AREA)

    # Seed from the existing mapping for this camera, so re-fitting starts where you left off.
    existing = pathlib.Path(args.out_dir) / f"{args.camera}_camera_mapping.json"
    seed = json.loads(existing.read_text()) if existing.exists() else {}
    if seed:
        print(f"[calibrate] starting from {existing}")

    cv2.namedWindow(WINDOW)
    tb = cv2.createTrackbar
    tb("rot90", WINDOW, int(seed.get("rot90", 1)), 3, lambda v: None)
    tb("angle deg +-20", WINDOW, int(seed.get("angle_deg", 0.0)) + 20, 40, lambda v: None)
    tb("k1 x1000 +-600", WINDOW, int(seed.get("k1", -0.33) * 1000) + 600, 1200, lambda v: None)
    tb("k2 x1000 +-300", WINDOW, int(seed.get("k2", 0.275) * 1000) + 300, 600, lambda v: None)
    tb("focal px", WINDOW, int(seed.get("focal_px", 512)), 1200, lambda v: None)
    tb("blend %", WINDOW, 50, 100, lambda v: None)

    print("[calibrate] straighten the rig's edges with k1/k2 and angle first, then match scale "
          "with focal; read the blend last. [s] save, [q] quit")

    while True:
        g = lambda n: cv2.getTrackbarPos(n, WINDOW)  # noqa: E731
        rot90, angle = g("rot90"), g("angle deg +-20") - 20
        k1 = (g("k1 x1000 +-600") - 600) / 1000.0
        k2 = (g("k2 x1000 +-300") - 300) / 1000.0
        focal = max(g("focal px"), 100)

        mapping = build_mapping(args.camera, raw.shape, rot90, angle, k1, k2, focal)
        # Rectify through the SAME function deploy uses, so what you align is what ships.
        rect = apply_mapping(raw, mapping)
        rect_view = cv2.resize(rect, (VIEW, VIEW), interpolation=cv2.INTER_NEAREST)

        alpha = g("blend %") / 100.0
        blend = cv2.addWeighted(rect_view, 1 - alpha, ref_view, alpha, 0)

        info = np.zeros((VIEW, VIEW, 3), dtype=np.uint8)
        for i, line in enumerate([
            f"camera   {args.camera}",
            f"rot90    {rot90}   angle {angle:+d} deg",
            f"k1       {k1:+.3f}",
            f"k2       {k2:+.3f}",
            f"focal    {focal} px",
            f"crop     {mapping['crop_size']} px",
            f"spans    {mapping['fov_deg']:.1f} deg",
            "",
            "[s] save   [q] quit",
        ]):
            cv2.putText(info, line, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1)

        cv2.imshow(WINDOW, np.vstack([
            np.hstack([rect_view, ref_view, blend]),
            np.hstack([squint_tile(rect_view), squint_tile(ref_view), info]),
        ]))
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            out_dir = pathlib.Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{args.camera}_camera_mapping.json"
            path.write_text(json.dumps(mapping, indent=2) + "\n")
            preview = out_dir / f"{args.camera}_camera_fit.png"
            cv2.imwrite(str(preview), np.hstack([rect_view, ref_view, blend]))
            print(f"[calibrate] saved {path} and {preview}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
