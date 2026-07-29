"""Fit a real camera's rectification against a sim reference render.

Drag sliders to rectify a raw photo (rotate -> undistort -> crop) until it matches a reference
render of the same sim camera, then save the mapping the deploy loop replays every tick.

    # once, in rl/environments/maniskill (needs the simulator):
    uv run python examples/dump_reference_views.py --out ../../deploy/utils/reference_views

    # here, per camera (no simulator needed):
    uv run python utils/calibrate_camera.py utils/captures/real_overhead_camera.png \\
        --reference utils/reference_views/overhead_camera.png --camera overhead

Pure OpenCV + numpy on purpose. The previous version rendered the sim live, which meant the
deploy tree had to install mani_skill and SAPIEN just to calibrate a camera. The sim side is
now a PNG, so this runs anywhere -- including on the robot host.

The trade: the sim's FOV is no longer a slider, because the reference render is fixed. If the
scale cannot be made to match, the sim FOV itself is wrong -- re-dump the reference with
`--overhead-fov` / `--wrist-fov` and re-fit. That is rare; the FOVs in config.py are already
calibrated. The sim CAMERA POSE was never a slider here on purpose: the URDF pose is ground
truth, and the old pose-offset fit is superseded by the FOV parameterization.

The crop is fully adjustable, matching what apply_mapping already replays: pan it with
crop cx/cy, resize it with crop size, and pull the undistorted view in or out with zoom
(< 1 shrinks the output focal so a strong barrel lens fits the canvas). Earlier versions
hardcoded a centred largest-square crop at zoom 1, which the mapping format never required.

Keys: [s] save, [c] re-centre the crop for the current rot90, [q] quit.
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


def rotated_dims(raw_shape, rot90):
    """(h, w) after the rot90 step. The crop fields are expressed in these coordinates."""
    return (raw_shape[1], raw_shape[0]) if rot90 % 2 else (raw_shape[0], raw_shape[1])


def build_mapping(camera, raw_shape, rot90, angle, k1, k2, focal, *,
                  crop_cx=None, crop_cy=None, crop_size=None, zoom=1.0,
                  out_size=SIM_SENSOR_SIZE):
    """The mapping dict, exactly as apply_mapping will replay it at deploy time.

    The crop defaults to the largest centred square, which is what this tool used to hardcode.
    """
    h, w = rotated_dims(raw_shape, rot90)
    crop = min(int(crop_size) if crop_size else min(h, w), h, w)
    return dict(
        camera=camera,
        source_image_size=[int(raw_shape[1]), int(raw_shape[0])],
        # Angular span of the crop at this focal length. Recorded for reference: it is the real
        # camera's span, and it can legitimately differ from the sim FOV, since the fit is what
        # reconciles the two. Tracks zoom, which scales the undistorted output focal.
        fov_deg=round(float(np.rad2deg(2 * np.arctan(crop / (2 * focal * zoom)))), 2),
        rot90=int(rot90), angle_deg=float(angle), k1=float(k1), k2=float(k2),
        focal_px=int(focal), zoom=round(float(zoom), 3),
        crop_cx=int(crop_cx if crop_cx is not None else w // 2),
        crop_cy=int(crop_cy if crop_cy is not None else h // 2),
        crop_size=int(crop),
        out_size=int(out_size),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="raw frame off the real camera (see utils/pull_frames.py)")
    parser.add_argument("--reference", required=True,
                        help="sim reference render for the same camera, from "
                             "rl/environments/maniskill/examples/dump_reference_views.py")
    parser.add_argument("--camera", choices=["overhead", "arm"], default="overhead",
                        help="which mapping to write: <camera>_camera_mapping.json")
    parser.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent / "camera_mappings"))
    parser.add_argument("--out-size", type=int, default=None,
                        help=f"rectified output resolution (default: the existing mapping's, "
                             f"else {SIM_SENSOR_SIZE}). Match the checkpoint's render size: "
                             f"{SIM_SENSOR_SIZE} for squint, 168 for dino_patch. Too small and "
                             "the encoder upsamples a blurrier view than it trained on.")
    args = parser.parse_args()

    raw = cv2.imread(args.image)
    if raw is None:
        raise SystemExit(f"could not read {args.image}")
    ref = cv2.imread(args.reference)
    if ref is None:
        raise SystemExit(f"could not read {args.reference} "
                         "(dump it with rl/environments/maniskill/examples/dump_reference_views.py)")
    ref_view = cv2.resize(ref, (VIEW, VIEW), interpolation=cv2.INTER_AREA)

    # Seed from the existing mapping for this camera, so re-fitting starts where you left off.
    existing = pathlib.Path(args.out_dir) / f"{args.camera}_camera_mapping.json"
    seed = json.loads(existing.read_text()) if existing.exists() else {}
    if seed:
        print(f"[calibrate] starting from {existing}")

    out_size = args.out_size or int(seed.get("out_size", SIM_SENSOR_SIZE))

    # Crop sliders span the largest raw dimension, since rot90 can swap h and w while running;
    # build_mapping and apply_mapping both clamp the crop back inside the frame.
    max_dim = int(max(raw.shape[0], raw.shape[1]))
    seed_h, seed_w = rotated_dims(raw.shape, int(seed.get("rot90", 1)))

    cv2.namedWindow(WINDOW)
    tb = cv2.createTrackbar
    tb("rot90", WINDOW, int(seed.get("rot90", 1)), 3, lambda v: None)
    tb("angle deg +-20", WINDOW, int(seed.get("angle_deg", 0.0)) + 20, 40, lambda v: None)
    tb("k1 x1000 +-600", WINDOW, int(seed.get("k1", -0.33) * 1000) + 600, 1200, lambda v: None)
    tb("k2 x1000 +-300", WINDOW, int(seed.get("k2", 0.275) * 1000) + 300, 600, lambda v: None)
    tb("focal px", WINDOW, int(seed.get("focal_px", 512)), 1200, lambda v: None)
    tb("zoom x100", WINDOW, int(round(float(seed.get("zoom", 1.0)) * 100)), 200, lambda v: None)
    tb("crop cx", WINDOW, int(seed.get("crop_cx", seed_w // 2)), max_dim, lambda v: None)
    tb("crop cy", WINDOW, int(seed.get("crop_cy", seed_h // 2)), max_dim, lambda v: None)
    tb("crop size", WINDOW, int(seed.get("crop_size", min(seed_h, seed_w))), max_dim,
       lambda v: None)
    tb("blend %", WINDOW, 50, 100, lambda v: None)

    print("[calibrate] straighten the rig's edges with k1/k2 and angle first, then match scale "
          "with focal and zoom, then frame it with crop cx/cy/size; read the blend last. "
          f"writing {out_size}px output. [s] save, [c] re-centre crop, [q] quit")

    while True:
        g = lambda n: cv2.getTrackbarPos(n, WINDOW)  # noqa: E731
        rot90, angle = g("rot90"), g("angle deg +-20") - 20
        k1 = (g("k1 x1000 +-600") - 600) / 1000.0
        k2 = (g("k2 x1000 +-300") - 300) / 1000.0
        focal = max(g("focal px"), 100)
        zoom = max(g("zoom x100"), 20) / 100.0
        crop_size = max(g("crop size"), 32)

        mapping = build_mapping(args.camera, raw.shape, rot90, angle, k1, k2, focal,
                                crop_cx=g("crop cx"), crop_cy=g("crop cy"),
                                crop_size=crop_size, zoom=zoom, out_size=out_size)
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
            f"focal    {focal} px   zoom {zoom:.2f}",
            f"crop     {mapping['crop_size']} px @ "
            f"({mapping['crop_cx']},{mapping['crop_cy']})",
            f"out      {mapping['out_size']} px",
            f"spans    {mapping['fov_deg']:.1f} deg",
            "",
            "[s] save  [c] centre  [q] quit",
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
        if key == ord("c"):
            # rot90 may have swapped h and w since the sliders were seeded.
            h, w = rotated_dims(raw.shape, rot90)
            cv2.setTrackbarPos("crop cx", WINDOW, w // 2)
            cv2.setTrackbarPos("crop cy", WINDOW, h // 2)
            cv2.setTrackbarPos("crop size", WINDOW, min(h, w))
            cv2.setTrackbarPos("zoom x100", WINDOW, 100)
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
