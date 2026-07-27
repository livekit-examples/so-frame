"""Real-camera -> sim-camera rectification.

A per-camera mapping JSON turns a raw wide-FOV frame into the view the policy trained on:
rotate -> undistort (barrel) -> centre-crop -> resize to the sim sensor size. ``apply_mapping``
replays it every tick; ``calibrate_camera.py`` is what fits one.

This is the only place the mapping is interpreted. The camera stack lives here too, so the
policy operator and the debug harness cannot disagree about which real camera feeds which sim
camera -- they used to declare that separately.
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

# The sim renders its sensor cameras at this resolution; a mapping's default output size.
SIM_SENSOR_SIZE = 128

# Channel order of the policy's RGB input: arm camera first, then overhead, matching the sim's
# (wrist_camera, overhead_camera) sensor order. Each entry is (portal track, mapping file).
CAMERA_STACK = (
    ("arm_camera", "arm_camera_mapping.json"),            # -> sim wrist_camera
    ("overhead_camera", "overhead_camera_mapping.json"),  # -> sim overhead_camera
)

MAPPINGS_DIR = pathlib.Path(__file__).resolve().parent / "camera_mappings"


def load_mapping(path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_mapping(image: np.ndarray, mapping: dict) -> np.ndarray:
    """Replay a saved mapping on a raw frame (rotate, undistort, crop, resize) -> square sim-sensor image."""
    rot90 = mapping["rot90"]
    if rot90:
        image = np.rot90(image, k=rot90).copy()

    h, w = image.shape[:2]
    angle = mapping["angle_deg"]
    if angle != 0.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, m, (w, h))

    # Undistort with a zoomed-out output matrix: zoom < 1 shrinks the output focal
    # so the full rectified view of a strong barrel lens fits the fixed canvas.
    f = mapping["focal_px"]
    zoom = mapping.get("zoom", 1.0)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    new_k = np.array([[f * zoom, 0, w / 2], [0, f * zoom, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.array([mapping["k1"], mapping["k2"], 0, 0, 0], dtype=np.float64)
    image = cv2.undistort(image, k, dist, None, new_k)

    cx, cy, size = mapping["crop_cx"], mapping["crop_cy"], mapping["crop_size"]
    size = min(size, h, w)  # oversized crop would go non-square and distort aspect
    half = size // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    x0, y0 = x1 - size, y1 - size  # re-anchor so the crop stays fully inside
    crop = image[max(0, y0):y1, max(0, x0):x1]

    out = mapping.get("out_size", SIM_SENSOR_SIZE)
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)


def load_mappings(mappings_dir=MAPPINGS_DIR, label="camera") -> dict[str, dict | None]:
    """Load each camera's mapping. A missing one falls back to a plain resize, which is OUT OF
    DISTRIBUTION -- the policy trained on the rectified view, so it is loudly flagged."""
    out: dict[str, dict | None] = {}
    for track, filename in CAMERA_STACK:
        path = pathlib.Path(mappings_dir) / filename
        if path.exists():
            out[track] = load_mapping(path)
            print(f"[{label}] {track}: mapping {filename}")
        else:
            out[track] = None
            print(f"[{label}] {track}: NO mapping ({filename} missing) -- falling back to a "
                  f"plain {SIM_SENSOR_SIZE}px resize. OUT OF DISTRIBUTION; fit one with "
                  "utils/calibrate_camera.py")
    return out


def rectify(frame_rgb: np.ndarray, mapping: dict | None) -> np.ndarray:
    """Rectify one raw frame, or plain-resize it if this camera has no mapping."""
    if mapping is not None:
        return apply_mapping(frame_rgb, mapping)
    return cv2.resize(frame_rgb, (SIM_SENSOR_SIZE, SIM_SENSOR_SIZE), interpolation=cv2.INTER_AREA)
