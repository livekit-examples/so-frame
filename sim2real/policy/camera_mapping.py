"""Real-camera -> sim-camera rectification for deployment.

Vendored from ``rl/maniskill/examples/calibrate_real_camera.py`` (the interactive
calibration tool) so the deploy stack is self-contained: a robot host runs this
without the mani_skill training project on it. Keep the two functions in sync
with that tool if its mapping format changes.

A mapping JSON is produced by the calibration tool once per camera and captures
how to turn a raw wide-FOV frame into the narrow, undistorted, cropped view the
policy was trained on: rotate -> undistort (barrel) -> center-crop to the sim
FOV -> resize to the sim sensor size. ``apply_mapping`` replays it every tick.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

SIM_SENSOR_SIZE = 128  # sim renders sensor cameras at 128x128 before the 32px squint


def load_mapping(path) -> dict:
    with open(path) as f:
        return json.load(f)


def apply_mapping(image: np.ndarray, mapping: dict) -> np.ndarray:
    """Replay a saved mapping on a raw camera frame: rotate, undistort, crop, resize.

    Returns a square sim-sensor-sized image (``out_size``, default 128), same
    dtype/channel order as the input.
    """
    rot90 = mapping["rot90"]
    if rot90:
        image = np.rot90(image, k=rot90).copy()

    h, w = image.shape[:2]
    angle = mapping["angle_deg"]
    if angle != 0.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, m, (w, h))

    # Undistort with a zoomed-out output camera matrix: with the same matrix on
    # both sides, rectifying a strong barrel distortion (a 120-degree-DFOV lens)
    # pushes the periphery outside the fixed canvas and silently narrows the
    # usable FOV. zoom < 1 shrinks the output focal so the full rectified view fits.
    f = mapping["focal_px"]
    zoom = mapping.get("zoom", 1.0)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    new_k = np.array([[f * zoom, 0, w / 2], [0, f * zoom, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.array([mapping["k1"], mapping["k2"], 0, 0, 0], dtype=np.float64)
    image = cv2.undistort(image, k, dist, None, new_k)

    cx, cy, size = mapping["crop_cx"], mapping["crop_cy"], mapping["crop_size"]
    size = min(size, h, w)  # an oversized crop would go non-square and distort aspect
    half = size // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    x0, y0 = x1 - size, y1 - size  # re-anchor so the crop stays fully inside
    crop = image[max(0, y0):y1, max(0, x0):x1]

    out = mapping.get("out_size", SIM_SENSOR_SIZE)
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)
