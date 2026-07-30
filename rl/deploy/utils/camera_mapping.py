"""Real-camera -> sim-camera rectification.

A per-camera mapping JSON turns a raw wide-FOV frame into the view the policy trained on:
rotate -> undistort (barrel) -> crop -> resize to the sim sensor size. ``apply_mapping`` replays it
every tick, ``build_mapping`` constructs one; rl/calibrate drives the latter from sliders.

The only place a mapping is interpreted: a fit must be replayed by the code that fit it, or
the real view drifts off distribution. CAMERA_STACK lives here too, so every caller agrees on
which real camera feeds which sim camera.
"""
from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

# The sim renders its sensor cameras at this resolution; a mapping's default output size.
SIM_SENSOR_SIZE = 128

# Channel order of the policy's RGB input, matching the sim's (wrist_camera, overhead_camera)
# sensor order. This ordering decides which real camera feeds which sim camera; changing it
# silently feeds the policy swapped views. Entries are (portal track, mapping file).
CAMERA_STACK = (
    ("arm_camera", "arm_camera_mapping.json"),            # -> sim wrist_camera
    ("overhead_camera", "overhead_camera_mapping.json"),  # -> sim overhead_camera
)

MAPPINGS_DIR = pathlib.Path(__file__).resolve().parent / "camera_mappings"


def load_mapping(path) -> dict:
    with open(path) as f:
        m = json.load(f)
    if "crop_size" in m or "crop_cx" in m:
        raise ValueError(
            f"{path} is a pre-offset mapping (crop_cx/crop_cy/crop_size). Framing is now "
            "offset_x/offset_y on the undistort output, which is not a rename: the old movable "
            "crop was pinned in the narrow axis and its size fought `zoom` for field of view. "
            "There is no exact conversion, so refit this camera in rl/calibrate."
        )
    return m


def rotated_dims(raw_shape, rot90) -> tuple[int, int]:
    """(h, w) after the rot90 step. The crop fields are expressed in these coordinates."""
    return (raw_shape[1], raw_shape[0]) if rot90 % 2 else (raw_shape[0], raw_shape[1])


def build_mapping(camera, raw_shape, rot90, angle, k1, k2, focal, *,
                  zoom=1.0, offset_x=0.0, offset_y=0.0, out_size=SIM_SENSOR_SIZE,
                  gain_r=1.0, gain_g=1.0, gain_b=1.0, gamma=1.0) -> dict:
    """A mapping dict, exactly as apply_mapping will replay it."""
    h, w = rotated_dims(raw_shape, rot90)
    size = min(h, w)
    return dict(
        camera=camera,
        source_image_size=[int(raw_shape[1]), int(raw_shape[0])],
        # Angular span of the kept square, recorded for reference. This is the REAL camera's span
        # and may differ from the sim FOV; the fit is what reconciles the two.
        fov_deg=round(float(np.rad2deg(2 * np.arctan(size / (2 * focal * zoom)))), 2),
        # Rounded only to keep the JSON readable, and never below what the UI lets you dial in:
        # a fit that snaps back a decimal place when reloaded is worse than a long number.
        rot90=int(rot90), angle_deg=round(float(angle), 2),
        k1=round(float(k1), 4), k2=round(float(k2), 4),
        focal_px=round(float(focal), 2), zoom=round(float(zoom), 4),
        offset_x=round(float(offset_x), 2), offset_y=round(float(offset_y), 2),
        out_size=int(out_size),
        # Colour, in the array's channel order (RGB on the deploy path).
        gain_r=round(float(gain_r), 4), gain_g=round(float(gain_g), 4),
        gain_b=round(float(gain_b), 4), gamma=round(float(gamma), 4),
    )


def match_colour_gains(real_rgb: np.ndarray, sim_rgb: np.ndarray, clip=(0.5, 2.0)) -> list[float]:
    """Per-channel gains that bring `real` channel means onto `sim`'s.

    Matching the SIM render is the point: neutralising to grey would only make the real camera
    self-consistent, while the policy needs it to look like what it trained on. Both frames must
    be the same scene, so hold the arm still and keep the objects consistent.
    """
    out = []
    for c in range(3):
        r = float(real_rgb[..., c].mean())
        s = float(sim_rgb[..., c].mean())
        g = 1.0 if r < 1e-3 else s / r
        out.append(min(clip[1], max(clip[0], g)))
    return out


def apply_mapping(image: np.ndarray, mapping: dict) -> np.ndarray:
    """Replay a saved mapping on a raw frame: rotate, undistort, crop, resize -> square image."""
    rot90 = mapping["rot90"]
    if rot90:
        image = np.rot90(image, k=rot90).copy()

    h, w = image.shape[:2]
    angle = mapping["angle_deg"]
    if angle != 0.0:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, m, (w, h))

    # zoom and offset both live in the undistort OUTPUT matrix. zoom scales the output focal
    # (< 1 shrinks, so a strong barrel lens fits the canvas); the offset shifts the principal
    # point, which translates the content within the canvas. Positive x moves it right, positive
    # y moves it down.
    #
    # Panning used to be a movable crop window, which could not work: the crop was the largest
    # square that fits, so in the narrow axis there was zero slack and its centre was pinned.
    # Shifting the principal point has no such limit, and it leaves exactly one control for field
    # of view instead of two that fought each other.
    f = mapping["focal_px"]
    zoom = mapping.get("zoom", 1.0)
    ox, oy = mapping.get("offset_x", 0.0), mapping.get("offset_y", 0.0)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)
    new_k = np.array([[f * zoom, 0, w / 2 + ox],
                      [0, f * zoom, h / 2 + oy],
                      [0, 0, 1]], dtype=np.float64)
    dist = np.array([mapping["k1"], mapping["k2"], 0, 0, 0], dtype=np.float64)
    image = cv2.undistort(image, k, dist, None, new_k)

    # Always the centred largest square: framing is the offset's job now.
    size = min(h, w)
    x0, y0 = (w - size) // 2, (h - size) // 2
    crop = image[y0:y0 + size, x0:x0 + size]

    out = mapping.get("out_size", SIM_SENSOR_SIZE)
    view = cv2.resize(crop, (out, out), interpolation=cv2.INTER_AREA)
    return apply_colour(view, mapping)


def apply_colour(view: np.ndarray, mapping: dict) -> np.ndarray:
    """Per-channel gain and gamma, in the CHANNEL ORDER OF THE ARRAY.

    Deploy hands rectify() RGB frames, so gain_r is channel 0 there. Applied after the resize
    because it is per-pixel and the output is far smaller than the raw frame.

    This exists because a colour cast is invisible to everything else here: rectification is
    geometry and never touches pixel values, so a camera that reads warm stays warm all the way
    into the encoder. The policy trained under SensorAugWrapper's +-10% per-channel gain and
    0.7-1.4 gamma, so the target is to land inside that envelope, not to be perfectly neutral.
    """
    gains = [float(mapping.get(k, 1.0)) for k in ("gain_r", "gain_g", "gain_b")]
    gamma = float(mapping.get("gamma", 1.0))
    if gamma == 1.0 and all(g == 1.0 for g in gains):
        return view
    x = view.astype(np.float32) / 255.0
    if gamma != 1.0:
        x = np.power(np.clip(x, 0.0, 1.0), gamma)
    x *= np.asarray(gains, dtype=np.float32)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def load_mappings(mappings_dir=MAPPINGS_DIR, label="camera") -> dict[str, dict | None]:
    """Load each camera's mapping. A missing one falls back to a plain resize, which is OUT OF
    DISTRIBUTION, so it is flagged loudly."""
    out: dict[str, dict | None] = {}
    for track, filename in CAMERA_STACK:
        path = pathlib.Path(mappings_dir) / filename
        why = None
        if not path.exists():
            why = f"{filename} missing"
        else:
            try:
                out[track] = load_mapping(path)
                print(f"[{label}] {track}: mapping {filename}")
                continue
            except ValueError:
                # An unusable mapping is the same situation as a missing one: fall back rather
                # than refusing to start, or you could not run the tool that fixes it.
                why = f"{filename} is a pre-offset mapping and needs a refit"
        out[track] = None
        print(f"[{label}] {track}: NO mapping ({why}) -- falling back to a "
              f"plain {SIM_SENSOR_SIZE}px resize. OUT OF DISTRIBUTION; fit one with "
              "rl/calibrate: uv run calibrate")
    return out


def rectify(frame_rgb: np.ndarray, mapping: dict | None, out_size=None) -> np.ndarray:
    """Rectify one raw frame, or plain-resize it if this camera has no mapping.

    ``out_size`` is the fallback resolution for the no-mapping case. It matters because the
    per-camera views get concatenated channel-wise: one camera on a 168px mapping and another on a
    128px fallback cannot stack at all.
    """
    if mapping is not None:
        return apply_mapping(frame_rgb, mapping)
    size = int(out_size or SIM_SENSOR_SIZE)
    return cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)


def stack_out_size(mappings: dict, default=None) -> int:
    """The resolution every camera's view must share to be stackable.

    Taken from whichever mappings exist, so a camera falling back to a plain resize still lands on
    the same size as its neighbours. Disagreeing mappings are a configuration error, not something
    to average: the largest wins and says so, since upsampling the smaller view is recoverable
    while a crash is not.
    """
    sizes = {int(m["out_size"]) for m in mappings.values() if m and "out_size" in m}
    if len(sizes) > 1:
        print(f"[camera] mappings disagree on out_size {sorted(sizes)}; using {max(sizes)}. "
              "Refit them at one resolution, ideally the checkpoint's.")
    return max(sizes) if sizes else int(default or SIM_SENSOR_SIZE)
