"""Pull raw camera frames off the live robot over the LiveKit portal room.

PASSIVE: never claims control, never sends actions, so it is safe while a policy drives.
Waits for a frame on both camera tracks, saves each raw RGB frame to --out, disconnects.
The saved frames are the input for camera calibration.
"""
import argparse
import asyncio
import pathlib
import sys

import cv2
import numpy as np

from livekit.portal import Observation, Operator, OperatorConfig, frame_bytes_to_numpy_rgb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.common import env, load_env, mint_token  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = _HERE.parent / "portal.yaml"
CAMERAS = ("arm_camera", "overhead_camera")


async def main(out: pathlib.Path, prefix: str, timeout: float) -> None:
    load_env(_HERE)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    out.mkdir(parents=True, exist_ok=True)

    op = Operator(OperatorConfig.from_yaml_file(CONFIG_PATH, room))
    latest: dict[str, Observation] = {}

    def on_observation(obs: Observation) -> None:
        latest["obs"] = obs

    op.on_observation(on_observation)

    ident = "frame-puller"
    print(f"[pull] connecting to {url} room '{room}' as '{ident}' (passive, no control) ...")
    await op.connect(url, mint_token(ident, room, name="frame puller"))

    saved: dict[str, np.ndarray] = {}
    deadline = asyncio.get_event_loop().time() + timeout
    try:
        while asyncio.get_event_loop().time() < deadline:
            obs = latest.get("obs")
            if obs is not None:
                for cam in CAMERAS:
                    if cam in saved:
                        continue
                    vf = obs.frames.get(cam)
                    if vf is not None:
                        saved[cam] = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
                        print(f"[pull] got {cam}: {vf.width}x{vf.height}")
                if len(saved) == len(CAMERAS):
                    break
            await asyncio.sleep(0.05)
    finally:
        await op.disconnect()

    if not saved:
        raise SystemExit(
            "[pull] no frames received. Is the robot publishing in this room? "
            "Check LIVEKIT_ROOM and that robot/run.py is up."
        )

    for cam, rgb in saved.items():
        path = out / f"{prefix}{cam}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[pull] wrote {path}  ({rgb.shape[1]}x{rgb.shape[0]})")
    missing = set(CAMERAS) - set(saved)
    if missing:
        print(f"[pull] WARNING: never received {sorted(missing)} within {timeout:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=_HERE / "captures")
    ap.add_argument("--prefix", default="real_", help="filename prefix (default 'real_')")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()
    asyncio.run(main(args.out, args.prefix, args.timeout))
