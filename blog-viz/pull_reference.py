"""Pull a matched reference capture off the live rig: both raw frames AND the joint state.

`rl/deploy/utils/pull_frames.py` saves the frames alone, which is all the calibration tool needs
because it drives the arm itself and therefore already knows the pose. The blog figures do not
drive anything, so they need the pose recorded alongside the pixels: without it the sim render in
a sim-vs-real figure is at a different arm pose than the photo it sits next to, and the wrist
camera, which sees almost nothing except the jaws, becomes impossible to compare.

PASSIVE. Never claims control, never sends an action, so it is safe to run while a policy drives.

    uv run --project ../rl/calibrate python pull_reference.py

Writes raw/real_<camera>.png and raw/reference.json (sim qpos, wire state, timestamp).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

import cv2

from livekit.portal import Observation, Operator, OperatorConfig, frame_bytes_to_numpy_rgb

_HERE = pathlib.Path(__file__).resolve().parent
DEPLOY = _HERE.parent / "rl" / "deploy"
sys.path.insert(0, str(DEPLOY))

from utils import bridge  # noqa: E402
from utils.camera_mapping import CAMERA_STACK  # noqa: E402
from utils.common import env, load_env, mint_token  # noqa: E402

CAMERAS = tuple(track for track, _ in CAMERA_STACK)


async def main(out: pathlib.Path, timeout: float) -> None:
    load_env(DEPLOY / "utils")
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    out.mkdir(parents=True, exist_ok=True)

    op = Operator(OperatorConfig.from_yaml_file(DEPLOY / "portal.yaml", room))
    latest: dict[str, Observation] = {}
    op.on_observation(lambda obs: latest.__setitem__("obs", obs))

    print(f"[ref] connecting to {url} room '{room}' (passive, no control) ...")
    await op.connect(url, mint_token("blog-reference", room, name="blog reference"))

    frames, state = {}, None
    deadline = asyncio.get_event_loop().time() + timeout
    try:
        while asyncio.get_event_loop().time() < deadline:
            obs = latest.get("obs")
            if obs is not None:
                # Take the state from the SAME observation that carried the last missing frame, so
                # the pose and the pixels are one instant rather than two.
                for cam in CAMERAS:
                    vf = obs.frames.get(cam)
                    if cam not in frames and vf is not None:
                        frames[cam] = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
                        print(f"[ref] got {cam}: {vf.width}x{vf.height}")
                if obs.state:
                    state = dict(obs.state)
                if len(frames) == len(CAMERAS) and state:
                    break
            await asyncio.sleep(0.05)
    finally:
        await op.disconnect()

    if len(frames) != len(CAMERAS) or not state:
        raise SystemExit(
            f"[ref] incomplete capture: frames={sorted(frames)} state={bool(state)}. "
            "Is the robot publishing in this room?"
        )

    for cam, rgb in frames.items():
        path = out / f"real_{cam}.png"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[ref] wrote {path}  ({rgb.shape[1]}x{rgb.shape[0]})")

    sim_qpos = bridge.real_to_sim(state)
    meta = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "room": room,
        "cameras": list(frames),
        # Sim units (rad, m), keyed by JOINT NAME, which is what SimMirror.render takes.
        "sim_qpos": {k.split(".")[0]: round(v, 6) for k, v in sim_qpos.items()},
        "wire_state": {k: round(v, 4) for k, v in state.items()},
    }
    (out / "reference.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[ref] wrote {out / 'reference.json'}")
    for name, value in meta["sim_qpos"].items():
        print(f"       {name:15s} {value:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=_HERE / "raw")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    asyncio.run(main(args.out, args.timeout))
