"""Calibrate the SO-101-on-frame rig: drive the arm and fit its camera mappings in one window.

MOVES THE ROBOT. The joint fields span the sim range and leave through the same bridge the policy
uses; the sim is rendered at the commanded qpos beside the live rectified real view, so a mapping
is fitted and checked at whatever pose the arm is in.

Imports the deploy project's bridge, camera stack and mapping code, so what is fitted here is
replayed byte-for-byte by the deploy loop. `--bridge` prints the joint self-test and exits.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import cv2
import numpy as np

from utils import bridge
from utils import camera_mapping
from utils.camera_mapping import (
    CAMERA_STACK,
    MAPPINGS_DIR,
    SIM_SENSOR_SIZE,
    load_mapping,
    rectify,
)
from utils.common import DEPLOY_ROOT

# This tool lives outside rl/deploy, so `.env` and `portal.yaml` resolve against the deploy tree
# rather than against this file.
_HERE = DEPLOY_ROOT
CONFIG_PATH = DEPLOY_ROOT / "portal.yaml"

# Where the mapping fields start for a camera that has never been fitted. k1/k2 are the barrel
# terms of the lenses on this rig, so a fresh fit begins roughly undistorted rather than raw.
MAP_DEFAULTS = {"rot90": 0, "angle_deg": 0.0, "k1": -0.33, "k2": 0.275, "focal_px": 512,
                "zoom": 1.0, "offset_x": 0.0, "offset_y": 0.0,
                "gain_r": 1.0, "gain_g": 1.0, "gain_b": 1.0, "gamma": 1.0}


def _mapping_for(track: str) -> dict | None:
    fname = dict(CAMERA_STACK)[track]
    path = MAPPINGS_DIR / fname
    if path.exists():
        try:
            return load_mapping(path)
        except ValueError as exc:
            # Not fatal: this is the tool that refits it.
            print(f"[debug] {track}: {exc}")
            return None
    print(f"[debug] {track}: NO mapping ({fname} missing) -- using a plain "
          f"{SIM_SENSOR_SIZE}px resize (OUT OF DISTRIBUTION; the same fallback the policy logs)")
    return None


def _up(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    """Nearest-upscale to `size`, so real and sim are blendable whatever out_size each carries."""
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_NEAREST)


def run_bridge() -> int:
    """Round-trip every joint real<->sim at its limits + midpoint and print the rail's mapping."""
    print("[debug] joint bridge round-trip (sim units <-> wire units)\n")
    header = f"{'joint':<18}{'sim':>10}{'-> real':>12}{'-> sim':>10}{'ok':>5}"
    print(header)
    print("-" * len(header))
    max_err = 0.0
    for k in bridge.JOINT_KEYS:
        lo, hi = bridge.SIM_LIMITS[k]
        for sim_v in (lo, (lo + hi) / 2, hi):
            real_v = bridge.sim_to_real({k: sim_v})[k]
            back = bridge.real_to_sim({k: real_v})[k]
            err = abs(back - sim_v)
            max_err = max(max_err, err)
            unit = "0-100" if k == "dof_slider.pos" else "deg"
            print(f"{k:<18}{sim_v:>10.4f}{real_v:>10.3f} {unit:<2}{back:>10.4f}"
                  f"{'  y' if err < 1e-6 else '  N':>5}")
    print("-" * len(header))
    print(f"[debug] max round-trip error: {max_err:.2e} (want ~0)")

    lo, hi = bridge.SIM_LIMITS["dof_slider.pos"]
    r_lo = bridge.sim_to_real({"dof_slider.pos": lo})["dof_slider.pos"]
    r_hi = bridge.sim_to_real({"dof_slider.pos": hi})["dof_slider.pos"]
    r_mid = bridge.sim_to_real({"dof_slider.pos": 0.0})["dof_slider.pos"]
    print(f"\n[debug] rail: sim {lo:+.3f} (FAR) -> real {r_lo:.3f}, "
          f"sim {hi:+.3f} (CLOSE/near-camera) -> real {r_hi:.3f}, "
          f"sim 0 (rest) -> real {r_mid:.3f}")
    offsets = {k: v for k, v in bridge.OFFSET_REAL.items()
               if v != 0.0 and k != "dof_slider.pos"}
    flips = {k: v for k, v in bridge.SIGN.items() if v != 1.0}
    print(f"\n[debug] non-zero arm offsets: {offsets or 'none'}   flipped signs: {flips or 'none'}")
    print("[debug] every other joint takes real .pos to be the URDF pose in degrees, which is what "
          "lerobot's follower calibration establishes. A missing offset shows as the arm resting "
          "visibly off the sim rest pose; a missing sign flip shows as one joint moving the wrong "
          "way. Both are visible in the calibrate window before anything moves.")
    return 0 if max_err < 1e-6 else 1


def run_ui_smoke(seconds: float, sim_size: int) -> int:
    """Render the window against synthetic data. Proves the UI alone, with no sim and no robot."""
    from PySide6 import QtWidgets

    import window as winmod

    rng = np.random.default_rng(0)
    tracks = [t for t, _ in CAMERA_STACK]
    tile = lambda: rng.integers(30, 220, (sim_size, sim_size, 3), dtype=np.uint8)  # noqa: E731

    QtWidgets.QApplication([])
    # Named inline rather than imported from sim_mirror, which would pull in mani_skill and SAPIEN.
    sim_names = {t: ("wrist_camera" if "arm" in t else "overhead_camera") for t in tracks}
    win = winmod.Window(tracks, sim_names,
                        {k: bridge.SIM_LIMITS[k] for k in bridge.JOINT_KEYS})
    win.show()
    print(f"[debug] ui smoke: {seconds:.0f}s, no sim, no robot. Close the window or wait.")
    t0, frames = time.perf_counter(), 0
    while time.perf_counter() - t0 < seconds and not win.closed:
        win.show_frames({"real": tile(), "sim": tile(), "blend": tile()},
                        {"real": tile(), "sim": tile(), "blend": tile()})
        win.set_text(f"registering {win.track()}", f"out {sim_size} px   spans 50.2 deg",
                     "robot SMOKE TEST")
        win.step()
        frames += 1
    dt = time.perf_counter() - t0
    print(f"[debug] {frames} frames in {dt:.1f}s ({frames/max(dt,1e-3):.0f} fps) -- UI is fine")
    return 0


async def run_calibrate(sim_size: int = 168) -> int:
    """Drive the arm and fit the camera mappings in one window. THIS MOVES THE ROBOT."""
    from livekit.portal import (
        Observation, Operator, OperatorConfig, frame_bytes_to_numpy_rgb,
    )
    from utils.common import env, load_env, mint_token, pace
    from PySide6 import QtWidgets

    import window as winmod

    load_env(_HERE)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))

    # Loaded before the robot is claimed: SAPIEN takes seconds to start, so don't hold control
    # while it does.
    from sim_mirror import REAL_TO_SIM_CAM, SimMirror
    print(f"[debug] loading the simulator ({sim_size}px) ...")
    mirror = SimMirror(sim_size)
    print("[debug] sim ready; the joint fields drive BOTH sim and real.")

    print("=" * 70)
    print("  CALIBRATE -- THIS MOVES THE REAL ROBOT. Keep an e-stop in reach.")
    print("  Joint fields are sim units; targets ramp at the trained speed, not instantly.")
    print("  Fields start at the arm's MEASURED pose, so nothing moves until you change one.")
    print("  Close the window to release control.")
    print("=" * 70)

    op = Operator(OperatorConfig.from_yaml_file(CONFIG_PATH, room))
    latest: dict[str, Observation | None] = {"obs": None}
    op.on_observation(lambda obs: latest.__setitem__("obs", obs))
    await op.connect(url, mint_token("debug-control", room, name="Debug Control"))
    me = op.local_identity()
    await op.set_active_operator(me)
    print("[debug] claimed control; waiting for the first observation ...")

    goal: dict[str, float] | None = None
    async for _ in pace(fps):
        if latest["obs"] is not None:
            goal = bridge.clamp_sim(bridge.real_to_sim(dict(latest["obs"].state)))
            break
    if goal is None:
        print("[debug] no observation -- is the robot in the room?")
        await op.disconnect(); op.close(); return 1
    target = dict(goal)

    mappings = {t: _mapping_for(t) for t, _ in CAMERA_STACK}
    tracks = [t for t, _ in CAMERA_STACK]

    QtWidgets.QApplication([])
    win = winmod.Window(tracks, dict(REAL_TO_SIM_CAM),
                        {k: bridge.SIM_LIMITS[k] for k in bridge.JOINT_KEYS})
    win.set_joints(target)
    win.show()
    seeded = None   # which camera the mapping fields currently hold

    # Peak measured joint speed against what the trained action limit allows, so you can see
    # whether the real arm keeps up with the rate the policy was trained at.
    sim_max_speed = {k: bridge.DELTA_LIMIT[k] * abs(bridge.SCALE[k]) * fps
                     for k in bridge.JOINT_KEYS}
    max_real_speed = {k: 0.0 for k in bridge.JOINT_KEYS}
    prev: dict = {"state": None, "t": None}

    try:
        async for _ in pace(fps):
            # fields -> goal -> ramp the target at the trained per-tick speed
            goal = bridge.clamp_sim(win.joint_values())
            for k in bridge.JOINT_KEYS:
                step = bridge.DELTA_LIMIT[k]
                target[k] += max(-step, min(step, goal[k] - target[k]))
            target = bridge.clamp_sim(target)
            obs = latest["obs"]

            if obs is not None:
                now = time.perf_counter()
                st = {k: float(v) for k, v in obs.state.items()}
                if prev["state"] is not None and prev["t"] is not None:
                    dt = now - prev["t"]
                    if dt > 1e-3:
                        for k in bridge.JOINT_KEYS:
                            if k in st and k in prev["state"]:
                                v = abs(st[k] - prev["state"][k]) / dt
                                if v < 1e4:      # ignore serial-read glitches
                                    max_real_speed[k] = max(max_real_speed[k], v)
                prev["state"], prev["t"] = st, now

            op.send_action(
                bridge.sim_to_real(target),
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us if obs is not None else None,
            )

            raw = {}
            for track in tracks:
                vf = obs.frames.get(track) if obs is not None else None
                raw[track] = (frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
                              if vf is not None else None)

            # Sim renders at the COMMANDED target, the same value on the wire, so both sides show
            # the same pose up to whatever the real arm has tracked.
            sim_frames = mirror.render({k.split(".")[0]: v for k, v in target.items()})

            sel = win.track()
            if seeded != sel:
                # Reseed from the saved mapping so you always nudge the fit that currently ships.
                win.set_mapping({**MAP_DEFAULTS, **(mappings[sel] or {})})
                seeded = sel
            field = win.mapping_values()

            live_map = None
            panels, other = {}, {}
            for track in tracks:
                rgb = raw[track]
                if track == sel and rgb is not None:
                    live_map = camera_mapping.build_mapping(
                        track.replace("_camera", ""), rgb.shape,
                        rot90=int(field["rot90"]), angle=field["angle_deg"],
                        k1=field["k1"], k2=field["k2"],
                        focal=max(field["focal_px"], 1.0),
                        zoom=max(field["zoom"], 0.2),
                        offset_x=field["offset_x"], offset_y=field["offset_y"],
                        out_size=sim_size,
                        gain_r=field["gain_r"], gain_g=field["gain_g"],
                        gain_b=field["gain_b"], gamma=field["gamma"])
                    use = live_map
                else:
                    use = mappings[track]
                real = _up(rectify(rgb, use)) if rgb is not None else None
                sim = _up(sim_frames[REAL_TO_SIM_CAM[track]])
                blend = None
                if real is not None:
                    a = win.blend_value()
                    blend = cv2.addWeighted(real, 1.0 - a, sim, a, 0)
                (panels if track == sel else other).update(real=real, sim=sim, blend=blend)
            win.show_frames(panels, other)

            m = live_map or {}
            offs = {k.split(".")[0]: v for k, v in bridge.OFFSET_REAL.items()
                    if v != 0.0 and k != bridge.RAIL}
            real_now = bridge.sim_to_real(target)
            win.set_text(
                f"{sel}  vs  {REAL_TO_SIM_CAM[sel]}",
                f"out    {m.get('out_size', '-')} px\n"
                f"spans  {m.get('fov_deg', '-')} deg\n"
                f"offset {offs or 'none'}\n"
                f"robot  {'streaming' if obs is not None else 'NO OBSERVATIONS'}",
                "\n".join(f"{k.split('.')[0][:11]:<12}{real_now[k]:>8.2f}"
                          f"{max_real_speed[k]:>8.1f}/{sim_max_speed[k]:<6.1f}"
                          for k in bridge.JOINT_KEYS),
            )
            win.step()
            if win.closed:
                break

            for req in win.take_requests():
                if req == "rest":
                    win.set_joints(bridge.clamp_sim(dict(bridge.SIM_REST)))
                elif req == "park":
                    win.set_joints(bridge.clamp_sim(bridge.real_to_sim(bridge.PARK_REAL)))
                elif req == "hold":
                    win.set_joints(dict(target))
                elif req == "clear_speed":
                    max_real_speed.update({k: 0.0 for k in max_real_speed})
                elif req == "recentre":
                    win.set_mapping(dict(field, offset_x=0.0, offset_y=0.0, zoom=1.0))
                elif req == "match_colour":
                    rgb = raw[sel]
                    if rgb is None or live_map is None:
                        print("[debug] no frame for this camera yet.")
                        continue
                    # Measure with the CURRENT gains removed, or each press compounds the last.
                    neutral = dict(live_map, gain_r=1.0, gain_g=1.0, gain_b=1.0, gamma=1.0)
                    g = camera_mapping.match_colour_gains(
                        camera_mapping.apply_mapping(rgb, neutral),
                        sim_frames[REAL_TO_SIM_CAM[sel]])
                    win.set_mapping(dict(field, gain_r=g[0], gain_g=g[1], gain_b=g[2], gamma=1.0))
                    print(f"[debug] {sel}: matched colour to sim, gains "
                          f"R{g[0]:.3f} G{g[1]:.3f} B{g[2]:.3f}")
                elif req == "save":
                    if live_map is None:
                        print("[debug] nothing to save: no frame for this camera yet.")
                        continue
                    MAPPINGS_DIR.mkdir(parents=True, exist_ok=True)
                    out = MAPPINGS_DIR / dict(CAMERA_STACK)[sel]
                    out.write_text(json.dumps(live_map, indent=2) + "\n")
                    mappings[sel] = live_map
                    print(f"[debug] saved {out}")
    except KeyboardInterrupt:
        print("\n[debug] Ctrl-C -- stopping.")
    finally:
        mirror.close()
        try:
            if op.active_operator() == me:
                await op.set_active_operator(None)
            await op.disconnect()
        finally:
            op.close()
    print("[debug] released control.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Drive the SO-101-on-frame rig and fit its camera mappings against a live sim "
                    "render. MOVES THE ROBOT.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--bridge", action="store_true",
                   help="print the joint bridge self-test and exit; no robot, no simulator")
    p.add_argument("--ui-smoke", type=float, default=0.0, metavar="SECONDS",
                   help="open the window with synthetic panels for N seconds and exit. No "
                        "simulator, no robot: isolates the UI from everything else.")
    p.add_argument("--sim-size", type=int, default=168,
                   help="sim render resolution, also the mapping's out_size (default 168, the "
                        "dino input size; use 128 for squint)")
    args = p.parse_args()

    if args.ui_smoke:
        return run_ui_smoke(args.ui_smoke, args.sim_size)
    if args.bridge:
        return run_bridge()
    run_bridge()   # always shown first: offsets and sign flips before anything moves
    return asyncio.run(run_calibrate(sim_size=args.sim_size))


if __name__ == "__main__":
    raise SystemExit(main())
