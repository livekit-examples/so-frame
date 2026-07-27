"""Debug harness for the deploy wiring -- check the camera mapping and the joint bridge
WITHOUT a policy in the loop.

Imports the same bridge and camera stack the policy operator uses, so what passes here is what
ships. Modes (see main): --frame, --bridge (default), --live, --snapshot, --control. Only
--control moves the robot.
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys
import time

import cv2
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from utils import bridge  # noqa: E402
from utils.camera_mapping import (  # noqa: E402
    CAMERA_STACK,
    MAPPINGS_DIR,
    SIM_SENSOR_SIZE,
    load_mapping,
    rectify,
)

# The squint encoder's input resolution. Only used to preview what survives the downsample;
# the real thing is SquintEncoder.preprocess, which the policy calls.
SQUINT_SIZE = 32


def _mapping_for(track: str) -> dict | None:
    fname = dict(CAMERA_STACK)[track]
    path = MAPPINGS_DIR / fname
    if path.exists():
        return load_mapping(path)
    print(f"[debug] {track}: NO mapping ({fname} missing) -- using a plain "
          f"{SIM_SENSOR_SIZE}px resize (OUT OF DISTRIBUTION; the same fallback the policy logs)")
    return None


def squint(view: np.ndarray) -> np.ndarray:
    """Area-downsample to the squint resolution, as SquintEncoder.preprocess does."""
    return cv2.resize(view, (SQUINT_SIZE, SQUINT_SIZE), interpolation=cv2.INTER_AREA)


def _labeled(img_rgb: np.ndarray, label: str, size: int = 256) -> np.ndarray:
    """Upscale (nearest) to `size` and caption it, BGR out."""
    up = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return bgr


def composite(raw_rgb: np.ndarray, track: str, mapping: dict | None) -> np.ndarray:
    """raw | rectified-128 | squint-32 strip (BGR), each upscaled for viewing."""
    view = rectify(raw_rgb, mapping)
    tiles = [
        _labeled(raw_rgb, f"{track} raw {raw_rgb.shape[1]}x{raw_rgb.shape[0]}"),
        _labeled(view, "rectified 128 (policy view)"),
        _labeled(squint(view), "squint 32 (net input)"),
    ]
    return np.hstack(tiles)


def run_frame(frame_path: str, track: str, out_dir: pathlib.Path) -> int:
    if track not in dict(CAMERA_STACK):
        print(f"[debug] --camera must be one of {[t for t, _ in CAMERA_STACK]}, got {track!r}")
        return 2
    raw_bgr = cv2.imread(frame_path)
    if raw_bgr is None:
        print(f"[debug] could not read frame {frame_path}")
        return 2
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    mapping = _mapping_for(track)
    strip = composite(raw_rgb, track, mapping)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"debug_{track}.png"
    cv2.imwrite(str(out_path), strip)
    kind = "mapping" if mapping is not None else "PLAIN-RESIZE fallback"
    print(f"[debug] {track}: {raw_rgb.shape[1]}x{raw_rgb.shape[0]} raw -> "
          f"{SIM_SENSOR_SIZE}px view ({kind}) -> {SQUINT_SIZE}px squint")
    if mapping is not None:
        print(f"[debug] mapping: fov {mapping.get('fov_deg','?')} deg, rot90={mapping['rot90']}, "
              f"k1={mapping['k1']}, focal={mapping['focal_px']}, "
              f"zoom={mapping.get('zoom',1.0)}, crop={mapping['crop_size']}")
    print(f"[debug] wrote {out_path}  (raw | rectified-128 | squint-32)")
    return 0


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
    zeros = [k for k in bridge.JOINT_KEYS
             if k != "dof_slider.pos" and bridge.OFFSET_REAL[k] == 0.0]
    if zeros:
        print(f"[debug] NEEDS-CALIBRATION: {len(zeros)} arm/gripper joints still "
              "assume real-zero == sim-zero (OFFSET_REAL=0). Measure real .pos at "
              "the sim-zero pose before any real rollout.")
    return 0 if max_err < 1e-6 else 1


async def run_snapshot(out_dir: pathlib.Path) -> int:
    """Join the room read-only, grab one RAW frame per camera, save, and exit (for calibration)."""
    from livekit.portal import (
        Observation, Operator, OperatorConfig, frame_bytes_to_numpy_rgb,
    )
    from utils.common import env, load_env, mint_token, pace

    load_env(_HERE)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    out_dir.mkdir(parents=True, exist_ok=True)

    op = Operator(OperatorConfig.from_yaml_file(_HERE.parent / "portal.yaml", room))
    latest: dict[str, Observation | None] = {"obs": None}
    op.on_observation(lambda obs: latest.__setitem__("obs", obs))
    print(f"[debug] connecting READ-ONLY to {url} room '{room}' to grab raw snapshots ...")
    await op.connect(url, mint_token("debug-snapshot", room, name="Debug Snapshot"))

    saved: dict[str, str] = {}
    ticks = 0
    try:
        async for _ in pace(fps):
            ticks += 1
            obs = latest["obs"]
            if obs is not None:
                for track, _ in CAMERA_STACK:
                    if track in saved or obs.frames.get(track) is None:
                        continue
                    vf = obs.frames[track]
                    rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
                    path = out_dir / f"snapshot_{track}.png"
                    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    saved[track] = str(path)
                    print(f"[debug] saved {path}  ({rgb.shape[1]}x{rgb.shape[0]} raw)")
                if len(saved) == len(CAMERA_STACK):
                    break
            if ticks > fps * 10:   # ~10s timeout
                break
    finally:
        await op.disconnect()
        op.close()
    if not saved:
        print("[debug] no camera frames received -- is the robot running in the room?")
        return 1
    print(f"[debug] done. Calibrate with: examples/move_sim_camera.py "
          f"{out_dir}/snapshot_arm_camera.png --camera wrist")
    return 0


async def run_live(out_dir: pathlib.Path, every: float) -> int:
    """Read-only: periodically dump the live rectified view and the state bridge round-trip."""
    from livekit.portal import (  # imported here so modes 1/2 need no livekit
        Observation, Operator, OperatorConfig, frame_bytes_to_numpy_rgb,
    )
    from utils.common import env, load_env, mint_token, pace

    load_env(_HERE)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = _HERE.parent / "portal.yaml"

    mappings = {t: _mapping_for(t) for t, _ in CAMERA_STACK}
    op = Operator(OperatorConfig.from_yaml_file(config_path, room))
    latest: dict[str, Observation | None] = {"obs": None}
    op.on_observation(lambda obs: latest.__setitem__("obs", obs))

    print(f"[debug] connecting READ-ONLY to {url} room '{room}' (no control, no motion)")
    await op.connect(url, mint_token("debug-mapping", room, name="Debug Mapping"))
    print(f"[debug] connected; dumping every {every}s to {out_dir} -- Ctrl-C to stop")

    last_dump = 0.0
    saw_obs = False
    try:
        async for _ in pace(fps):
            obs = latest["obs"]
            if obs is None:
                continue
            if not saw_obs:
                saw_obs = True
                print(f"[debug] first observation: state keys={sorted(obs.state)[:3]}..., "
                      f"cameras={sorted(obs.frames)}")
            now = time.perf_counter()
            if now - last_dump < every:
                continue
            last_dump = now

            sim_qpos = bridge.real_to_sim(dict(obs.state))
            back = bridge.sim_to_real(sim_qpos)
            drift = max((abs(back[k] - float(obs.state[k])) for k in sim_qpos), default=0.0)
            slider = sim_qpos.get("dof_slider.pos")
            print(f"[debug] state ok (roundtrip drift {drift:.2e}); "
                  f"rail sim={slider:.3f} " if slider is not None else "[debug] state ok; ")

            for track, _ in CAMERA_STACK:
                vf = obs.frames.get(track)
                if vf is None:
                    print(f"[debug]   {track}: no frame this tick")
                    continue
                rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
                cv2.imwrite(str(out_dir / f"live_{track}.png"),
                            composite(rgb, track, mappings[track]))
            print(f"[debug]   dumped live_*.png to {out_dir}")
    except KeyboardInterrupt:
        print("\n[debug] stopping ...")
    finally:
        try:
            await op.disconnect()
        finally:
            op.close()
    return 0


_SIM_REST = bridge.SIM_REST
# Short aliases + index accepted for joint names in the REPL.
_ALIASES = {"slider": "dof_slider.pos", "rail": "dof_slider.pos",
            "pan": "shoulder_pan.pos", "lift": "shoulder_lift.pos",
            "elbow": "elbow_flex.pos", "wrist": "wrist_flex.pos",
            "roll": "wrist_roll.pos", "grip": "gripper.pos", "gripper": "gripper.pos"}


def _resolve_joint(name: str) -> str | None:
    if name in bridge.JOINT_KEYS:
        return name
    if name + ".pos" in bridge.JOINT_KEYS:
        return name + ".pos"
    if name in _ALIASES:
        return _ALIASES[name]
    if name.isdigit() and 0 <= int(name) < len(bridge.JOINT_KEYS):
        return bridge.JOINT_KEYS[int(name)]
    return None


def _help() -> None:
    joints = ", ".join(f"{i}:{k.split('.')[0]}" for i, k in enumerate(bridge.JOINT_KEYS))
    print("  commands (values are SIM units: rail metres, arm radians):\n"
          "    far | near        rail to the far / near-camera end (rail mapping check)\n"
          "    rest              go to the sim rest pose\n"
          "    <joint> <val>     set a joint's absolute sim target (e.g. `slider 0.3`, `pan -0.4`)\n"
          "    n <joint> <d>     nudge a joint by +d sim units (e.g. `n lift -0.1`)\n"
          "    p                 print current target (sim) and the real values sent\n"
          "    ? | help          this help\n"
          "    q | quit          release control and exit\n"
          f"    joints: {joints}")


async def run_control(assume_yes: bool) -> int:
    """Claim the robot and drive joint targets by hand (typed REPL), ramping at the trained speed."""
    from livekit.portal import Observation, Operator, OperatorConfig, RpcInvocationData
    from utils.common import env, load_env, mint_token, pace

    load_env(_HERE)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    config_path = _HERE.parent / "portal.yaml"

    uncal = [k for k in bridge.JOINT_KEYS
             if k != "dof_slider.pos" and bridge.OFFSET_REAL[k] == 0.0]
    print("=" * 70)
    print("  MANUAL CONTROL -- THIS MOVES THE REAL ROBOT.")
    print("  Targets ramp at the trained speed (rail ~12 cm/s), not instantly.")
    if uncal:
        print(f"  WARNING: {len(uncal)} arm/gripper joints are NOT calibrated "
              "(OFFSET_REAL=0).")
        print("  The RAIL is calibrated; test it first (`far`/`near`). Arm moves may "
              "be wrong.")
    print("  Keep an e-stop / power cut in reach. Ctrl-C stops sending immediately.")
    print("=" * 70)
    if not assume_yes:
        if input("  type 'drive' to proceed: ").strip() != "drive":
            print("[debug] aborted."); return 1

    op = Operator(OperatorConfig.from_yaml_file(config_path, room))
    latest: dict[str, Observation | None] = {"obs": None}
    op.on_observation(lambda obs: latest.__setitem__("obs", obs))

    print(f"[debug] connecting to {url} room '{room}' ...")
    await op.connect(url, mint_token("debug-control", room, name="Debug Control"))
    me = op.local_identity()
    await op.set_active_operator(me)   # claim control
    print("[debug] claimed control. Waiting for the first observation to seed target ...")

    # Seed goal/target from the first real pose so the first send is a no-op.
    goal: dict[str, float] | None = None
    async for _ in pace(fps):
        obs = latest["obs"]
        if obs is not None:
            goal = bridge.clamp_sim(bridge.real_to_sim(dict(obs.state)))
            break
    if goal is None:
        print("[debug] no observation -- is the robot in the room?"); await op.disconnect(); op.close(); return 1
    target = dict(goal)
    print("[debug] seeded from current pose. Type `?` for commands.")

    # Background stdin reader -> queue, so the send loop never blocks.
    queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_event_loop()
    stop = {"quit": False}

    async def reader() -> None:
        while not stop["quit"]:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                await queue.put("q"); return
            await queue.put(line.strip())

    def apply(cmd: str) -> None:
        nonlocal goal, target
        if not cmd:
            return
        parts = cmd.split()
        head = parts[0].lower()
        if head in ("q", "quit"):
            stop["quit"] = True
        elif head in ("?", "help"):
            _help()
        elif head == "p":
            real = bridge.sim_to_real(target)
            print("  sim target : " + "  ".join(f"{k.split('.')[0]}={target[k]:+.3f}" for k in bridge.JOINT_KEYS))
            print("  real sent  : " + "  ".join(f"{k.split('.')[0]}={real[k]:+.3f}" for k in bridge.JOINT_KEYS))
        elif head == "rest":
            goal = bridge.clamp_sim(dict(_SIM_REST)); print("[debug] goal -> sim rest")
        elif head in ("far", "near"):
            lo, hi = bridge.SIM_LIMITS["dof_slider.pos"]
            goal = dict(goal, **{"dof_slider.pos": lo if head == "far" else hi})
            print(f"[debug] rail goal -> {head} (sim {goal['dof_slider.pos']:+.3f} -> "
                  f"real {bridge.sim_to_real(goal)['dof_slider.pos']:.3f})")
        elif head == "n" and len(parts) == 3:
            j = _resolve_joint(parts[1])
            if j is None: print(f"[debug] unknown joint {parts[1]!r}"); return
            goal = bridge.clamp_sim(dict(goal, **{j: goal[j] + float(parts[2])}))
        elif len(parts) == 2 and _resolve_joint(parts[0]):
            j = _resolve_joint(parts[0])
            goal = bridge.clamp_sim(dict(goal, **{j: float(parts[1])}))
        else:
            print(f"[debug] ? {cmd!r} -- type `?` for commands")

    reader_task = asyncio.ensure_future(reader())
    try:
        async for _ in pace(fps):
            while not queue.empty():
                try:
                    apply(queue.get_nowait())
                except ValueError:
                    print("[debug] bad number")
            if stop["quit"]:
                break
            # Ramp target toward goal at the trained per-tick delta.
            for k in bridge.JOINT_KEYS:
                d = goal[k] - target[k]
                step = bridge.DELTA_LIMIT[k]
                target[k] += max(-step, min(step, d))
            target = bridge.clamp_sim(target)
            obs = latest["obs"]
            op.send_action(
                bridge.sim_to_real(target),
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us if obs is not None else None,
            )
    except KeyboardInterrupt:
        print("\n[debug] Ctrl-C -- stopping sends.")
    finally:
        stop["quit"] = True
        reader_task.cancel()
        try:
            if op.active_operator() == me:
                await op.set_active_operator(None)  # release control
            await op.disconnect()
        finally:
            op.close()
    print("[debug] released control.")
    return 0


_JOINT_TB = {
    "dof_slider.pos": "slider (0 far-100 near)", "shoulder_pan.pos": "pan",
    "shoulder_lift.pos": "lift", "elbow_flex.pos": "elbow", "wrist_flex.pos": "wristF",
    "wrist_roll.pos": "wristR", "gripper.pos": "grip",
}
_TB_MAX = 1000


def _sim_to_tb(k: str, v: float) -> int:
    lo, hi = bridge.SIM_LIMITS[k]
    return int(round((v - lo) / (hi - lo) * _TB_MAX))


def _tb_to_sim(k: str, pos: int) -> float:
    lo, hi = bridge.SIM_LIMITS[k]
    return lo + (pos / _TB_MAX) * (hi - lo)


async def run_control_ui(assume_yes: bool) -> int:
    """Manual control via an OpenCV window (trackbar per joint + live camera view). THIS MOVES THE ROBOT."""
    from livekit.portal import (
        Observation, Operator, OperatorConfig, frame_bytes_to_numpy_rgb,
    )
    from utils.common import env, load_env, mint_token, pace

    load_env(_HERE)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    config_path = _HERE.parent / "portal.yaml"

    uncal = [k for k in bridge.JOINT_KEYS if k != "dof_slider.pos" and bridge.OFFSET_REAL[k] == 0.0]
    print("=" * 70)
    print("  MANUAL CONTROL (OpenCV UI) -- THIS MOVES THE REAL ROBOT.")
    print("  Drag a joint's trackbar; the arm ramps to it at the trained speed.")
    if uncal:
        print(f"  WARNING: {len(uncal)} arm/gripper joints uncalibrated (OFFSET_REAL=0); rail is calibrated.")
    print("  Focus the window -- keys: r=rest  h=hold-here  q/esc=quit. Ctrl-C also stops.")
    print("=" * 70)
    if not assume_yes and input("  type 'drive' to proceed: ").strip() != "drive":
        print("[debug] aborted.")
        return 1

    op = Operator(OperatorConfig.from_yaml_file(config_path, room))
    latest: dict[str, Observation | None] = {"obs": None}
    op.on_observation(lambda obs: latest.__setitem__("obs", obs))
    await op.connect(url, mint_token("debug-control", room, name="Debug Control"))
    me = op.local_identity()
    await op.set_active_operator(me)   # claim control
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

    win = "deploy manual control"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for k in bridge.JOINT_KEYS:
        cv2.createTrackbar(_JOINT_TB[k], win, _sim_to_tb(k, target[k]), _TB_MAX, lambda v: None)

    def set_trackbars(pose):
        for k in bridge.JOINT_KEYS:
            cv2.setTrackbarPos(_JOINT_TB[k], win, _sim_to_tb(k, pose[k]))

    mappings = {t: _mapping_for(t) for t, _ in CAMERA_STACK}
    # Speed check: sim commands DELTA_LIMIT * |SCALE| * fps in wire units. Compare
    # against the real robot's measured max; if well below, the hardware can't keep up.
    sim_max_speed = {k: bridge.DELTA_LIMIT[k] * abs(bridge.SCALE[k]) * fps for k in bridge.JOINT_KEYS}
    max_real_speed = {k: 0.0 for k in bridge.JOINT_KEYS}
    prev = {"state": None, "t": None}
    print("[debug] UI open. Drag trackbars to drive; focus the window for keys.")
    try:
        async for _ in pace(fps):
            for k in bridge.JOINT_KEYS:                       # trackbars -> goal
                goal[k] = _tb_to_sim(k, cv2.getTrackbarPos(_JOINT_TB[k], win))
            goal = bridge.clamp_sim(goal)
            for k in bridge.JOINT_KEYS:                       # ramp target -> goal
                step = bridge.DELTA_LIMIT[k]
                target[k] += max(-step, min(step, goal[k] - target[k]))
            target = bridge.clamp_sim(target)
            obs = latest["obs"]

            # measure real achieved joint speed (wire units / s)
            if obs is not None:
                now = time.perf_counter()
                st = {k: float(v) for k, v in obs.state.items()}
                if prev["state"] is not None and prev["t"] is not None:
                    dt = now - prev["t"]
                    if dt > 1e-3:
                        for k in bridge.JOINT_KEYS:
                            if k in st and k in prev["state"]:
                                v = abs(st[k] - prev["state"][k]) / dt
                                if v < 1e4:  # ignore serial-read glitches
                                    max_real_speed[k] = max(max_real_speed[k], v)
                prev["state"], prev["t"] = st, now

            op.send_action(
                bridge.sim_to_real(target),
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us if obs is not None else None,
            )

            tiles = []
            for track, _ in CAMERA_STACK:
                vf = obs.frames.get(track) if obs is not None else None
                if vf is None:
                    tile = np.zeros((256, 256, 3), np.uint8)
                    cv2.putText(tile, f"{track}: no frame", (8, 128),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
                    tile = _labeled(rectify(rgb, mappings[track]), track, 256)
                tiles.append(tile)
            panel = np.zeros((256, 400, 3), np.uint8)
            real = bridge.sim_to_real(target)
            lines = ["joint    tgt   vmax real/sim /s"]
            for k in bridge.JOINT_KEYS:
                n = k.split(".")[0][:6]
                lines.append(f"{n:<6}{real[k]:+7.1f}  {max_real_speed[k]:5.1f}/{sim_max_speed[k]:5.1f}")
            if obs is None:
                lines.append("NO OBS -- is the robot up?")
            lines.append("[c]clear vmax [r]rest [q]quit")
            for i, ln in enumerate(lines):
                cv2.putText(panel, ln, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.imshow(win, np.hstack(tiles + [panel]))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                set_trackbars(bridge.clamp_sim(dict(_SIM_REST)))
            if key == ord("h"):
                set_trackbars(target)   # hold: snap the goal to where we are now
            if key == ord("c"):
                for k in max_real_speed:
                    max_real_speed[k] = 0.0
    except KeyboardInterrupt:
        print("\n[debug] Ctrl-C -- stopping.")
    finally:
        cv2.destroyAllWindows()
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
        description="Debug the deploy camera mapping + joint bridge (no policy, no motion).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--frame", help="raw camera image to run the mapping on (mode 1)")
    p.add_argument("--camera", default="overhead_camera",
                   help="which camera's mapping to use for --frame "
                        "(arm_camera | overhead_camera)")
    p.add_argument("--bridge", action="store_true", help="joint bridge self-test (mode 2)")
    p.add_argument("--live", action="store_true", help="live read-only dump vs the robot (mode 3)")
    p.add_argument("--snapshot", action="store_true",
                   help="pull one RAW frame per camera through Portal into --out (for calibration)")
    p.add_argument("--control", action="store_true",
                   help="MANUAL CONTROL via an OpenCV window (trackbar per joint + live camera; MOVES THE ROBOT)")
    p.add_argument("--cli", action="store_true",
                   help="with --control, use the typed REPL instead of the OpenCV UI")
    p.add_argument("--yes", action="store_true", help="skip the --control safety confirmation")
    p.add_argument("--out", default="/tmp", help="output dir for dumped PNGs")
    p.add_argument("--every", type=float, default=1.0, help="live dump interval, seconds")
    args = p.parse_args()
    out_dir = pathlib.Path(args.out)

    if args.frame:
        return run_frame(args.frame, args.camera, out_dir)
    if args.snapshot:
        return asyncio.run(run_snapshot(out_dir))
    if args.control:
        return asyncio.run(run_control(args.yes) if args.cli else run_control_ui(args.yes))
    if args.live:
        return asyncio.run(run_live(out_dir, args.every))
    return run_bridge()  # default / --bridge


if __name__ == "__main__":
    raise SystemExit(main())
