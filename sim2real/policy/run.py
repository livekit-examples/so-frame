"""Drive the SO-101-on-frame rig from the trained Squint RL policy.

Joins the LiveKit room as a Portal `operator`. While active, each tick: take the
robot's fused observation (7-DOF state + two RAW camera frames), rebuild the sim
view, run one inference, send joint targets back.

Per-tick data flow (the sim2real wiring):

    robot raw frames (640x480, wide FOV)
      -> camera_mapping.apply_mapping(per camera)   # rectify to sim view (128x128)
      -> stack wrist|overhead on channels           # 128x128x6, sim RGB obs
      -> SquintPolicy.act(rgb6, sim_state14)         # squint to 32, encoder+actor
      -> normalized delta action a in [-1, 1]^7
      -> integrate: sim_target += a * DELTA_LIMIT, clamp to joint limits
      -> bridge.sim_to_real(sim_target)              # rad/m -> deg/mm
      -> op.send_action(real_target)

Channel order (wrist 0:3 | overhead 3:6) and sim-unit proprio must match
training. State is qpos(7) + controller target(7) = 14: qpos is the measured
joints; the target is this loop's running integrated target.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

import numpy as np

from livekit.portal import (
    Observation,
    Operator,
    OperatorConfig,
    RpcInvocationData,
    frame_bytes_to_numpy_rgb,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common import env, load_env, mint_token, pace  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import bridge  # noqa: E402
from agent import SquintPolicy  # noqa: E402
from camera_mapping import apply_mapping, load_mapping  # noqa: E402

CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "portal.yaml"
NAME = env("OPERATOR_NAME", "squint")
TITLE = env("OPERATOR_TITLE", "Squint RL")

# Sim RGB channel order: wrist (0:3) then overhead (3:6), matching the sim's
# FlattenRGBDObservationWrapper stack. Each maps to a portal track + its mapping.
CAMERA_STACK = (
    ("arm_camera", "arm_camera_mapping.json"),        # -> sim wrist_camera
    ("overhead_camera", "overhead_camera_mapping.json"),  # -> sim overhead_camera
)


def load_mappings() -> dict[str, dict | None]:
    """Load each camera's rectification mapping. Missing -> plain resize fallback,
    logged loudly (an unrectified view is out-of-distribution and breaks the
    rollout)."""
    out: dict[str, dict | None] = {}
    for track, fname in CAMERA_STACK:
        path = _HERE / fname
        if path.exists():
            out[track] = load_mapping(path)
            print(f"[policy-{NAME}] {track}: using mapping {fname}")
        else:
            out[track] = None
            print(f"[policy-{NAME}] {track}: NO mapping ({fname} missing) -- "
                  "falling back to a plain 128px resize (OUT OF DISTRIBUTION; "
                  "calibrate with rl/maniskill/examples/calibrate_real_camera.py)")
    return out


def rectify(frame_rgb: np.ndarray, mapping: dict | None) -> np.ndarray:
    if mapping is not None:
        return apply_mapping(frame_rgb, mapping)
    import cv2
    return cv2.resize(frame_rgb, (128, 128), interpolation=cv2.INTER_AREA)


def build_rgb6(obs: Observation, mappings: dict[str, dict | None]) -> np.ndarray | None:
    """Rectify both cameras and stack into the 128x128x6 sim RGB obs (trained
    channel order). None if a camera frame is missing."""
    chans = []
    for track, _ in CAMERA_STACK:
        vf = obs.frames.get(track)
        if vf is None:
            return None
        rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
        chans.append(rectify(rgb, mappings[track]))
    return np.concatenate(chans, axis=-1)  # HxWx6


def build_state14(sim_qpos: dict[str, float], sim_target: dict[str, float]) -> np.ndarray:
    """qpos(7) then controller target(7), in joint order -- the 14-dim proprio
    the policy trained on."""
    q = [sim_qpos[k] for k in bridge.JOINT_KEYS]
    t = [sim_target[k] for k in bridge.JOINT_KEYS]
    return np.asarray(q + t, dtype=np.float32)


async def main(auto_claim: bool = False, max_lag: float | None = None,
               binary_gripper: bool = True, viz: bool = False) -> None:
    load_env(_HERE)
    gripper_idx = bridge.JOINT_KEYS.index("gripper.pos")
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    checkpoint = env("SQUINT_CHECKPOINT",
                     str(_HERE.parents[1] / "rl" / "maniskill" / "runs"
                         / "v31_clean_recipe" / "ckpt_best.pt"))

    policy = SquintPolicy(checkpoint, device=env("POLICY_DEVICE", "") or None)
    mappings = load_mappings()

    op = Operator(OperatorConfig.from_yaml_file(CONFIG_PATH, room))
    latest_obs: Observation | None = None
    control = {"enabled": False}
    # Running integrated target (sim units) the delta controller tracks. Seeded
    # from the first measured qpos on claim so the first action is a no-jump delta.
    sim_target: dict[str, float] | None = None

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs

    def on_active_operator_changed(identity: str | None) -> None:
        nonlocal sim_target
        active = identity == op.local_identity()
        if active and not control["enabled"]:
            sim_target = None  # reseed from the next observed pose
        control["enabled"] = active
        print(f"[policy-{NAME}] active operator now: {identity}")

    op.on_observation(on_observation)
    op.on_active_operator_changed(on_active_operator_changed)

    attrs = {
        "vla_demo.kind": "policy",
        "vla_demo.title": TITLE,
        "vla_demo.claim": f"run_policy_{NAME}",
        "vla_demo.stop": f"stop_policy_{NAME}",
        "vla_demo.fields": json.dumps([]),
    }

    async def run_policy(data: RpcInvocationData) -> str:
        nonlocal sim_target
        sim_target = None
        control["enabled"] = True
        me = op.local_identity()
        await op.set_active_operator(me)
        print(f"[policy-{NAME}] claim from '{data.caller_identity}' -> driving")
        return json.dumps({"ok": True, "active": me})

    async def stop_policy(data: RpcInvocationData) -> str:
        control["enabled"] = False
        if op.active_operator() == op.local_identity():
            await op.set_active_operator(None)
        print(f"[policy-{NAME}] stop -> idle")
        return json.dumps({"ok": True})

    op.register_rpc_method(f"run_policy_{NAME}", run_policy)
    op.register_rpc_method(f"stop_policy_{NAME}", stop_policy)

    print(f"[policy-{NAME}] connecting to {url} as 'policy-{NAME}' in room '{room}' ...")
    await op.connect(url, mint_token(f"policy-{NAME}", room, name=TITLE, attributes=attrs))

    if auto_claim:
        sim_target = None                       # reseed from the next observed pose
        control["enabled"] = True
        await op.set_active_operator(op.local_identity())
        print(f"[policy-{NAME}] connected @ {fps} fps; --claim: DRIVING now (self-claimed)")
    else:
        print(f"[policy-{NAME}] connected @ {fps} fps; idle until run_policy_{NAME} "
              "(or pass --claim to drive immediately)")

    empty = 0
    if viz:
        import cv2
        cv2.namedWindow(f"policy-{NAME} view", cv2.WINDOW_NORMAL)
        print(f"[policy-{NAME}] --viz: showing the two rectified camera views (press q to close)")
    try:
        async for _ in pace(fps):
            if not control["enabled"]:
                continue
            obs = latest_obs
            if obs is None:
                empty += 1
                if empty in (fps, fps * 5):
                    print(f"[policy-{NAME}] no observations for {empty // fps}s -- "
                          "is the robot in this room?")
                continue
            empty = 0

            sim_qpos = bridge.real_to_sim(dict(obs.state))
            if sim_target is None:  # seed target from the current pose on (re)claim
                sim_target = dict(sim_qpos)

            rgb6 = build_rgb6(obs, mappings)
            if rgb6 is None:
                continue  # wait for both camera frames

            if viz:
                # Show what the policy consumes: the two rectified 128px views AND
                # their 32px squints (the actual net input), area-downsampled to match
                # SquintPolicy._squint, then upscaled nearest so pixels stay honest.
                def _tile(chans, label):
                    bgr = cv2.cvtColor(np.ascontiguousarray(chans), cv2.COLOR_RGB2BGR)
                    up = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(up, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    return up
                wrist, overhead = rgb6[..., 0:3], rgb6[..., 3:6]
                wrist_sq = cv2.resize(wrist, (32, 32), interpolation=cv2.INTER_AREA)
                overhead_sq = cv2.resize(overhead, (32, 32), interpolation=cv2.INTER_AREA)
                grid = np.vstack([  # 2x2: rectified 128 on top, 32px squints below
                    np.hstack([_tile(wrist, "wrist 128"), _tile(overhead, "overhead 128")]),
                    np.hstack([_tile(wrist_sq, "wrist 32"), _tile(overhead_sq, "overhead 32")]),
                ])
                cv2.imshow(f"policy-{NAME} view", grid)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            state14 = build_state14(sim_qpos, sim_target)
            try:
                action = policy.act(rgb6, state14)  # normalized [-1, 1]^7
            except Exception as exc:
                print(f"[policy-{NAME}] inference error: {exc}")
                continue

            # Near-binary gripper: threshold to sign so it drives fully open/closed,
            # matching training (RandomizationConfig.binary_gripper). MUST match how the
            # loaded checkpoint was trained -- pass --no-binary-gripper for a policy
            # trained with a continuous gripper (e.g. pre-binary v31/v33).
            if binary_gripper:
                action[gripper_idx] = 1.0 if float(action[gripper_idx]) > 0 else -1.0

            # Integrate the normalized delta into the running sim target, clamp
            # to joint limits, convert to real units, send.
            stepped = {
                k: sim_target[k] + float(action[i]) * bridge.DELTA_LIMIT[k]
                for i, k in enumerate(bridge.JOINT_KEYS)
            }
            sim_target = bridge.clamp_sim(stepped)
            # --max-lag: cap how far the target may lead the MEASURED pose (per joint,
            # in per-step deltas). Keeps the target from winding up ahead of a slower
            # arm, so it can't overshoot/oscillate, while the loop keeps flowing at
            # 10 Hz. Off by default.
            if max_lag is not None:
                sim_target = {
                    k: min(sim_qpos[k] + max_lag * bridge.DELTA_LIMIT[k],
                           max(sim_qpos[k] - max_lag * bridge.DELTA_LIMIT[k], sim_target[k]))
                    for k in bridge.JOINT_KEYS
                }
            real_target = bridge.sim_to_real(sim_target)
            op.send_action(
                real_target,
                timestamp_us=int(time.time() * 1_000_000),
                in_reply_to_ts_us=obs.timestamp_us,
            )
    except KeyboardInterrupt:
        print(f"\n[policy-{NAME}] stopping ...")
    finally:
        if viz:
            import cv2
            cv2.destroyAllWindows()
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point (`uv run policy-squint` / `python policy/run.py`)."""
    parser = argparse.ArgumentParser(description="Run the Squint RL policy over LiveKit Portal.")
    parser.add_argument("--checkpoint", default=os.environ.get("SQUINT_CHECKPOINT"),
                        help="Path to the RL checkpoint .pt (env: SQUINT_CHECKPOINT). "
                             "Default: rl/maniskill/runs/v31_clean_recipe/ckpt_best.pt")
    parser.add_argument("--claim", action="store_true",
                        help="claim control on startup and drive immediately (headless; "
                             "no web-ui / RPC needed). MOVES THE ROBOT once it has frames.")
    parser.add_argument("--max-lag", type=float, default=None,
                        help="anti-oscillation: cap how far the target may lead the measured "
                             "pose, in per-step deltas (e.g. 2.0), so it can't wind up ahead of "
                             "a slower arm and overshoot. Keeps the arm flowing at 10 Hz.")
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=True,
                        help="threshold the gripper action to fully open/closed, matching the "
                             "binary-gripper training default. Use --no-binary-gripper for a "
                             "continuous-gripper checkpoint (pre-binary v31/v33).")
    parser.add_argument("--viz", action="store_true",
                        help="show the two rectified camera views (what the policy sees) in an "
                             "OpenCV window while driving; press q to close")
    args = parser.parse_args()
    if args.checkpoint:
        os.environ["SQUINT_CHECKPOINT"] = args.checkpoint
    asyncio.run(main(auto_claim=args.claim, max_lag=args.max_lag,
                     binary_gripper=args.binary_gripper, viz=args.viz))


if __name__ == "__main__":
    cli()
