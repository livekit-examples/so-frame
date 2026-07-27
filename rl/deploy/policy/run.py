"""Drive the SO-101-on-frame rig from a trained RL policy over LiveKit Portal.

Each tick: rectify both raw camera frames to the sim view, stack arm|overhead, run one
inference, integrate the normalized delta into a running sim target, and send joint targets back.

Nothing here is architecture-specific. The checkpoint carries its own encoder kind, resolution
and proprio layout, so there are no --arch / --dino-res flags to get wrong, and the proprio
vector is assembled through the layout the policy was trained on rather than a hardcoded width.

Debug keys (terminal or --viz window): r = reset to rest then hold paused, p = pause/resume,
q = quit.
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
from utils import bridge  # noqa: E402
from utils.camera_mapping import CAMERA_STACK, MAPPINGS_DIR, load_mappings, rectify  # noqa: E402
from utils.common import env, load_env, mint_token, pace  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from agent import Policy  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "portal.yaml"
NAME = env("OPERATOR_NAME", "policy")
TITLE = env("OPERATOR_TITLE", "RL policy")

def build_rgb(obs: Observation, mappings: dict[str, dict | None]) -> np.ndarray | None:
    """Rectify both cameras and stack them channel-wise, as the sim did; None if a frame is missing."""
    chans = []
    for track, _ in CAMERA_STACK:
        vf = obs.frames.get(track)
        if vf is None:
            return None
        rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
        chans.append(rectify(rgb, mappings[track]))
    return np.concatenate(chans, axis=-1)


def build_proprio(sim_qpos: dict[str, float], sim_target: dict[str, float]) -> dict:
    """The proprio fields the env exposes, by name. The policy's own ProprioSpec decides the
    order and width, and raises if this set does not match what it trained on."""
    return {
        "noisy_qpos": [sim_qpos[k] for k in bridge.JOINT_KEYS],
        "controller.target_qpos": [sim_target[k] for k in bridge.JOINT_KEYS],
    }


async def main(auto_claim: bool = False, max_lag: float | None = None,
               binary_gripper: bool = False, viz: bool = False) -> None:
    load_env(_HERE)
    gripper_idx = bridge.JOINT_KEYS.index("gripper.pos")
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    checkpoint = env("POLICY_CHECKPOINT", required=True)

    policy = Policy(checkpoint, device=env("POLICY_DEVICE", "") or None)
    mappings = load_mappings(label=f"policy-{NAME}")

    op = Operator(OperatorConfig.from_yaml_file(CONFIG_PATH, room))
    latest_obs: Observation | None = None
    control = {"enabled": False}
    mode = {"state": "run"}  # run | paused | resetting
    reset_wait = {"ticks": 0, "deadline": 0.0}
    # Settle tolerance in DELTA_LIMIT units, close to training's initial qpos noise.
    SETTLE_TOL = 1.5
    SETTLE_SECS = 1.0   # measured pose must hold within tolerance this long
    SETTLE_TIMEOUT = 5.0  # give up waiting and pause anyway (e.g. gripper blocked)
    # Running integrated target (sim units); seeded from the first measured qpos
    # on claim so the first action is a no-jump delta.
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

    def apply_key(ch: str) -> bool:
        """Handle one debug key; returns False to quit the run loop."""
        nonlocal sim_target
        ch = ch.lower()
        if ch == "q":  # not esc: arrow keys in cbreak stdin start with \x1b
            return False
        if ch == "p":
            if mode["state"] == "paused":
                mode["state"] = "run"
                sim_target = None  # reseed so resume is a no-jump
                print(f"[policy-{NAME}] resumed")
            else:
                mode["state"] = "paused"
                print(f"[policy-{NAME}] PAUSED -- holding pose (p to resume, r to reset)")
        elif ch == "r":
            mode["state"] = "resetting"
            reset_wait["ticks"], reset_wait["deadline"] = 0, 0.0
            print(f"[policy-{NAME}] RESET -- ramping to the rest pose (slider mid), "
                  "will hold PAUSED once settled (stage the scene, then p to run)")
        return True

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
        mode["state"] = "run"
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
        sim_target = None
        control["enabled"] = True
        await op.set_active_operator(op.local_identity())
        print(f"[policy-{NAME}] connected @ {fps} fps; --claim: DRIVING now (self-claimed)")
    else:
        print(f"[policy-{NAME}] connected @ {fps} fps; idle until run_policy_{NAME} "
              "(or pass --claim to drive immediately)")

    # Raw single-key capture from the terminal (cbreak = no Enter; restored on exit).
    keys: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    stdin_fd, old_termios = None, None
    if sys.stdin.isatty():
        import termios
        import tty
        stdin_fd = sys.stdin.fileno()
        old_termios = termios.tcgetattr(stdin_fd)
        tty.setcbreak(stdin_fd)
        loop.add_reader(
            stdin_fd,
            lambda: keys.put_nowait(os.read(stdin_fd, 1).decode(errors="ignore")),
        )
        print(f"[policy-{NAME}] keys: r=reset-to-rest+restart  p=pause/resume  q=quit")

    empty = 0
    if viz:
        import cv2
        cv2.namedWindow(f"policy-{NAME} view", cv2.WINDOW_NORMAL)
        print(f"[policy-{NAME}] --viz: showing the two rectified camera views (press q to close)")
    try:
        async for _ in pace(fps):
            quit_requested = False
            while not keys.empty():
                if not apply_key(keys.get_nowait()):
                    quit_requested = True
            if quit_requested:
                break
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

            rgb = build_rgb(obs, mappings)
            if rgb is None:
                continue  # wait for both camera frames

            if viz:
                # Show the two rectified 128px views and their 32px squints (the net input).
                def _tile(chans, label):
                    bgr = cv2.cvtColor(np.ascontiguousarray(chans), cv2.COLOR_RGB2BGR)
                    up = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(up, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    return up
                wrist, overhead = rgb[..., 0:3], rgb[..., 3:6]
                wrist_sq = cv2.resize(wrist, (32, 32), interpolation=cv2.INTER_AREA)
                overhead_sq = cv2.resize(overhead, (32, 32), interpolation=cv2.INTER_AREA)
                grid = np.vstack([  # rectified 128 on top, 32px squints below
                    np.hstack([_tile(wrist, "wrist 128"), _tile(overhead, "overhead 128")]),
                    np.hstack([_tile(wrist_sq, "wrist 32"), _tile(overhead_sq, "overhead 32")]),
                ])
                cv2.imshow(f"policy-{NAME} view", grid)
                k = cv2.waitKey(1) & 0xFF
                if k != 255 and not apply_key("q" if k == 27 else chr(k)):
                    break

            # Seed the target from the current pose on (re)claim/resume. Must sit
            # AFTER viz-window key handling, which can clear sim_target mid-tick.
            if sim_target is None:
                sim_target = dict(sim_qpos)

            if mode["state"] == "paused":
                continue  # servos hold the last target

            if mode["state"] == "resetting":
                # Ramp the target to the rest pose at the trained per-tick speed.
                for k in bridge.JOINT_KEYS:
                    step = bridge.DELTA_LIMIT[k]
                    d = bridge.SIM_REST[k] - sim_target[k]
                    sim_target[k] = (bridge.SIM_REST[k] if abs(d) <= step
                                     else sim_target[k] + (step if d > 0 else -step))
                sim_target = bridge.clamp_sim(sim_target)
                op.send_action(
                    bridge.sim_to_real(sim_target),
                    timestamp_us=int(time.time() * 1_000_000),
                    in_reply_to_ts_us=obs.timestamp_us,
                )
                ramped = all(abs(bridge.SIM_REST[k] - sim_target[k]) < 1e-9
                             for k in bridge.JOINT_KEYS)
                if ramped:
                    if reset_wait["deadline"] == 0.0:
                        reset_wait["deadline"] = time.perf_counter() + SETTLE_TIMEOUT
                    settled = all(
                        abs(sim_qpos[k] - bridge.SIM_REST[k]) < SETTLE_TOL * bridge.DELTA_LIMIT[k]
                        for k in bridge.JOINT_KEYS if k in sim_qpos)
                    reset_wait["ticks"] = reset_wait["ticks"] + 1 if settled else 0
                    if reset_wait["ticks"] >= int(SETTLE_SECS * fps):
                        why = "settled"
                    elif time.perf_counter() > reset_wait["deadline"]:
                        why = "settle timeout (a joint never reached rest)"
                    else:
                        continue
                    sim_target = None  # reseed from the settled pose next tick
                    mode["state"] = "paused"
                    resid = ", ".join(
                        f"{k.split('.')[0]}={sim_qpos[k] - bridge.SIM_REST[k]:+.3f}"
                        for k in bridge.JOINT_KEYS if k in sim_qpos)
                    print(f"[policy-{NAME}] reset {why} -- residual from rest: {resid}")
                    print(f"[policy-{NAME}] holding PAUSED -- stage the scene, "
                          "then p to start the policy")
                continue

            try:
                action = policy.act(rgb, build_proprio(sim_qpos, sim_target))
            except Exception as exc:
                print(f"[policy-{NAME}] inference error: {exc}")
                continue

            # Threshold the gripper to fully open/closed; MUST match how the checkpoint was
            # trained. Training went back to the continuous gripper, so this is off by default;
            # pass --binary-gripper for the v35-era binary-jaw checkpoints.
            if binary_gripper:
                action[gripper_idx] = 1.0 if float(action[gripper_idx]) > 0 else -1.0

            # Integrate the delta, clamp to joint limits, convert to real units, send.
            stepped = {
                k: sim_target[k] + float(action[i]) * bridge.DELTA_LIMIT[k]
                for i, k in enumerate(bridge.JOINT_KEYS)
            }
            sim_target = bridge.clamp_sim(stepped)
            # --max-lag: cap how far the target may lead the measured pose (per joint,
            # in per-step deltas), so it can't wind up ahead of a slower arm.
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
        if stdin_fd is not None and old_termios is not None:
            import termios
            loop.remove_reader(stdin_fd)
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_termios)
        if viz:
            import cv2
            cv2.destroyAllWindows()
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(description="Run a trained RL policy on the real rig over LiveKit Portal.")
    parser.add_argument("--checkpoint", default=os.environ.get("POLICY_CHECKPOINT"),
                        help="path to a trained checkpoint .pt (env: POLICY_CHECKPOINT). It "
                             "records its own architecture, resolution and proprio layout.")
    parser.add_argument("--claim", action="store_true",
                        help="claim control on startup and drive immediately (headless; "
                             "no web-ui / RPC needed). MOVES THE ROBOT once it has frames.")
    parser.add_argument("--max-lag", type=float, default=None,
                        help="anti-oscillation: cap how far the target may lead the measured "
                             "pose, in per-step deltas (e.g. 2.0), so it can't wind up ahead of "
                             "a slower arm and overshoot. Keeps the arm flowing at 10 Hz.")
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=False,
                        help="threshold the gripper action to fully open/closed. Off by default, "
                             "matching the continuous-gripper training default; pass "
                             "--binary-gripper for a v35-era binary-gripper checkpoint.")
    parser.add_argument("--viz", action="store_true",
                        help="show the two rectified camera views (what the policy sees) in an "
                             "OpenCV window while driving; press q to close")
    args = parser.parse_args()
    if args.checkpoint:
        os.environ["POLICY_CHECKPOINT"] = args.checkpoint
    asyncio.run(main(auto_claim=args.claim, max_lag=args.max_lag,
                     binary_gripper=args.binary_gripper, viz=args.viz))


if __name__ == "__main__":
    cli()
