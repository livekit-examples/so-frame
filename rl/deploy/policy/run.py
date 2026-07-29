"""Drive the SO-101-on-frame rig from a trained RL policy over LiveKit Portal.

Each tick: rectify both raw camera frames to the sim view, stack arm|overhead, run one
inference, integrate the normalized delta into a running sim target, and send joint targets back.

Claims control on startup and starts PAUSED, holding the pose. --no-start-paused drives on
launch; --no-claim stays idle until the web UI claims it.

Debug keys (terminal or --viz window): p = pause/resume, r = reset to rest then hold paused,
0 = ramp the rail alone to wire 0 (end of travel) for re-zeroing, q = quit.
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
    """Rectify both cameras and stack them channel-wise as the sim did; None if a frame is missing."""
    chans = []
    for track, _ in CAMERA_STACK:
        vf = obs.frames.get(track)
        if vf is None:
            return None
        rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
        chans.append(rectify(rgb, mappings[track]))
    return np.concatenate(chans, axis=-1)


VIZ_KEYS = "[r] rest  [0] rail to wire 0  [p] pause/resume  [q] quit"


def _viz_tile(chans, label):
    """One camera view, upscaled to a fixed panel and labelled."""
    import cv2
    bgr = cv2.cvtColor(np.ascontiguousarray(chans), cv2.COLOR_RGB2BGR)
    up = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.putText(up, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return up


def viz_grid(rgb, status: str, net_res: int):
    """Rectified views over what the encoder gets, or a placard saying why not (rgb=None)."""
    import cv2
    if rgb is None:
        grid = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(grid, "NO FRAME INCOMING", (74, 250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (60, 60, 220), 2)
    else:
        wrist, overhead = rgb[..., 0:3], rgb[..., 3:6]
        src = wrist.shape[0]
        # Lower row is the encoder's input resolution, so a mapping that does not match the
        # checkpoint shows up as a visibly upsampled tile.
        small = [cv2.resize(c, (net_res, net_res), interpolation=cv2.INTER_AREA)
                 for c in (wrist, overhead)]
        grid = np.vstack([
            np.hstack([_viz_tile(wrist, f"wrist {src}"), _viz_tile(overhead, f"overhead {src}")]),
            np.hstack([_viz_tile(small[0], f"wrist {net_res}"),
                       _viz_tile(small[1], f"overhead {net_res}")]),
        ])
    bar = np.zeros((30, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, status, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    # Legend: the keys work in this window as well as the terminal.
    legend = np.full((30, grid.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(legend, VIZ_KEYS, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
    return np.vstack([grid, bar, legend])


def build_proprio(sim_qpos: dict[str, float], sim_target: dict[str, float]) -> dict:
    """The proprio fields the env exposes, by name. The policy's ProprioSpec decides order and
    width, and raises if this set does not match what it trained on."""
    return {
        "noisy_qpos": [sim_qpos[k] for k in bridge.JOINT_KEYS],
        "controller.target_qpos": [sim_target[k] for k in bridge.JOINT_KEYS],
    }


async def main(claim: bool = True, start_paused: bool = True, max_lag: float | None = None,
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
    mode = {"state": "run", "label": "reset"}  # run | paused | resetting
    # Joints the current ramp is driving, and where to; anything absent holds its target.
    goal: dict[str, float] = {}
    reset_wait = {"ticks": 0, "deadline": 0.0}
    # Settle tolerance in DELTA_LIMIT units, close to training's initial qpos noise.
    SETTLE_TOL = 1.5
    SETTLE_SECS = 1.0   # measured pose must hold within tolerance this long
    SETTLE_TIMEOUT = 5.0  # give up waiting and pause anyway (e.g. gripper blocked)
    # Running integrated target (sim units), seeded from the first measured qpos on claim.
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
            goal.clear()
            goal.update(bridge.SIM_REST)
            mode["state"], mode["label"] = "resetting", "reset"
            reset_wait["ticks"], reset_wait["deadline"] = 0, 0.0
            print(f"[policy-{NAME}] RESET -- ramping to the rest pose (slider mid), "
                  "will hold PAUSED once settled (stage the scene, then p to run)")
        elif ch == "0":
            # The rail's sim low limit is wire 0, the end of travel for re-zeroing. Arm holds.
            goal.clear()
            goal[bridge.RAIL] = bridge.SIM_LIMITS[bridge.RAIL][0]
            mode["state"], mode["label"] = "resetting", "zero-slider"
            reset_wait["ticks"], reset_wait["deadline"] = 0, 0.0
            print(f"[policy-{NAME}] ZERO SLIDER -- ramping the rail to wire 0 (end of travel) "
                  "at the trained per-tick speed; the arm holds. Will hold PAUSED once settled")
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

    if claim:
        # Self-claim so the keys work without the web UI, but hold the pose by default.
        sim_target = None
        control["enabled"] = True
        mode["state"] = "paused" if start_paused else "run"
        await op.set_active_operator(op.local_identity())
        if start_paused:
            print(f"[policy-{NAME}] connected @ {fps} fps; claimed and PAUSED -- "
                  "press p to start driving (--no-start-paused to drive on launch)")
        else:
            print(f"[policy-{NAME}] connected @ {fps} fps; claimed and DRIVING now "
                  "(--no-start-paused was passed)")
    else:
        print(f"[policy-{NAME}] connected @ {fps} fps; --no-claim: idle until the web UI "
              f"calls run_policy_{NAME}")

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
        print(f"[policy-{NAME}] keys: {VIZ_KEYS}")

    empty = 0
    if viz:
        import cv2
        cv2.namedWindow(f"policy-{NAME} view", cv2.WINDOW_NORMAL)
        print(f"[policy-{NAME}] --viz: showing the two rectified camera views (press q to close)")

    net_res = policy.meta["res"]
    try:
        async for _ in pace(fps):
            quit_requested = False
            while not keys.empty():
                if not apply_key(keys.get_nowait()):
                    quit_requested = True
            if quit_requested:
                break
            obs = latest_obs
            # Rectify before the early-outs so an unclaimed policy still previews its view.
            rgb = build_rgb(obs, mappings) if obs is not None else None

            if viz:
                if not control["enabled"]:
                    status = "UNCLAIMED -- waiting for the web UI to claim (--no-claim was set)"
                elif obs is None:
                    status = "no observations -- is the robot in this room?"
                elif rgb is None:
                    status = "connected, waiting for both camera frames"
                else:
                    status = f"{mode['state'].upper()} -- driving at {fps} Hz"
                cv2.imshow(f"policy-{NAME} view", viz_grid(rgb, status, net_res))
                k = cv2.waitKey(1) & 0xFF
                if k != 255 and not apply_key("q" if k == 27 else chr(k)):
                    break

            if not control["enabled"]:
                continue
            if obs is None:
                empty += 1
                if empty in (fps, fps * 5):
                    print(f"[policy-{NAME}] no observations for {empty // fps}s -- "
                          "is the robot in this room?")
                continue
            empty = 0

            sim_qpos = bridge.real_to_sim(dict(obs.state))
            if rgb is None:
                continue  # wait for both camera frames

            # Seed the target from the current pose on (re)claim/resume. Must sit AFTER
            # viz-window key handling, which can clear sim_target mid-tick.
            if sim_target is None:
                sim_target = dict(sim_qpos)

            if mode["state"] == "paused":
                continue  # servos hold the last target

            if mode["state"] == "resetting":
                # Ramp the goal joints at the trained per-tick speed; the rest hold their target.
                for k in goal:
                    step = bridge.DELTA_LIMIT[k]
                    d = goal[k] - sim_target[k]
                    sim_target[k] = (goal[k] if abs(d) <= step
                                     else sim_target[k] + (step if d > 0 else -step))
                sim_target = bridge.clamp_sim(sim_target)
                op.send_action(
                    bridge.sim_to_real(sim_target),
                    timestamp_us=int(time.time() * 1_000_000),
                    in_reply_to_ts_us=obs.timestamp_us,
                )
                ramped = all(abs(goal[k] - sim_target[k]) < 1e-9 for k in goal)
                if ramped:
                    if reset_wait["deadline"] == 0.0:
                        reset_wait["deadline"] = time.perf_counter() + SETTLE_TIMEOUT
                    settled = all(
                        abs(sim_qpos[k] - goal[k]) < SETTLE_TOL * bridge.DELTA_LIMIT[k]
                        for k in goal if k in sim_qpos)
                    reset_wait["ticks"] = reset_wait["ticks"] + 1 if settled else 0
                    if reset_wait["ticks"] >= int(SETTLE_SECS * fps):
                        why = "settled"
                    elif time.perf_counter() > reset_wait["deadline"]:
                        why = "settle timeout (a joint never reached its goal)"
                    else:
                        continue
                    sim_target = None  # reseed from the settled pose next tick
                    mode["state"] = "paused"
                    resid = ", ".join(
                        f"{k.split('.')[0]}={sim_qpos[k] - goal[k]:+.3f}"
                        for k in goal if k in sim_qpos)
                    print(f"[policy-{NAME}] {mode['label']} {why} -- residual: {resid}")
                    print(f"[policy-{NAME}] holding PAUSED -- stage the scene, "
                          "then p to start the policy")
                continue

            try:
                action = policy.act(rgb, build_proprio(sim_qpos, sim_target))
            except Exception as exc:
                print(f"[policy-{NAME}] inference error: {exc}")
                continue

            # Threshold the gripper to fully open/closed; MUST match how the checkpoint was
            # trained. Off by default, matching the continuous-gripper training default.
            if binary_gripper:
                action[gripper_idx] = 1.0 if float(action[gripper_idx]) > 0 else -1.0

            # Integrate the delta, clamp to joint limits, convert to real units, send.
            stepped = {
                k: sim_target[k] + float(action[i]) * bridge.DELTA_LIMIT[k]
                for i, k in enumerate(bridge.JOINT_KEYS)
            }
            sim_target = bridge.clamp_sim(stepped)
            # --max-lag: cap how far the target may lead the measured pose, per joint, in
            # per-step deltas.
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
    parser.add_argument("--claim", action=argparse.BooleanOptionalAction, default=True,
                        help="claim control on startup, so the debug keys work with no web UI "
                             "(default). --no-claim stays idle until the web UI claims it.")
    parser.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=True,
                        help="claim but hold the pose until you press p (default). "
                             "--no-start-paused MOVES THE ROBOT as soon as frames arrive.")
    parser.add_argument("--max-lag", type=float, default=None,
                        help="anti-oscillation: cap how far the target may lead the measured "
                             "pose, in per-step deltas (e.g. 2.0), so it cannot overshoot "
                             "ahead of a slower arm.")
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=False,
                        help="threshold the gripper action to fully open/closed. Off by default, "
                             "matching the continuous-gripper training default; must match how "
                             "the checkpoint was trained.")
    parser.add_argument("--viz", action="store_true",
                        help="show the two rectified camera views (what the policy sees) in an "
                             "OpenCV window while driving; press q to close")
    args = parser.parse_args()
    if args.checkpoint:
        os.environ["POLICY_CHECKPOINT"] = args.checkpoint
    asyncio.run(main(claim=args.claim, start_paused=args.start_paused, max_lag=args.max_lag,
                     binary_gripper=args.binary_gripper, viz=args.viz))


if __name__ == "__main__":
    cli()
