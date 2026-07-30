"""Drive the SO-101-on-frame rig from a trained RL policy over LiveKit Portal.

Each tick: rectify both raw camera frames to the sim view, stack arm|overhead, run one
inference, integrate the normalized delta into a running sim target, and send joint targets back.

Claims control on startup and starts PAUSED, holding the pose. --no-start-paused drives on
launch; --no-claim stays idle until the web UI claims it.

An action is a delta per control period, so it is a velocity: it keeps being integrated every tick
until a new one replaces it, as sim did. EVERY joint sustains, the gripper included, because sim
made no distinction between deciding and integrating and the relative timing of the arm's approach
against the jaw's closure is what a grasp is.

What bounds the arm is --max-lag, in action steps: the target advances only while it is within that
far of the measured pose, so it can never run away from a slower arm. A new decision happens once
the arm is back inside the budget, or after a second of waiting, which also releases the target for
that tick so a joint that can never arrive cannot freeze the arm. Keep the budget wide: it is also
the decision gate, so a tight one throttles the visual feedback rate, and a policy closing its loop
slower than it trained at overshoots whatever it was reaching for. The gripper is exempt from the
shared budget and carries its own lead cap, GRIPPER_LEAD.

--viz changes the budget and the rail's step size live while the arm moves; if the timeout count
keeps climbing, the budget is too tight for this arm.

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
from utils.camera_mapping import (  # noqa: E402
    CAMERA_STACK, MAPPINGS_DIR, load_mappings, rectify, stack_out_size,
)
from utils.common import env, load_env, mint_token, pace  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from agent import Policy  # noqa: E402

_HERE = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "portal.yaml"
NAME = env("OPERATOR_NAME", "policy")
TITLE = env("OPERATOR_TITLE", "RL policy")

CHECKPOINT_DIR = _HERE.parent / "checkpoints"
# --arch shorthand -> (filename, expected kind, expected pool). The expectations are checked
# after loading: this selects a FILE, it does not tell the policy what architecture to build,
# which the checkpoint still declares for itself. So a file swapped for a different
# architecture fails loudly here instead of running on features it was not trained on.
CHECKPOINTS: dict[str, tuple[str, str, str | None]] = {
    "squint":           ("squint_ckpt.pt",            "squint",      None),
    "dino_patch":       ("dino_patch_policy_ckpt.pt", "dino_patch",  None),
    "dino_global_mean": ("dino_global_mean_ckpt.pt",  "dino_global", "mean"),
    "dino_cls":         ("dino_cls_ckpt.pt",          "dino_global", "cls"),
}

# Joints the lag budget covers: everything but the gripper. Its whole range is 9.6 steps, so a jaw
# closed on the object sits several steps short of its commanded position for as long as it holds,
# and no useful budget would ever clear. Holding its target back would also bleed grip force, since
# a position servo only pushes as hard as the distance it is asked to close.
GATED: tuple[str, ...] = tuple(k for k in bridge.JOINT_KEYS if k != "gripper.pos")

# How far the gripper target may lead the measured jaw, in action steps. This is the gripper's own
# budget, and it exists because the jaw sustains its action every tick like every other joint (see
# the integration block) while being exempt from the shared gate above.
#
# Without a bound, one action held over N ticks winds the target across N steps of a range only 9.6
# steps wide, so a single decision at a low decision rate could command most of a closure. The cap
# is deliberately wide rather than tight: the lead IS the grip force, so 3 steps (0.6 rad of
# commanded closure past where the jaw actually sits) keeps the servo pushing hard on a held object
# while still preventing windup.
GRIPPER_LEAD = 3.0

# The rail's trained per-step delta, in mm. A full-command action moves the carriage this far in one
# control period, so it is also its top speed in mm per 100 ms.
RAIL_STEP_MM = bridge.DELTA_LIMIT[bridge.RAIL] * 1000.0


def deltas(rail_step_mm: float) -> dict[str, float]:
    """Per-step deltas, with the rail's overridable.

    Everything else is the trained contract and is not adjustable here: changing what an action
    means physically is a sim2real mismatch, not a tuning knob. The rail is the exception worth
    exposing, because 7 mm per step is the one figure in that contract taken off a control UI
    rather than measured, and it sets both the carriage's top speed and the unit its lag is
    counted in. Move it and you are running the policy on a different action space than it trained
    on, deliberately.
    """
    out = dict(bridge.DELTA_LIMIT)
    out[bridge.RAIL] = max(float(rail_step_mm), 0.1) / 1000.0
    return out


def joint_lag(sim_target: dict, sim_qpos: dict, delta: dict) -> tuple[float, str]:
    """Worst lag across the gated joints, in action steps, with the joint responsible."""
    worst, who = 0.0, "-"
    for k in GATED:
        if k not in sim_qpos:
            continue
        d = abs(sim_target[k] - sim_qpos[k]) / delta[k]
        if d > worst:
            worst, who = d, k.split(".")[0]
    return worst, who


class FrameCache:
    """The latest rectified view per camera, stacked channel-wise as sim did.

    Holds each camera separately because requiring both frames in the SAME observation caps the
    decision rate at however often that happens to occur. The two cameras are independent USB
    devices published as separate tracks, so their frames do not reliably land in one observation:
    with the streams unaligned, only some observations carry both and the policy loses a decision on
    every one that does not, no matter how fast inference is or how open the lag gate is. Caching
    decouples the decision rate from that alignment, and collapses to the old behaviour exactly when
    frames do arrive together.

    ``out_size`` forces one resolution across cameras. Without it a camera on a mapping and a camera
    on the plain-resize fallback produce different sizes and the stack fails outright.

    ``MAX_AGE`` is what keeps this honest. Pairing a fresh frame with a slightly older one beats
    losing the decision, because skipping one leaves an action in force that is older still. But an
    unbounded cache would let a dead camera feed the policy a frozen view forever, so a track older
    than MAX_AGE invalidates the whole stack: the same rule the None return enforced, that a lost
    camera must not mean the arm keeps moving blind.
    """

    MAX_AGE = 0.25   # s. ~2.5 ticks at 10 Hz: tolerate a skipped frame, never a dead camera.

    def __init__(self, mappings: dict[str, dict | None], out_size: int):
        self._mappings, self._out_size = mappings, out_size
        self._held: dict[str, tuple[np.ndarray, float]] = {}

    def update(self, obs: Observation | None, now: float) -> None:
        """Rectify and keep whichever camera frames this observation carried."""
        if obs is None:
            return
        for track, _ in CAMERA_STACK:
            vf = obs.frames.get(track)
            if vf is None:
                continue
            rgb = frame_bytes_to_numpy_rgb(vf.data, vf.width, vf.height)
            self._held[track] = (
                rectify(rgb, self._mappings[track], out_size=self._out_size), now,
            )

    def stack(self, now: float) -> np.ndarray | None:
        """The current stack, or None if any camera has nothing fresh enough to contribute."""
        parts = []
        for track, _ in CAMERA_STACK:
            held = self._held.get(track)
            if held is None or now - held[1] > self.MAX_AGE:
                return None
            parts.append(held[0])
        return np.concatenate(parts, axis=-1)


def build_proprio(sim_qpos: dict[str, float], sim_target: dict[str, float]) -> dict:
    """The proprio fields the env exposes, by name. The policy's ProprioSpec decides order and
    width, and raises if this set does not match what it trained on."""
    return {
        "noisy_qpos": [sim_qpos[k] for k in bridge.JOINT_KEYS],
        "controller.target_qpos": [sim_target[k] for k in bridge.JOINT_KEYS],
    }


async def main(claim: bool = True, start_paused: bool = True,
               max_lag: float = 3.0, rail_step: float = RAIL_STEP_MM,
               viz: bool = False, arch: str | None = None) -> None:
    load_env(_HERE)
    delta = deltas(rail_step)
    if abs(rail_step - RAIL_STEP_MM) > 1e-6:
        print(f"[policy-{NAME}] rail step {rail_step:.1f} mm, not the trained "
              f"{RAIL_STEP_MM:.1f} mm: the rail's action space differs from training")
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))
    checkpoint = env("POLICY_CHECKPOINT", required=True)

    policy = Policy(checkpoint, device=env("POLICY_DEVICE", "") or None)
    if arch:
        _, want_kind, want_pool = CHECKPOINTS[arch]
        got_kind = policy.meta["kind"]
        got_pool = (policy.meta.get("encoder_kwargs") or {}).get("pool")
        if got_kind != want_kind or (want_pool is not None and got_pool != want_pool):
            raise SystemExit(
                f"[policy-{NAME}] --arch {arch} expects {want_kind}"
                + (f"/pool={want_pool}" if want_pool else "")
                + f", but {checkpoint} contains {got_kind}"
                + (f"/pool={got_pool}" if got_pool else "")
                + ". The file was replaced with a different architecture; pass --checkpoint "
                  "explicitly if that is deliberate."
            )
    mappings = load_mappings(label=f"policy-{NAME}")

    net_res = policy.meta["res"]
    # One resolution for every camera. Prefer what the mappings were fitted at; with none fitted,
    # fall back to the checkpoint's own input size rather than a constant, so an uncalibrated run is
    # at least not upsampling on top of being out of distribution.
    stack_res = stack_out_size(mappings, default=net_res)
    if stack_res != net_res:
        print(f"[policy-{NAME}] camera views are {stack_res}px but the encoder wants {net_res}px; "
              f"it will resample. Refit the mappings with --sim-size {net_res} to feed it natively.")
    frames = FrameCache(mappings, stack_res)

    # Warm the encoder BEFORE connecting. dino_patch fetches its frozen backbone from torch.hub on
    # the FIRST act(), which blocks for seconds; doing that mid-episode stalls the event loop and
    # the portal stops delivering observations while it happens.
    _t0 = time.perf_counter()
    policy.act(np.zeros((stack_res, stack_res, 3 * policy.num_cams), np.uint8),
               build_proprio(bridge.SIM_REST, bridge.SIM_REST))
    first_ms = (time.perf_counter() - _t0) * 1e3
    _t0 = time.perf_counter()
    policy.act(np.zeros((stack_res, stack_res, 3 * policy.num_cams), np.uint8),
               build_proprio(bridge.SIM_REST, bridge.SIM_REST))
    steady_ms = (time.perf_counter() - _t0) * 1e3
    budget_ms = 1000.0 / fps
    print(f"[policy-{NAME}] encoder warm: first call {first_ms:.0f} ms, then {steady_ms:.0f} ms "
          f"per tick of a {budget_ms:.0f} ms budget")
    if steady_ms > 0.6 * budget_ms:
        print(f"[policy-{NAME}] inference is using >60% of the tick budget. It runs off the event "
              "loop so the stream stays alive, but set POLICY_DEVICE=mps for real headroom.")

    op = Operator(OperatorConfig.from_yaml_file(CONFIG_PATH, room))
    latest_obs: Observation | None = None
    control = {"enabled": False}
    mode = {"state": "run", "label": "reset"}  # run | paused | resetting
    # Joints the current ramp is driving, and where to; anything absent holds its target.
    goal: dict[str, float] = {}
    reset_wait = {"ticks": 0, "deadline": 0.0}
    # When an r/k/0 ramp counts as arrived. Tolerance in DELTA_LIMIT units, close to training's
    # initial qpos noise. Unrelated to the decision gate below.
    RESET_TOL = 1.5
    RESET_HOLD = 1.0     # measured pose must hold within tolerance this long
    RESET_TIMEOUT = 5.0  # give up waiting and pause anyway (e.g. gripper blocked)
    # Ceiling on how long the lag gate may hold the target still before it yields for one tick,
    # deciding and advancing anyway, so a joint that never arrives cannot wedge the arm for good.
    LAG_TIMEOUT = 1.0
    # Running integrated target (sim units), seeded from the first measured qpos on claim.
    sim_target: dict[str, float] | None = None

    obs_stats = {"t": 0.0}   # perf_counter of the last observation, for the age readout
    # Decision gate. The loop keeps ticking at fps (stream, keys, viz); inference happens only
    # once the arm has caught up with the target it was last given.
    #
    #   lag / worst  the worst joint's shortfall in action-step units, and which joint. Read it
    #                against the budget: the gate freezes the target once lag exceeds it, so this
    #                pins near whatever --max-lag is and only measures the ARM with the gate open.
    #   timeouts     times a joint never arrived and the gate yielded anyway. Nonzero is a fault.
    #   ms_avg       EMA of inference wall time, measured around the whole await so it includes the
    #                thread hand-off the loop waits on, not just the forward pass.
    decide = {"last": 0.0, "n": 0, "timeouts": 0, "lag": 0.0, "worst": "-",
              "ms_avg": 0.0, "stamps": []}

    # Decisions per second over the last DECIDE_WINDOW seconds, not since the first decision ever.
    # A lifetime average is the wrong number here: pause, reset, unclaimed and no-frames ticks all
    # skip the decision block, so time spent staging the scene stayed in the denominator forever and
    # made a healthy loop read slow. Windowed, it reports what the loop is doing now, and reads 0
    # while the policy is not driving, which is the truth rather than a diluted average.
    DECIDE_WINDOW = 5.0

    def decide_rate(now: float) -> float:
        stamps = decide["stamps"]
        while stamps and now - stamps[0] > DECIDE_WINDOW:
            stamps.pop(0)
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        return (len(stamps) - 1) / span if span > 1e-6 else 0.0
    last_action = None      # the velocity currently in force; integrated every tick until replaced

    def on_observation(obs: Observation) -> None:
        nonlocal latest_obs
        latest_obs = obs
        obs_stats["t"] = time.perf_counter()

    def on_active_operator_changed(identity: str | None) -> None:
        nonlocal sim_target, last_action
        active = identity == op.local_identity()
        if active and not control["enabled"]:
            sim_target, last_action = None, None  # reseed from the next observed pose
        control["enabled"] = active
        print(f"[policy-{NAME}] active operator now: {identity}")

    op.on_observation(on_observation)
    op.on_active_operator_changed(on_active_operator_changed)

    def apply_key(ch: str) -> bool:
        """Handle one debug key; returns False to quit the run loop."""
        nonlocal sim_target, last_action
        ch = ch.lower()
        # Every key below either stops or re-stages the arm, so whatever velocity was in force must
        # not survive: without this, unpausing resumes gliding on an action from before the pause.
        last_action = None
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
        elif ch == "k":
            # PARK, not rest: a folded pose to leave the arm idle in, gripper open. Unpausing from
            # here starts the policy outside its trained initial state, so stage with r instead.
            goal.clear()
            goal.update(bridge.clamp_sim(bridge.real_to_sim(bridge.PARK_REAL)))
            mode["state"], mode["label"] = "resetting", "park"
            reset_wait["ticks"], reset_wait["deadline"] = 0, 0.0
            print(f"[policy-{NAME}] PARK -- ramping to the folded park pose, gripper open; "
                  "will hold PAUSED once settled (r stages a rollout instead)")
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
        nonlocal sim_target, last_action
        sim_target, last_action = None, None
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
        print(f"[policy-{NAME}] keys: p pause/resume  r rest  k park  0 zero rail  q quit")
    else:
        # Worth saying out loud: with stdin redirected there is no way to press anything, and the
        # failure is otherwise indistinguishable from a key handler that is broken.
        print(f"[policy-{NAME}] stdin is not a tty, so the terminal keys are OFF"
              + (" -- use the --viz window's keys or buttons" if viz else
                 " -- run it in a terminal, or pass --viz for a window with the keys as buttons"))

    empty = 0
    win = None
    if viz:
        try:
            from PySide6 import QtWidgets

            import viz as vizmod          # policy/viz.py, needs `uv sync --group viz`
        except ImportError as exc:
            raise SystemExit(f"--viz needs the ui toolkit: uv sync --group viz  ({exc})")
        QtWidgets.QApplication([])
        win = vizmod.Window([t for t, _ in CAMERA_STACK],
                            [k.split(".")[0] for k in bridge.JOINT_KEYS],
                            max_lag=max_lag, rail_step=rail_step, gated=GATED)
        win.show()
        print(f"[policy-{NAME}] --viz open")

    try:
        async for tick in pace(fps):
            quit_requested = False
            while not keys.empty():
                if not apply_key(keys.get_nowait()):
                    quit_requested = True
            if quit_requested:
                break
            obs = latest_obs
            # Rectify before the early-outs so an unclaimed policy still previews its view.
            tick_t = time.perf_counter()
            frames.update(obs, tick_t)
            rgb = frames.stack(tick_t)

            if win is not None:
                # The window owns the budgets and the rail step once it is open; the flags are
                # starting values.
                max_lag = win.max_lag()
                rail_step = win.rail_step()
                delta = deltas(rail_step)
                if not control["enabled"]:
                    status, alarm = "Unclaimed", True
                elif obs is None:
                    status, alarm = "No observations", True
                elif rgb is None:
                    status, alarm = "Waiting for both cameras", True
                else:
                    status, alarm = mode["state"].capitalize(), False
                rows = []
                if sim_target is not None and obs is not None:
                    meas = bridge.real_to_sim(dict(obs.state))
                    rows = [
                        (k.split(".")[0],
                         None if last_action is None else last_action[i],
                         (sim_target[k] - meas[k]) / delta[k] if k in meas else 0.0)
                        for i, k in enumerate(bridge.JOINT_KEYS)
                    ]
                win.show_stack(rgb)
                win.set_state(
                    status, alarm,
                    f"{decide_rate(time.perf_counter()):.1f}/{fps} dec/s"
                    f"    infer {decide['ms_avg']:.0f} ms of {budget_ms:.0f}"
                    f"    lag {decide['lag']:.2f}/{max_lag:.2f} ({decide['worst']})"
                    f"    rail step {rail_step:.1f} mm"
                    f"    {stack_res} -> {net_res} px",
                    rows, threshold=max_lag)
                win.step()
                if win.closed:
                    break
                for ch in win.take_keys():
                    if not apply_key(ch):
                        raise KeyboardInterrupt

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

            # Seed the target from the current pose on (re)claim/resume. Must sit AFTER
            # viz-window key handling, which can clear sim_target mid-tick.
            if sim_target is None:
                sim_target = dict(sim_qpos)

            if tick % fps == 0:
                now_t = time.perf_counter()
                age = (now_t - obs_stats["t"]) * 1e3 if obs_stats["t"] else -1
                print(f"[policy-{NAME}] {mode['state']:<8} "
                      f"{decide_rate(now_t):>4.1f}/{fps} dec/s   "
                      f"infer {decide['ms_avg']:>3.0f} ms   "
                      f"lag {decide['lag']:.2f}/{max_lag:.2f} {decide['worst']:<13} "
                      f"obs {age:>3.0f} ms")
                # Anything that needs a human is its own line, so steady state stays one row and a
                # fault cannot hide in the middle of it. Silence here means nothing is wrong.
                if rgb is None:
                    missing = [t for t, _ in CAMERA_STACK
                               if obs is None or obs.frames.get(t) is None]
                    print(f"[policy-{NAME}]   no view: waiting on {missing or 'both cameras'}")
                if decide["ms_avg"] > budget_ms:
                    print(f"[policy-{NAME}]   inference {decide['ms_avg']:.0f} ms exceeds the "
                          f"{budget_ms:.0f} ms tick; decisions cannot reach {fps}/s. "
                          "POLICY_DEVICE=mps, or drop --viz.")
                if decide["timeouts"]:
                    print(f"[policy-{NAME}]   {decide['timeouts']} gate timeouts on "
                          f"{decide['worst']}: it is not reaching its target. Raise --max-lag; a "
                          "budget under the arm's natural trail throttles every joint.")

            if mode["state"] == "paused":
                # Hold by COMMANDING the current target every tick, not by falling silent.
                # Silence stalled the stream: pause was the only state that never called
                # send_action, and every action carries in_reply_to_ts_us, so with nothing
                # replying the portal's state buffer fills ("state buffer full (5), dropped 1
                # oldest") and stops pairing states with frames. Observations dry up, the view
                # freezes, and it only recovers on reset because reset resumes sending.
                # Re-sending the same target is a no-op for the servos, which already latch it.
                op.send_action(
                    bridge.sim_to_real(sim_target),
                    timestamp_us=int(time.time() * 1_000_000),
                    in_reply_to_ts_us=obs.timestamp_us,
                )
                continue

            if mode["state"] == "resetting":
                # Ramp the goal joints at the trained per-tick speed; the rest hold their target.
                for k in goal:
                    step = delta[k]
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
                        reset_wait["deadline"] = time.perf_counter() + RESET_TIMEOUT
                    settled = all(
                        abs(sim_qpos[k] - goal[k]) < RESET_TOL * delta[k]
                        for k in goal if k in sim_qpos)
                    reset_wait["ticks"] = reset_wait["ticks"] + 1 if settled else 0
                    if reset_wait["ticks"] >= int(RESET_HOLD * fps):
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

            # An action is a delta per CONTROL PERIOD, which is to say a velocity: sim integrated
            # one every 100 ms step. So an action that is still in force keeps being integrated
            # every tick here too, and a decision replaces the velocity rather than being the only
            # thing that produces motion.
            #
            # Freezing the target between decisions is what made the rail crawl. One step is 7 mm,
            # so a frozen target moves the carriage at (decisions/s x 7 mm): 1.6 cm/s at 2.3
            # decisions/s, against the 7 cm/s it was trained at. Worse, the decision rate is global
            # while the mechanisms are not, so a slow arm joint throttled the rail to its rate.
            #
            # What bounds it is the lag budget: the target only advances while every gated joint is
            # within `max_lag` steps of where it was last put. So the target leads the arm by at
            # most that much and no more, a velocity-clamped ramp rather than the runaway that had
            # the jaw closing after it passed the cube.
            lag, worst = joint_lag(sim_target, sim_qpos, delta)
            decide["lag"], decide["worst"] = lag, worst
            now = time.perf_counter()
            over = lag > max_lag
            # The budget is a wait, not a stop. A joint that physically cannot arrive -- against a
            # mechanical stop, a dead servo, or simply drooping further behind its target than the
            # budget is wide -- would otherwise hold the target still forever, and the arm would
            # freeze a tick or two after the first decision with nothing but the gripper still
            # moving. So once a wait has run LAG_TIMEOUT the budget yields for that tick: the
            # decision fires AND the target advances, which is what makes the gate a rate limiter
            # rather than a latch. Creeping a step a second is the escape hatch, not a working rate;
            # if the timeouts keep counting up, the arm's steady-state error is wider than --max-lag
            # and the budget is what needs raising.
            timed_out = over and now - decide["last"] > LAG_TIMEOUT
            behind = over and not timed_out

            # Decide once the arm has caught up, or once the wait has timed out.
            if not behind:
                if over:
                    decide["timeouts"] += 1
                # Only inference needs pixels. r/k/0 are joint-only ramps and must not be blocked
                # on a camera, or a dropped frame silently turns them into no-ops.
                if rgb is not None:
                    try:
                        # OFF the event loop. policy.act() is synchronous torch; on CPU it is ~90 ms
                        # of a 100 ms tick, and running it inline meant the loop never reached an
                        # await, so the portal's observation callbacks never fired and the video
                        # froze for exactly as long as the policy was driving. In a thread,
                        # callbacks keep arriving during inference.
                        _infer_t0 = time.perf_counter()
                        action = await asyncio.to_thread(
                            policy.act, rgb, build_proprio(sim_qpos, sim_target))
                        ms = (time.perf_counter() - _infer_t0) * 1e3
                        decide["ms_avg"] = ms if not decide["ms_avg"] else (
                            0.8 * decide["ms_avg"] + 0.2 * ms)
                    except Exception as exc:
                        print(f"[policy-{NAME}] inference error: {exc}")
                        action = None
                    if action is not None:
                        last_action = action
                        decide["last"], decide["n"] = now, decide["n"] + 1
                        decide["stamps"].append(now)

            # Integrate whatever action is in force. Without pixels nothing advances: a lost camera
            # must not mean the arm keeps moving blind on a stale command.
            #
            # The GRIPPER SUSTAINS TOO, on every tick, and it did not use to. Applying it only on
            # the tick it was decided made the jaw move at the decision rate while the arm moved at
            # the tick rate, and on this rig those differ by about 4x. Sim never had that split: one
            # control step was both a decision and an integration, so a full closure took 9.6 steps
            # and just under a second, with the arm advancing at the same time. Rate-limiting only
            # the jaw stretched that to several seconds of the arm continuing at full trained speed,
            # which is the grasp closing after the gripper has already passed the cube.
            if last_action is not None and rgb is not None:
                stepped = dict(sim_target)
                for i, k in enumerate(bridge.JOINT_KEYS):
                    if k in GATED and behind:
                        continue    # budget spent; wait for the joints, unless the wait timed out
                    stepped[k] += float(last_action[i]) * delta[k]
                    if k not in GATED and k in sim_qpos:
                        # The jaw's own budget, since the shared gate cannot cover it. Bounded
                        # against the measured jaw so a sustained action cannot wind the target
                        # across a range that is only 9.6 steps wide. See GRIPPER_LEAD.
                        lead = GRIPPER_LEAD * delta[k]
                        stepped[k] = min(sim_qpos[k] + lead, max(sim_qpos[k] - lead, stepped[k]))
                sim_target = bridge.clamp_sim(stepped)

            op.send_action(
                bridge.sim_to_real(sim_target),
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
        if win is not None:
            win.close()
        try:
            await op.disconnect()
        finally:
            op.close()


def cli() -> None:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(description="Run a trained RL policy on the real rig over LiveKit Portal.")
    parser.add_argument("--arch", choices=sorted(CHECKPOINTS),
                        help="run the named checkpoint from checkpoints/. Shorthand for "
                             "--checkpoint; the loaded file still declares its own architecture, "
                             "and a mismatch with the name is an error.")
    parser.add_argument("--checkpoint", default=os.environ.get("POLICY_CHECKPOINT"),
                        help="explicit path to a checkpoint .pt (env: POLICY_CHECKPOINT), for "
                             "files outside checkpoints/. Overrides --arch.")
    parser.add_argument("--claim", action=argparse.BooleanOptionalAction, default=True,
                        help="claim control on startup, so the debug keys work with no web UI "
                             "(default). --no-claim stays idle until the web UI claims it.")
    parser.add_argument("--start-paused", action=argparse.BooleanOptionalAction, default=True,
                        help="claim but hold the pose until you press p (default). "
                             "--no-start-paused MOVES THE ROBOT as soon as frames arrive.")
    parser.add_argument("--max-lag", type=float, default=3.0, metavar="DELTAS",
                        help="how far the target may lead the measured pose, in action steps "
                             "(default 3.0). An action is a velocity and keeps being applied every "
                             "tick, so this is what bounds it: the target cannot run away from a "
                             "slower arm. It doubles as the decision gate, since a new action is "
                             "computed once the arm is back inside the budget, which is why a tight "
                             "budget costs feedback rate: rl/calibrate measured every arm joint at "
                             "2-4x the speed it is commanded at, so lag this small is servo droop "
                             "rather than a joint failing to keep up, and waiting it out throttles "
                             "the loop for nothing. The gripper is exempt and carries its own lead "
                             "cap instead: a jaw closed on the object never arrives, and holding "
                             "its target back would bleed grip force. With --viz this is only the "
                             "starting value.")
    parser.add_argument("--rail-step", type=float, default=RAIL_STEP_MM, metavar="MM",
                        help=f"how far one full-command action moves the carriage, in mm (default "
                             f"{RAIL_STEP_MM:.1f}, the trained value). Also its top speed per "
                             "control period, and the unit rail lag is counted in. Changing it "
                             "runs the policy on a different action space than it trained on; the "
                             "point is to measure what the carriage actually does per step. With "
                             "--viz this is only the starting value.")
    parser.add_argument("--viz", action="store_true",
                        help="open a window (needs `uv sync --group viz`) with the rectified views "
                             "the encoder is fed, a bar per joint for the last action and its lag, "
                             "the debug keys as both keys and buttons, and live sliders for the lag "
                             "gates. The window takes the key presses while it has focus; the "
                             "terminal only hears them while the terminal does.")
    args = parser.parse_args()
    # An explicit path wins over the shorthand, which wins over the env var.
    if args.checkpoint:
        os.environ["POLICY_CHECKPOINT"] = args.checkpoint
        args.arch = None
    elif args.arch:
        path = CHECKPOINT_DIR / CHECKPOINTS[args.arch][0]
        if not path.exists():
            raise SystemExit(f"--arch {args.arch} expects {path}, which does not exist. "
                             "Pull it from the training box or pass --checkpoint.")
        os.environ["POLICY_CHECKPOINT"] = str(path)
    elif not os.environ.get("POLICY_CHECKPOINT"):
        raise SystemExit("pass --arch {" + ",".join(sorted(CHECKPOINTS))
                         + "}, or --checkpoint <path>, or set POLICY_CHECKPOINT.")
    asyncio.run(main(claim=args.claim, start_paused=args.start_paused,
                     max_lag=args.max_lag, rail_step=args.rail_step,
                     viz=args.viz, arch=args.arch))


if __name__ == "__main__":
    cli()
