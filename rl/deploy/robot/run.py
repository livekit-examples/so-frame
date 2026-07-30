"""Robot runtime for the SO-101-on-frame rig: the leslider (arm + linear rail).

Joins the LiveKit room as the Portal `robot`, publishes 7-DOF state + RAW camera
frames every tick, and applies the latest action (absolute position targets).

The rig is one integrated 7-DOF follower (`SO101SliderPosFollower`) over a single
Feetech bus: SO-101 arm plus the rail as a real STS3215 motor. Wire units are arm
degrees (use_degrees=True), gripper + rail 0..100. The follower names the rail
`slider.pos`, our wire calls it `dof_slider.pos`, same range, so the boundary is a
key rename.

Wire contract is ../portal.yaml (7 joints, 2 cameras) in REAL units. The policy
operator (policy/run.py) owns the sim<->real bridge and camera rectification, so the
robot stays generic and the web UI sees the true wide-FOV stream.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import platform
import sys
import time

from livekit.portal import Action, Robot, RobotConfig, RpcInvocationData

from lerobot.cameras import Cv2Backends, Cv2Rotation
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot_robot_so101_slider_pos import (
    SO101SliderPosFollower,
    SO101SliderPosFollowerConfig,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils import bridge as bridge_park  # noqa: E402  (PARK_REAL only; wire units, no sim maths)
from utils.common import env, load_env, mint_token, pace  # noqa: E402

IDENTITY = "robot"
CONFIG_PATH = pathlib.Path(__file__).resolve().parents[1] / "portal.yaml"

# Camera track name -> (env var for device, default). Must match portal.yaml.
CAMERAS = {
    "arm_camera": ("SO101_CAM_ARM", "/dev/video0"),
    "overhead_camera": ("SO101_CAM_OVERHEAD", "/dev/video2"),
}

# Follower rail key -> wire rail key. Both 0..100, so translation is a key rename.
FOLLOWER_SLIDER_KEY = "slider.pos"
WIRE_SLIDER_KEY = "dof_slider.pos"

# The pose `reset_to_zero_position` drives to: the shared park pose, in wire units already.
# Defined in utils.bridge so the policy's park key and this RPC cannot drift apart. Override any
# joint via its RESET_POSE_* env var.
REST_POSE_DEFAULTS: dict[str, float] = dict(bridge_park.PARK_REAL)
REST_POSE_ENV_KEYS: dict[str, str] = {
    WIRE_SLIDER_KEY:     "RESET_POSE_SLIDER",
    "shoulder_pan.pos":  "RESET_POSE_SHOULDER_PAN",
    "shoulder_lift.pos": "RESET_POSE_SHOULDER_LIFT",
    "elbow_flex.pos":    "RESET_POSE_ELBOW_FLEX",
    "wrist_flex.pos":    "RESET_POSE_WRIST_FLEX",
    "wrist_roll.pos":    "RESET_POSE_WRIST_ROLL",
    "gripper.pos":       "RESET_POSE_GRIPPER",
}


def build_rest_pose() -> dict[str, float]:
    pose: dict[str, float] = {}
    for key, default in REST_POSE_DEFAULTS.items():
        raw = env(REST_POSE_ENV_KEYS[key], "")
        pose[key] = float(raw) if raw else default
    return pose


def build_follower(camera_fps: int) -> SO101SliderPosFollower:
    """The integrated 7-DOF leslider follower (arm + gripper + rail) + 2 cameras.

    camera_fps is the CAMERA's hardware capture rate (SO101_CAM_FPS, default 30), NOT
    the control rate: lerobot validates the camera delivers it and the USB cameras only
    do 30. The loop ticks at the slower PORTAL_FPS and samples the latest frame. Linux
    forces the V4L2 backend, since OpenCV's auto-picker no-ops the fourcc/resolution set."""
    width, height = int(env("SO101_CAM_WIDTH", "640")), int(env("SO101_CAM_HEIGHT", "480"))
    backend = Cv2Backends.V4L2 if platform.system() == "Linux" else Cv2Backends.ANY
    cameras = {}
    for name, (var, default) in CAMERAS.items():
        cam = env(var, default)  # macOS: integer index; Linux: /dev/videoN
        cameras[name] = OpenCVCameraConfig(
            index_or_path=int(cam) if cam.isdigit() else cam,
            fps=camera_fps, width=width, height=height,
            rotation=Cv2Rotation.NO_ROTATION, fourcc="MJPG", backend=backend,
        )
    return SO101SliderPosFollower(SO101SliderPosFollowerConfig(
        id=env("LESLIDER_ID", "leslider"),
        port=env("LESLIDER_PORT", env("SO101_PORT", "/dev/ttyACM0")),
        cameras=cameras,
        use_degrees=True,   # arm joints in degrees, matching utils/bridge.py
        # 0 = rail runs to its goal at full speed; a positive raw-ticks/s cap slows it.
        slider_goal_speed=int(env("LESLIDER_SLIDER_GOAL_SPEED", "0")),
        read_current=False,
    ))


def split_obs(observation: dict) -> tuple[dict[str, float], dict[str, object]]:
    """Split the follower's flat observation into (wire state, camera frames).

    Motor scalars are the `.pos`/`.vel` keys, the rest are camera ndarrays. `slider.pos`
    is renamed to the wire's `dof_slider.pos`."""
    state: dict[str, float] = {}
    for key, value in observation.items():
        if key.endswith(".pos") or key.endswith(".vel"):
            state[WIRE_SLIDER_KEY if key == FOLLOWER_SLIDER_KEY else key] = float(value)
    frames = {c: observation[c] for c in CAMERAS
              if c in observation and observation[c] is not None}
    return state, frames


def wire_action_to_follower(action: dict[str, float]) -> dict[str, float]:
    """Wire action -> follower action: rename dof_slider.pos back to slider.pos."""
    return {(FOLLOWER_SLIDER_KEY if k == WIRE_SLIDER_KEY else k): float(v)
            for k, v in action.items()}


async def main() -> None:
    load_env(pathlib.Path(__file__).resolve().parent)
    url = env("LIVEKIT_URL", required=True)
    room = env("LIVEKIT_ROOM", "so-frame")
    fps = int(env("PORTAL_FPS", "10"))            # control/publish rate (policy trained at 10)
    camera_fps = int(env("SO101_CAM_FPS", "30"))  # camera HARDWARE rate (USB cams only do 30)

    follower = build_follower(camera_fps)
    robot = Robot(RobotConfig.from_yaml_file(CONFIG_PATH, room))
    latest_action: dict[str, float] = {}
    rest_pose = build_rest_pose()
    print("[robot] rest pose: " + ", ".join(f"{k}={v:+.2f}" for k, v in rest_pose.items()))

    def on_action(action: Action) -> None:
        nonlocal latest_action
        latest_action = {k: float(v) for k, v in action.values.items()}

    def on_operator_left(identity: str) -> None:
        nonlocal latest_action
        if identity == robot.active_operator():
            print(f"[robot] active operator '{identity}' left; releasing")
            latest_action = {}
            asyncio.create_task(robot.set_active_operator(None))

    async def release_control(data: RpcInvocationData) -> str:
        nonlocal latest_action
        latest_action = {}
        await robot.set_active_operator(None)
        print(f"[robot] release_control from '{data.caller_identity}' -> arm idle")
        return json.dumps({"ok": True})

    async def reset_to_zero_position(data: RpcInvocationData) -> str:
        """Drive the arm + rail to the parked rest pose and hold it."""
        nonlocal latest_action
        await robot.set_active_operator(None)
        latest_action = dict(rest_pose)
        print(f"[robot] reset_to_zero_position from '{data.caller_identity}' -> parking")
        return json.dumps({"ok": True})

    robot.on_action(on_action)
    robot.on_operator_left(on_operator_left)
    robot.on_active_operator_changed(lambda i: print(f"[robot] active operator now: {i}"))
    robot.register_rpc_method("release_control", release_control)
    robot.register_rpc_method("reset_to_zero_position", reset_to_zero_position)

    follower.connect()
    print(f"[robot] connecting to {url} as '{IDENTITY}' in room '{room}' ...")
    await robot.connect(url, mint_token(IDENTITY, room))
    print(f"[robot] connected; streaming {list(CAMERAS)} + 7-DOF state at {fps} fps; ctrl-c to stop")

    BUS_ERRORS = (ConnectionError, RuntimeError, DeviceNotConnectedError)

    try:
        async for _ in pace(fps):
            ts_us = int(time.time() * 1_000_000)
            try:
                obs = follower.get_observation()
            except BUS_ERRORS as exc:
                print(f"[robot] motor read failed: {exc}")
                continue

            state, frames = split_obs(obs)
            robot.send_state(state, timestamp_us=ts_us)

            for name in CAMERAS:
                frame = frames.get(name)
                if frame is not None:
                    robot.send_video_frame(name, frame, timestamp_us=ts_us)

            if latest_action:
                try:
                    follower.send_action(wire_action_to_follower(latest_action))
                except BUS_ERRORS as exc:
                    print(f"[robot] action send failed: {exc}")
    except KeyboardInterrupt:
        print("\n[robot] stopping ...")
    finally:
        try:
            await robot.disconnect()
        finally:
            robot.close()
            follower.disconnect()


def cli() -> None:
    """Console-script entry point (`uv run robot` / `python robot/run.py`)."""
    asyncio.run(main())


if __name__ == "__main__":
    cli()
