"""Shared sim + real helpers for the figure scripts.

Runs the ManiSkill env on this Mac through MoltenVK and SAPIEN's cpu render backend, so every sim
frame in the post is formed by the same code training used rather than by a lookalike. Import this
before anything that touches sapien: it sets VK_ICD_FILENAMES at import time.

Run the scripts through rl/calibrate's project, which is the one environment where the simulator
and the robot feed coexist:

    uv run --project ../rl/calibrate python fig2_calibration.py

The task constants, camera FOVs, object colours and the deploy camera mappings are all imported
from the projects that own them. Nothing here re-declares a number that lives somewhere else.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import sys

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parent
RAW = _HERE / "raw"
OUT = _HERE / "out"

_MOLTENVK_ICD = pathlib.Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")
if platform.system() == "Darwin" and not os.environ.get("VK_ICD_FILENAMES"):
    if not _MOLTENVK_ICD.exists():
        raise SystemExit("sim rendering on macOS needs MoltenVK: brew install molten-vk")
    os.environ["VK_ICD_FILENAMES"] = str(_MOLTENVK_ICD)

# rl/deploy is imported as a path, not a package, exactly as rl/calibrate does it.
sys.path.insert(0, str(REPO / "rl" / "deploy"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.camera_mapping import CAMERA_STACK, load_mappings, rectify  # noqa: E402

# Real camera track -> the sim camera it is calibrated against, same table rl/calibrate uses.
REAL_TO_SIM_CAM = {"arm_camera": "wrist_camera", "overhead_camera": "overhead_camera"}
# Display order everywhere in the post: wrist first, matching the channel order of the stack the
# encoder is fed (CAMERA_STACK's order).
CAMERA_ORDER = tuple(track for track, _ in CAMERA_STACK)
CAMERA_TITLE = {"arm_camera": "wrist camera", "overhead_camera": "overhead camera"}

# The resolution the deployed dino_patch checkpoint's cameras rendered at. 168 px is 12 DINOv2
# patches a side, so a sim frame here is exactly the token grid the policy reads.
RES = 168


def config():
    """The training config module, with the object colours the deployed policy trained under.

    Every v4/v5 run passed --object_colors distinct, so the cube is blue and the bin yellow. This
    must be set BEFORE the env is built, the same ordering sac/loop.py observes, because
    pick_place reads these when it builds the scene.
    """
    from soframe_rl_maniskill import config as cfg
    cfg.COLOR_CUBE = cfg.BLUE
    cfg.COLOR_BIN = cfg.YELLOW
    return cfg


def build_env(*, dr=False, seed=1, res=RES, render_size=512, segmentation=False, state=False):
    """The pick-place env on the cpu backend. Returns (env, unwrapped)."""
    config()   # colours first

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import soframe_rl_maniskill.envs  # noqa: F401
    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    obs_mode = "rgb"
    if segmentation:
        obs_mode += "+segmentation"
    if state:
        obs_mode += "+state"

    env = gym.make(
        "SOFramePickPlaceBin-v1",
        num_envs=1,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        sim_backend="cpu",
        domain_randomization=dr,
        sensor_configs=dict(width=res, height=res),
        human_render_camera_configs=dict(width=render_size, height=render_size),
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=state)
    env.reset(seed=seed)
    return env, env.unwrapped


def set_qpos(unwrapped, sim_qpos: dict):
    """Drive the sim robot to a joint pose given by NAME, and flush the camera mounts.

    A SAPIEN-mounted camera renders from the transforms cached at the previous capture, so the
    first observation after a set_qpos still shows the old pose. The discarded capture flushes the
    mount; calling update_render twice does not. Same dance as rl/calibrate's SimMirror.
    """
    import torch

    idx = {n: i for i, n in enumerate(unwrapped.agent.joint_names)}
    qpos = unwrapped.agent.robot.get_qpos().clone()
    for name, value in sim_qpos.items():
        if name in idx:
            qpos[0, idx[name]] = float(value)
    unwrapped.agent.robot.set_qpos(qpos)
    unwrapped.scene.update_render()
    unwrapped.get_obs()
    unwrapped.scene.update_render()
    return torch


def sim_views(unwrapped) -> dict:
    """{sim camera name: HxWx3 uint8 RGB} from the current sim state."""
    obs = unwrapped.get_obs()
    return {
        cam: obs["sensor_data"][cam]["rgb"][0].cpu().numpy().astype(np.uint8)
        for cam in set(REAL_TO_SIM_CAM.values())
    }


def park_objects(unwrapped, far=5.0):
    """Move the cube and bin out of frame, leaving lighting and the rig honest.

    For the calibration figure: what is being compared there is the RIG's geometry, and the sim's
    objects spawn at random poses that have nothing to do with where the real ones happen to sit.
    """
    import torch
    from mani_skill.utils.structs.pose import Pose

    unwrapped.item.set_pose(Pose.create_from_pq(torch.tensor([[far, far, 0.5]])))
    unwrapped.bin.set_pose(Pose.create_from_pq(torch.tensor([[far + 1, far + 1, 0.5]])))


def place_objects(unwrapped, cube_xy, bin_xy, cube_yaw=0.0, bin_yaw=0.0):
    """Put the cube and bin at chosen workspace coordinates (metres, robot frame)."""
    import torch
    from mani_skill.utils.structs.pose import Pose

    cfg = config()

    def yaw_q(a):
        return torch.tensor([[np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)]], dtype=torch.float32)

    unwrapped.item.set_pose(Pose.create_from_pq(
        torch.tensor([[cube_xy[0], cube_xy[1], cfg.WORK_SURFACE_Z + cfg.CUBE_HALF]]), yaw_q(cube_yaw)
    ))
    unwrapped.bin.set_pose(Pose.create_from_pq(
        torch.tensor([[bin_xy[0], bin_xy[1], cfg.WORK_SURFACE_Z]]), yaw_q(bin_yaw)
    ))
    unwrapped.scene.update_render()
    unwrapped.get_obs()
    unwrapped.scene.update_render()


# ---------------------------------------------------------------------------------------------
# The real side
# ---------------------------------------------------------------------------------------------
def reference() -> dict:
    """The matched capture written by pull_reference.py: frames plus the arm pose that made them."""
    path = RAW / "reference.json"
    if not path.exists():
        raise SystemExit(
            f"no reference capture at {path}. With the robot online, run:\n"
            "    uv run --project ../rl/calibrate python pull_reference.py"
        )
    return json.loads(path.read_text())


def real_frame(camera: str) -> np.ndarray:
    """The raw 640x480 RGB frame for a camera track, as captured off the rig."""
    path = RAW / f"real_{camera}.png"
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"missing {path}; run pull_reference.py with the robot online")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def mappings():
    """The fitted camera mappings the deploy loop replays, straight out of rl/deploy."""
    return load_mappings()


def rectify_real(camera: str, out_size=RES, maps=None) -> np.ndarray:
    """Raw real frame -> the sim-matched view the encoder is actually fed."""
    maps = maps if maps is not None else mappings()
    return rectify(real_frame(camera), maps.get(camera), out_size)


# ---------------------------------------------------------------------------------------------
# Shared image ops
# ---------------------------------------------------------------------------------------------
def squint(img, k=4):
    """The training squint: area-average downsample by a factor of k (128 -> 32)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // k, h // k), interpolation=cv2.INTER_AREA)


def blow_up(img, size):
    """Nearest-neighbour upscale, so a low-res frame shows its pixels instead of a blur."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def camera_projection(unwrapped, name="overhead_camera"):
    """(K, world->cam 3x4) for a sensor camera, in OpenCV convention."""
    p = unwrapped.scene.sensors[name].get_params()
    return np.array(p["intrinsic_cv"][0].cpu()), np.array(p["extrinsic_cv"][0].cpu())


def project(points_world, K, world2cam):
    """Project Nx3 world points to Nx2 pixel coordinates."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    hom = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    cam = (world2cam @ hom.T).T
    z = np.clip(cam[:, 2:3], 1e-6, None)
    return (K @ (cam / z).T).T[:, :2]


def unproject_to_plane(uv, K, world2cam, z_plane=0.0):
    """Pixel coordinates -> where their ray meets a horizontal plane, in world coordinates.

    The inverse of `project` for anything known to be lying on the work surface. Because the
    rectified real view is fitted to this camera, a pixel in a real frame can be read as a world
    position on the surface, which is how a real object's position is recovered without measuring
    it by hand.
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    R, t = world2cam[:, :3], world2cam[:, 3]
    centre = -R.T @ t
    rays = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    dirs = (R.T @ np.linalg.inv(K) @ rays.T).T
    s = (z_plane - centre[2]) / dirs[:, 2]
    return centre + s[:, None] * dirs


def colour_centroid(rgb, want, margin=25):
    """Centroid pixel of the most `want`-coloured region ('blue' or 'yellow'), or None."""
    a = rgb.astype(int)
    if want == "blue":
        mask = (a[..., 2] > a[..., 0] + margin) & (a[..., 2] > a[..., 1] + margin)
    else:
        mask = (a[..., 0] > a[..., 2] + margin) & (a[..., 1] > a[..., 2] + margin)
    if mask.sum() < 3:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def save(fig, name):
    """Write a figure into out/ and say where it went."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    print(f"[viz] wrote {path}")
    return path
