"""Base environment with domain randomization and dual-camera vision, adapted from
[Squint](https://github.com/aalmuzairee/squint)'s ``envs/base_random_env.py``. Handles
gripper/lighting randomization, greenscreen overlay, and wrist + overhead cameras that
track the URDF's calibrated mounts (via forward kinematics) with small jitter."""

import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import cv2
import dacite
import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from mani_skill.utils.structs import Pose
from mani_skill.utils.visualization.misc import tile_images

from .. import config


@dataclass
class RandomizationConfig:
    initial_qpos_noise_scale: float = 0.02
    """Noise scale for initial robot joint positions."""
    apply_overlay: bool = False
    """Composite `rgb_overlay_path` in behind the robot and the task objects, per pixel, using a
    segmentation render. OFF: it costs an extra render target and readback per camera per step, and
    config.SENSOR_FAR already culls the only background geometry (the ground plane), leaving the
    black clear colour the default overlay image painted anyway.

    Turn on only to composite a REAL background photo, and then also pass an obs_mode including
    segmentation (`--obs_mode rgb+segmentation`), which is what feeds the mask."""
    rgb_overlay_path: Optional[str] = os.path.join(os.path.dirname(__file__), "black_overlay.png")
    """Path to background image. If None and apply_overlay=True, uses black background."""
    visual_fidelity: str = "raster"
    """Rendering fidelity: "flat" (fast shadowless, for cheap ablations), "raster" (default;
    PBR materials + softbox-like lighting, GPU-parallel-sensor compatible, trainable), or
    "raytraced" (PBR + overhead area light; rt/rt-fast shaders only, for one-off renders/
    single-env evals, never training)."""

    gripper_stiffness_range: Sequence[float] = (500, 2000)
    """Per-episode gripper joint stiffness range."""
    gripper_damping_range: Sequence[float] = (50, 200)
    """Per-episode gripper joint damping range."""
    binary_gripper: bool = False
    """Threshold the gripper action to its sign each step (fully open/closed only); arm/rail stay
    continuous, and the action space stays continuous. OFF: the reward pays a ramp in gripper
    openness (see ``PickPlaceBin.compute_dense_reward``), which needs intermediate jaw positions to
    have a gradient at all."""
    arm_stiffness_range: Sequence[float] = (600, 1400)
    """Per-episode stiffness range for arm + rail joints (nominal 1e3, +-40% for sim2real).
    Set (1000, 1000) to disable."""
    arm_damping_range: Sequence[float] = (60, 140)
    """Per-episode damping range for arm + rail joints (nominal 1e2)."""
    randomize_lighting: bool = True
    """Whether to randomize ambient lighting."""

    wrist_camera_fov: float = config.WRIST_CAMERA_FOV
    """Base vertical FOV of the wrist camera (default in config.py)."""
    overhead_camera_fov: float = config.OVERHEAD_CAMERA_FOV
    """Base vertical FOV of the overhead camera, calibrated against the real rig (config.py)."""

    # These FOVs define the view the policy trains on. Deploy rectifies each real camera to match;
    # the only handoff is the reference renders from examples/dump_reference_views.py.

    # Constant camera pose corrections in the camera link's local frame (+X = view direction),
    # applied before jitter (the printed holder's CAD pose vs the actual lens pose).
    wrist_camera_pos_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant position offset (meters, camera-local: +X forward)."""
    wrist_camera_rot_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant rotation offset (roll, pitch, yaw radians, camera-local)."""
    overhead_camera_pos_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant position offset (meters, camera-local: +X forward)."""
    overhead_camera_rot_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant rotation offset (roll, pitch, yaw radians, camera-local)."""

    # Per-env camera jitter, drawn ONCE PER SCENE BUILD (reconfigure) and baked into each camera's
    # local pose on its mount link, so it costs nothing per step. Pass --reconfiguration_freq to
    # resample during a run.
    wrist_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    """Max position noise (x, y, z) meters, on top of the calibrated mount pose."""
    wrist_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) radians, on top of the calibrated mount pose."""
    wrist_camera_fov_noise: float = np.deg2rad(1)
    """Noise scale for camera FOV."""

    overhead_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    """Max position noise (x, y, z) meters, on top of the calibrated mount pose."""
    overhead_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) radians, on top of the calibrated mount pose."""
    overhead_camera_fov_noise: float = np.deg2rad(1)
    """Noise scale for camera FOV."""

    @classmethod
    def resolve(cls, cfg: Union["RandomizationConfig", dict]) -> "RandomizationConfig":
        """Merge a dict of overrides into this class's defaults (instances pass through).

        The parameter is `cfg`, not `config`: it must not shadow the module-level
        ``from .. import config``.
        """
        if isinstance(cfg, cls):
            return cfg
        merged = asdict(cls())
        common.dict_merge(merged, cfg if isinstance(cfg, dict) else asdict(cfg))
        return dacite.from_dict(data_class=cls, data=merged, config=dacite.Config(strict=True))


class BaseRandomEnv(BaseEnv):
    """Base environment with domain randomization and overlay support; subclasses add the
    task-specific scene and inherit cameras from ``DualCameraEnv``."""

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        domain_randomization: bool = True,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization
        self.domain_randomization_config = RandomizationConfig.resolve(domain_randomization_config)

        self._objects_to_remove_from_greenscreen: list[Union[Actor, Link]] = []
        self._segmentation_ids_to_keep: torch.Tensor = None
        self._rgb_overlay_image: torch.Tensor = None
        self._overlay_initialized = False

        self._rgb_overlay_np = None
        if (
            self.domain_randomization_config.apply_overlay
            and self.domain_randomization_config.rgb_overlay_path is not None
        ):
            path = self.domain_randomization_config.rgb_overlay_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"rgb_overlay_path {path} not found.")
            self._rgb_overlay_np = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=int(config.SIM_HZ),
            control_freq=int(config.CONTROL_HZ),
            # Solver effort is a cost knob, not a physical constant; see config.py.
            scene_config=SceneConfig(
                solver_position_iterations=config.SOLVER_POSITION_ITERATIONS,
                solver_velocity_iterations=config.SOLVER_VELOCITY_ITERATIONS,
                enable_friction_every_iteration=config.FRICTION_EVERY_ITERATION,
            ),
        )

    @property
    def _default_human_render_camera_configs(self):
        # Videos only; the policy never sees it. Sized from config because its render target is
        # allocated per sub-scene (~1 GB of VRAM at 512x512 and 1024 envs).
        pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        size = config.HUMAN_RENDER_RESOLUTION
        return CameraConfig("render_camera", pose, size, size, 52 * np.pi / 180, 0.01, 100)

    @property
    def apply_greenscreen(self) -> bool:
        return self.domain_randomization_config.apply_overlay

    def _load_scene(self, options: dict):
        self._objects_to_remove_from_greenscreen = []

    def _load_lighting(self, options: dict):
        fidelity = self.domain_randomization_config.visual_fidelity
        if fidelity == "raytraced":
            # One overhead softbox (area light), matching the real rig. SAPIEN lights emit
            # along +x, so the pose rotates +x down; oversized so shadows wash out.
            self.scene.set_ambient_light([0.45, 0.45, 0.48])
            self.scene.add_area_light_for_ray_tracing(
                sapien.Pose(p=[-0.225, -0.5, 1.0], q=[0.7071068, 0, 0.7071068, 0]),
                [2.4, 2.4, 2.2], 0.5, 0.4,
            )
            return

        if self.domain_randomization and self.domain_randomization_config.randomize_lighting:
            ambient_colors = self._batched_episode_rng.uniform(0.2, 0.5, size=(3,))
        else:
            ambient_colors = np.full((self.num_envs, 3), 0.3)
        if fidelity == "raster":
            # The lightbox is a diffuse surrounding source (ambient in a rasterizer); the
            # boost keeps the arm's vertical surfaces lit (a straight-down light leaves them dark).
            ambient_colors = ambient_colors + 0.25
        for i, scene in enumerate(self.scene.sub_scenes):
            scene.render_system.ambient_light = ambient_colors[i]

        if fidelity == "raster":
            # Rasterizer stand-in for the softbox: the real lightbox is essentially shadow-free, so
            # most light is boosted ambient + shadowless vertical fill and the single
            # shadow-casting key is faint (~0.75:1 shadowed:lit).
            self.scene.add_directional_light(
                [0.15, 0.1, -1], [0.4, 0.39, 0.38],
                shadow=config.RASTER_SHADOWS, shadow_scale=2.0,
                shadow_map_size=config.SHADOW_MAP_SIZE,
            )
            self.scene.add_directional_light([0, 0, -1], [0.4, 0.39, 0.38])
            self.scene.add_directional_light([-1, -0.3, -0.6], [0.25, 0.25, 0.28])
            return

        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _force_limit(self, joint, idx: int) -> float:
        """The joint's configured drive force limit, snapshotted on first use.

        ``set_drive_properties`` takes stiffness, damping AND force limit together, so the gain
        randomizers below must pass a force limit back in. It has to be this snapshot: passing a
        constant instead silently overwrites the agent's real per-joint limits (the six STS3215
        joints are capped at their ~3 N.m stall torque) on every episode reset.
        """
        if not hasattr(self, "_force_limit_snapshot"):
            self._force_limit_snapshot = {}
        key = (joint.name, idx)
        if key not in self._force_limit_snapshot:
            self._force_limit_snapshot[key] = float(joint._objs[idx].get_force_limit())
        return self._force_limit_snapshot[key]

    def _randomize_gripper_speed(self, env_idx: torch.Tensor):
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range

        if not hasattr(self, "_gripper_stiffness"):
            default_stiffness = (stiff_lo + stiff_hi) / 2
            default_damping = (damp_lo + damp_hi) / 2
            self._gripper_stiffness = torch.full((self.num_envs,), default_stiffness, device=self.device)
            self._gripper_damping = torch.full((self.num_envs,), default_damping, device=self.device)

        if not self.domain_randomization:
            return
        if stiff_lo == stiff_hi and damp_lo == damp_hi:
            return

        stiffnesses = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
        dampings = self._batched_episode_rng[env_idx].uniform(damp_lo, damp_hi)
        gripper_joint = self.agent.robot.joints_map["gripper"]

        for i, idx in enumerate(env_idx.tolist()):
            gripper_joint._objs[idx].set_drive_properties(
                stiffnesses[i], dampings[i], force_limit=self._force_limit(gripper_joint, idx)
            )
            self._gripper_stiffness[idx] = stiffnesses[i]
            self._gripper_damping[idx] = dampings[i]

    def _randomize_arm_gains(self, env_idx: torch.Tensor):
        """Per-episode stiffness/damping on the arm + rail joints (gripper randomized
        separately), so the policy doesn't overfit sim's fixed PD."""
        stiff_lo, stiff_hi = self.domain_randomization_config.arm_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.arm_damping_range
        if not self.domain_randomization:
            return
        if stiff_lo == stiff_hi and damp_lo == damp_hi:
            return
        arm_joints = [self.agent.robot.joints_map[n]
                      for n in self.agent.joint_names if n != "gripper"]
        stiffnesses = self._batched_episode_rng[env_idx].uniform(stiff_lo, stiff_hi)
        dampings = self._batched_episode_rng[env_idx].uniform(damp_lo, damp_hi)
        for i, idx in enumerate(env_idx.tolist()):
            for joint in arm_joints:
                joint._objs[idx].set_drive_properties(
                    stiffnesses[i], dampings[i], force_limit=self._force_limit(joint, idx)
                )

    def get_gripper_params(self) -> dict[str, torch.Tensor]:
        stiff_lo, stiff_hi = self.domain_randomization_config.gripper_stiffness_range
        damp_lo, damp_hi = self.domain_randomization_config.gripper_damping_range

        stiff_range = stiff_hi - stiff_lo if stiff_hi != stiff_lo else 1.0
        damp_range = damp_hi - damp_lo if damp_hi != damp_lo else 1.0

        return {
            "gripper_stiffness": (self._gripper_stiffness - stiff_lo) / stiff_range,
            "gripper_damping": (self._gripper_damping - damp_lo) / damp_range,
        }

    def remove_object_from_greenscreen(self, obj: Union[Articulation, Actor, Link]):
        if isinstance(obj, Articulation):
            for link in obj.get_links():
                self._objects_to_remove_from_greenscreen.append(link)
        elif isinstance(obj, (Actor, Link)):
            self._objects_to_remove_from_greenscreen.append(obj)

    def _after_reconfigure(self, options: dict):
        super()._after_reconfigure(options)

        if not self.domain_randomization_config.apply_overlay:
            self._objects_to_remove_from_greenscreen = []
            return

        per_scene_ids = []
        for obj in self._objects_to_remove_from_greenscreen:
            per_scene_ids.append(obj.per_scene_id)

        if per_scene_ids:
            self._segmentation_ids_to_keep = torch.unique(torch.concatenate(per_scene_ids))
        else:
            self._segmentation_ids_to_keep = torch.tensor([], dtype=torch.int64)

        if not self._overlay_initialized:
            # The first non-render sensor camera sets the overlay resolution.
            for name, sensor in self._sensor_configs.items():
                if isinstance(sensor, CameraConfig) and name != "render_camera":
                    if self._rgb_overlay_np is not None:
                        resized = cv2.resize(self._rgb_overlay_np, (sensor.width, sensor.height))
                        self._rgb_overlay_image = common.to_tensor(resized, device=self.device)
                    else:
                        self._rgb_overlay_image = torch.zeros(
                            (sensor.height, sensor.width, 3), dtype=torch.uint8, device=self.device
                        )
                    break
            if self._rgb_overlay_image is None and self._rgb_overlay_np is not None:
                self._rgb_overlay_image = common.to_tensor(self._rgb_overlay_np, device=self.device)

        self._overlay_initialized = True
        self._objects_to_remove_from_greenscreen = []

    def _green_screen_rgb(self, rgb: torch.Tensor, segmentation: torch.Tensor, overlay: torch.Tensor) -> torch.Tensor:
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg, dtype=torch.bool)

        if self._segmentation_ids_to_keep.device != actor_seg.device:
            self._segmentation_ids_to_keep = self._segmentation_ids_to_keep.to(actor_seg.device)

        mask[torch.isin(actor_seg, self._segmentation_ids_to_keep)] = False
        mask = mask[..., None]

        original_dtype = rgb.dtype
        rgb = rgb.float()
        overlay = overlay.float()
        result = rgb * (~mask) + overlay * mask

        return result.to(original_dtype)

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        obs = super()._get_obs_sensor_data(apply_texture_transforms)

        if not self.domain_randomization_config.apply_overlay:
            return obs

        if not (self.obs_mode_struct.visual.rgb and self.obs_mode_struct.visual.segmentation):
            return obs

        if self._rgb_overlay_image is None:
            return obs

        for camera_name, camera_obs in obs.items():
            if not isinstance(camera_obs, dict) or "rgb" not in camera_obs:
                continue
            if "segmentation" not in camera_obs:
                continue
            if camera_name == "render_camera":
                continue

            overlay = self._rgb_overlay_image
            if overlay.device != camera_obs["rgb"].device:
                self._rgb_overlay_image = overlay.to(camera_obs["rgb"].device)
                overlay = self._rgb_overlay_image

            obs[camera_name]["rgb"] = self._green_screen_rgb(
                camera_obs["rgb"],
                camera_obs["segmentation"],
                overlay,
            )

        return obs

    def render_all(self):
        """Renders all human render cameras and sensors together, excluding segmentation."""
        images = []
        for obj in self._hidden_objects:
            obj.show_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=True)
        render_images = self.scene.get_human_render_camera_images()
        sensor_images = self.get_sensor_images()

        for image in sensor_images.values():
            for key, img in image.items():
                if "segmentation" not in key:
                    images.append(img)
        for image in render_images.values():
            images.append(image)

        return tile_images(images)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._randomize_gripper_speed(env_idx)
        self._randomize_arm_gains(env_idx)

    def step(self, action):
        """Near-binary gripper (threshold to sign); see RandomizationConfig.binary_gripper."""
        cfg = self.domain_randomization_config
        if cfg.binary_gripper:
            gi = self.agent.joint_names.index("gripper")
            act = common.to_tensor(action, device=self.device).clone().float()
            act[..., gi] = torch.where(act[..., gi] > 0, 1.0, -1.0)
            action = act
        return super().step(action)


def _jitter_pose(rand_vals: torch.Tensor, pos_noise, rot_noise) -> Pose:
    """Per-env (position, roll/pitch/yaw) offset pose from uniforms in [-1, 1], shape (n, 6)."""
    dx = pos_noise[0] * rand_vals[:, 0]
    dy = pos_noise[1] * rand_vals[:, 1]
    dz = pos_noise[2] * rand_vals[:, 2]
    d_roll = rot_noise[0] * rand_vals[:, 3]
    d_pitch = rot_noise[1] * rand_vals[:, 4]
    d_yaw = rot_noise[2] * rand_vals[:, 5]

    # Euler (roll, pitch, yaw) -> quaternion, batched.
    cj, sj = torch.cos(d_pitch / 2), torch.sin(d_pitch / 2)
    ck, sk = torch.cos(d_yaw / 2), torch.sin(d_yaw / 2)
    ci, si = torch.cos(d_roll / 2), torch.sin(d_roll / 2)

    q_py_w, q_py_x, q_py_y, q_py_z = cj * ck, -sj * sk, sj * ck, cj * sk

    qw = q_py_w * ci - q_py_x * si
    qx = q_py_w * si + q_py_x * ci
    qy = q_py_y * ci + q_py_z * si
    qz = q_py_z * ci - q_py_y * si

    p = torch.stack([dx, dy, dz], dim=-1)
    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return Pose.create_from_pq(p=p, q=q)


class DualCameraEnv(BaseRandomEnv):
    """Wrist camera (follows the gripper) + static overhead camera.

    Both are SAPIEN-mounted directly on the URDF's calibrated camera links, so their world poses
    come from the link inside the render update and there is nothing to maintain per step. The
    calibration correction and the per-env jitter are composed into each camera's LOCAL pose on that
    link, once, when the scene is built.

    `FlattenRGBDObservationWrapper` concatenates both cameras' RGB along the channel axis (`rgb`
    becomes H x W x 6)."""

    @property
    def _default_sensor_configs(self):
        dr = self.domain_randomization_config

        def fov_noise(scale):
            if self.domain_randomization and scale > 0:
                return scale * (2 * self._batched_episode_rng.rand() - 1)
            return 0

        return [
            CameraConfig(
                "wrist_camera",
                pose=self._camera_local_pose(
                    dr.wrist_camera_pos_offset, dr.wrist_camera_rot_offset,
                    dr.wrist_camera_pos_noise, dr.wrist_camera_rot_noise,
                ),
                width=config.SENSOR_RESOLUTION,
                height=config.SENSOR_RESOLUTION,
                fov=dr.wrist_camera_fov + fov_noise(dr.wrist_camera_fov_noise),
                near=config.SENSOR_NEAR,
                far=config.SENSOR_FAR,
                mount=self.agent.wrist_camera_link,
            ),
            CameraConfig(
                "overhead_camera",
                pose=self._camera_local_pose(
                    dr.overhead_camera_pos_offset, dr.overhead_camera_rot_offset,
                    dr.overhead_camera_pos_noise, dr.overhead_camera_rot_noise,
                ),
                width=config.SENSOR_RESOLUTION,
                height=config.SENSOR_RESOLUTION,
                fov=dr.overhead_camera_fov + fov_noise(dr.overhead_camera_fov_noise),
                near=config.SENSOR_NEAR,
                far=config.SENSOR_FAR,
                mount=self.agent.overhead_camera_link,
            ),
        ]

    @staticmethod
    def _offset_pose(pos_offset, rot_offset) -> sapien.Pose:
        """Constant local-frame correction pose from (xyz, rpy) config values."""
        from transforms3d.euler import euler2quat

        return sapien.Pose(p=list(pos_offset), q=euler2quat(*rot_offset))

    def _camera_local_pose(self, pos_offset, rot_offset, pos_noise, rot_noise) -> Pose:
        """A camera's pose in its mount link's frame: constant calibration correction, then per-env
        jitter. Batched over envs, drawn from the episode RNG so a seed reproduces it. Returns a
        plain `sapien.Pose` when there is no jitter, keeping that path a single shared pose."""
        offset = self._offset_pose(pos_offset, rot_offset)

        jitters = any(n > 0 for n in (*pos_noise, *rot_noise))
        if not (self.domain_randomization and jitters):
            return offset

        # (num_envs, 6) uniforms in [-1, 1]: dx dy dz, then d_roll d_pitch d_yaw.
        rand_vals = common.to_tensor(
            self._batched_episode_rng.uniform(low=-1.0, high=1.0, size=(6,)),
            device=self.device,
        ).float()
        jitter = _jitter_pose(rand_vals, pos_noise, rot_noise)

        # Both operands explicitly (num_envs, ...), so the multiply needs no broadcasting.
        base = Pose.create_from_pq(
            p=common.to_tensor(offset.p, device=self.device).float().repeat(self.num_envs, 1),
            q=common.to_tensor(offset.q, device=self.device).float().repeat(self.num_envs, 1),
        )
        return base * jitter
