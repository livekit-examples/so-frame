"""Base environment with domain randomization and dual-camera vision.

Adapted from [Squint](https://github.com/aalmuzairee/squint)'s ``envs/base_random_env.py``.
Handles:
- Gripper stiffness/damping randomization
- Lighting randomization
- Robot color randomization
- Background overlay (greenscreen) compositing
- Wrist + overhead cameras that track the URDF's calibrated mounts, with small jitter

Camera base poses come from the URDF's calibrated ``frame_wrist_camera``/
``frame_overhead_camera`` links (see ``simulation/urdf/README.md``) via forward kinematics,
rather than Squint's hand-measured constant offset from ``gripper_link``.
"""

import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.structs import Pose
from mani_skill.utils.visualization.misc import tile_images


@dataclass
class RandomizationConfig:
    # === Static settings (not affected by domain_randomization flag) ===
    initial_qpos_noise_scale: float = 0.02
    """Noise scale for initial robot joint positions."""
    apply_overlay: bool = True
    """Whether to apply background overlay (greenscreen). If False, returns raw simulation images."""
    rgb_overlay_path: Optional[str] = os.path.join(os.path.dirname(__file__), "black_overlay.png")
    """Path to background image. If None and apply_overlay=True, uses black background."""
    realism_mode: bool = False
    """Favor visual fidelity over speed, for one-off visualization renders (see
    ``examples/render_realistic.py``): a shadow-casting 3-point lighting rig instead of the
    flat ambient + shadowless lights used for training, and slightly rougher (less plasticky)
    cube/bin materials. Off by default -- zero effect on training throughput or behavior.
    Combine with ``human_render_camera_configs=dict(shader_pack="rt"|"rt-fast")`` for
    ray-traced reflections/soft shadows/GI, which this flag alone does not add."""

    # === Common randomization settings (affected by domain_randomization flag) ===
    gripper_stiffness_range: Sequence[float] = (500, 2000)
    """Range for gripper joint stiffness randomization (per-episode)."""
    gripper_damping_range: Sequence[float] = (50, 200)
    """Range for gripper joint damping randomization (per-episode)."""
    robot_color: Optional[Union[str, Sequence[float]]] = None
    """Robot color in RGB (0-1). Set to "random" for per-episode randomization."""
    randomize_lighting: bool = True
    """Whether to randomize ambient lighting."""

    # === Wrist camera jitter (on top of the URDF-calibrated base pose) ===
    wrist_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    """Max position noise (x, y, z), meters, on top of the calibrated mount pose."""
    wrist_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) in radians, on top of the calibrated mount pose."""
    wrist_camera_fov_noise: float = np.deg2rad(1)
    """Noise scale for camera FOV. Base FOV comes from the URDF camera (`fovy` in the MJCF twin)."""

    # === Overhead camera jitter (on top of the URDF-calibrated base pose) ===
    overhead_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    """Max position noise (x, y, z), meters, on top of the calibrated mount pose."""
    overhead_camera_rot_noise: Sequence[float] = (np.deg2rad(1), np.deg2rad(1), np.deg2rad(1))
    """Max rotation noise (roll, pitch, yaw) in radians, on top of the calibrated mount pose."""
    overhead_camera_fov_noise: float = np.deg2rad(1)
    """Noise scale for camera FOV. Base FOV comes from the URDF camera (`fovy` in the MJCF twin)."""

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


class BaseRandomEnv(BaseEnv):
    """Base environment with domain randomization and overlay support.

    Subclasses (e.g. ``pick_place.PickPlaceBin``) add the task-specific scene and
    call into ``WristCameraEnv`` for the camera.
    """

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        domain_randomization: bool = True,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization

        self.domain_randomization_config = RandomizationConfig()
        if isinstance(domain_randomization_config, dict):
            merged_config = self.domain_randomization_config.dict()
            common.dict_merge(merged_config, domain_randomization_config)
            for key, value in merged_config.items():
                if hasattr(self.domain_randomization_config, key):
                    setattr(self.domain_randomization_config, key, value)
        elif isinstance(domain_randomization_config, RandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        # Overlay state
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
        return SimConfig(sim_freq=100, control_freq=10)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, 0.3, 0.35], [0.3, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 52 * np.pi / 180, 0.01, 100)

    @property
    def apply_greenscreen(self) -> bool:
        return self.domain_randomization_config.apply_overlay

    def _load_scene(self, options: dict):
        self._objects_to_remove_from_greenscreen = []

    def _load_lighting(self, options: dict):
        if self.domain_randomization_config.realism_mode:
            # Just one real light: an overhead softbox (area light), matching the real
            # rig's own single box light (see simulation/usd/README.md) -- for one-off
            # visualization renders (see examples/render_realistic.py). A directional
            # "sun" light casts a hard-edged shadow no matter how dim, so this uses a
            # light with physical size for a soft, diffuse shadow, which needs the
            # ray-traced shader (rt/rt-fast) to render correctly. The lightbox's vertical
            # side walls naturally render darker than the floor under a straight-overhead
            # light (their surface normal faces sideways, away from the light, not up
            # toward it) -- moderate ambient softens that contrast. Training never sets
            # realism_mode, so this branch has no effect on the flat/shadowless lighting
            # used during RL.
            self.scene.set_ambient_light([0.3, 0.3, 0.32])
            self.scene.add_area_light_for_ray_tracing(
                sapien.Pose(p=[-0.245, -0.4, 1.0]), [16, 16, 14.4], 0.25, 0.5,
            )
            return

        if self.domain_randomization and self.domain_randomization_config.randomize_lighting:
            # Neutral white-gray light at a random per-env brightness. Intensity varies,
            # hue never does: object colors stay honest, and the light rig (point/box)
            # needs no hue management.
            levels = self._batched_episode_rng.uniform(0.2, 0.5)
            for i, scene in enumerate(self.scene.sub_scenes):
                scene.render_system.ambient_light = [levels[i]] * 3
        else:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])

        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_camera_mount(self):
        """Create the wrist- and overhead-camera mount actors used for sim-to-real jitter."""
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.wrist_camera_mount = builder.build_kinematic("wrist_camera_mount")

        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.overhead_camera_mount = builder.build_kinematic("overhead_camera_mount")

    def _randomize_robot_color(self):
        if self.domain_randomization_config.robot_color is None:
            return

        for link in self.agent.robot.links:
            for i, obj in enumerate(link._objs):
                render_body_component: RenderBodyComponent = obj.entity.find_component_by_type(
                    RenderBodyComponent
                )
                if render_body_component is None:
                    continue

                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if (
                            self.domain_randomization
                            and self.domain_randomization_config.robot_color == "random"
                        ):
                            color = self._batched_episode_rng[i].uniform(0.0, 1.0, size=(3,)).tolist()
                        else:
                            color = list(self.domain_randomization_config.robot_color)
                        part.material.set_base_color(color + [1])

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
            gripper_joint._objs[idx].set_drive_properties(stiffnesses[i], dampings[i], force_limit=100)
            self._gripper_stiffness[idx] = stiffnesses[i]
            self._gripper_damping[idx] = dampings[i]

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

        if not self._overlay_initialized and self._rgb_overlay_np is not None:
            for name, sensor in self._sensor_configs.items():
                if isinstance(sensor, CameraConfig) and name != "render_camera":
                    resized = cv2.resize(self._rgb_overlay_np, (sensor.width, sensor.height))
                    self._rgb_overlay_image = common.to_tensor(resized, device=self.device)
                    break

            if self._rgb_overlay_image is None and self._rgb_overlay_np is not None:
                self._rgb_overlay_image = common.to_tensor(self._rgb_overlay_np, device=self.device)

        if not self._overlay_initialized and self._rgb_overlay_image is None:
            for name, sensor in self._sensor_configs.items():
                if isinstance(sensor, CameraConfig) and name != "render_camera":
                    self._rgb_overlay_image = torch.zeros(
                        (sensor.height, sensor.width, 3), dtype=torch.uint8, device=self.device
                    )
                    break

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


def _sample_jitter_pose(num_envs: int, pos_noise, rot_noise, enabled: bool, device) -> Pose:
    """Small random (position, roll/pitch/yaw) offset pose, batched over envs. Used to
    jitter a camera mount on top of its URDF-calibrated base pose for sim-to-real
    robustness. Returns identity offsets when `enabled` is False."""
    if enabled:
        rand_vals = 2 * torch.rand(num_envs, 6, device=device) - 1
        dx = pos_noise[0] * rand_vals[:, 0]
        dy = pos_noise[1] * rand_vals[:, 1]
        dz = pos_noise[2] * rand_vals[:, 2]
        d_roll = rot_noise[0] * rand_vals[:, 3]
        d_pitch = rot_noise[1] * rand_vals[:, 4]
        d_yaw = rot_noise[2] * rand_vals[:, 5]
    else:
        dx = dy = dz = torch.zeros(num_envs, device=device)
        d_roll = d_pitch = d_yaw = torch.zeros(num_envs, device=device)

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
    """Wrist camera (follows the gripper) plus a static overhead camera for localizing the
    cube and bin. `FlattenRGBDObservationWrapper` concatenates both cameras' RGB along the
    channel axis (`rgb` becomes H x W x 6); `utils.ColorJitterWrapper` jitters each camera's
    3 channels independently since `torchvision.transforms.ColorJitter` requires exactly 3.

    Both cameras' poses come straight from forward kinematics of the URDF's calibrated
    `frame_wrist_camera`/`frame_overhead_camera` joints, plus small jitter for sim-to-real
    robustness.
    """

    # Base FOV of the URDF cameras (see simulation/mjcf/so101_on_frame.xml `fovy`, its twin).
    WRIST_CAMERA_FOV = np.deg2rad(58)
    OVERHEAD_CAMERA_FOV = np.deg2rad(60)

    @property
    def _default_sensor_configs(self):
        config = self.domain_randomization_config

        def fov_noise(scale):
            if self.domain_randomization and scale > 0:
                return scale * (2 * self._batched_episode_rng.rand() - 1)
            return 0

        return [
            CameraConfig(
                "wrist_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.WRIST_CAMERA_FOV + fov_noise(config.wrist_camera_fov_noise),
                near=0.01,
                far=100,
                mount=self.wrist_camera_mount,
            ),
            CameraConfig(
                "overhead_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=self.OVERHEAD_CAMERA_FOV + fov_noise(config.overhead_camera_fov_noise),
                near=0.01,
                far=100,
                mount=self.overhead_camera_mount,
            ),
        ]

    def _update_camera_poses(self):
        """Follow the URDF-calibrated camera links, plus small jitter on each."""
        config = self.domain_randomization_config

        wrist_jitter = _sample_jitter_pose(
            self.num_envs, config.wrist_camera_pos_noise, config.wrist_camera_rot_noise,
            self.domain_randomization, self.device,
        )
        self.wrist_camera_mount.set_pose(self.agent.wrist_camera_link.pose * wrist_jitter)

        overhead_jitter = _sample_jitter_pose(
            self.num_envs, config.overhead_camera_pos_noise, config.overhead_camera_rot_noise,
            self.domain_randomization, self.device,
        )
        self.overhead_camera_mount.set_pose(self.agent.overhead_camera_link.pose * overhead_jitter)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_camera_poses()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()
        # super().reset() rendered its obs before the camera mounts moved, so re-render:
        # otherwise every episode's first frame uses the previous episode's camera poses
        # (and the first-ever frame renders from uninitialized mounts at the origin).
        obs = self.get_obs(info)
        return obs, info

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        self._update_camera_poses()
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()


DefaultCameraEnv = DualCameraEnv
DefaultRandomizationConfig = RandomizationConfig
