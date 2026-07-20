"""Base environment with domain randomization and dual-camera vision.

Adapted from [Squint](https://github.com/aalmuzairee/squint)'s ``envs/base_random_env.py``.
Handles:
- Gripper stiffness/damping randomization
- Lighting randomization
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
    visual_fidelity: str = "raster"
    """Rendering fidelity, one of:
    - "flat": the fast shadowless lighting, kept for cheap ablations.
    - "raster" (default): PBR materials under softbox-like lighting (dominant shadowless
      near-vertical light plus a faint shadow-casting key, approximating the real rig's
      diffuse lightbox and the "raytraced" area light). Compatible with the GPU-parallel
      sensor cameras (shadows and textures render fine on their standard memory-optimized
      "minimal" shader), so it's trainable at full env counts.
    - "raytraced": the same PBR materials plus an overhead softbox area light matching
      the real rig. Renders correctly only under the rt/rt-fast shaders, which the
      GPU-parallel sensor path doesn't support -- for one-off renders and single-env
      evals (see ``examples/render_realistic.py``), never training."""

    # === Common randomization settings (affected by domain_randomization flag) ===
    action_delay_steps_range: Sequence[int] = (0, 1)
    """Per-episode action delay in control steps, sampled uniformly (inclusive ends)
    under domain randomization; 0 delay when domain_randomization is off. Models the
    real control loop's camera-capture -> inference -> servo-write latency (measured
    roughly 0.5-1.5 steps at 10 Hz control, hence (0, 1)): the policy acts on what the
    world looked like `delay` steps ago. A wider (0, 2) spread was tried in v26/v27 and
    made contact timing needlessly uncertain during grasping. Delayed slots start as
    zero actions (= hold, under the delta target controller) at episode start."""
    gripper_stiffness_range: Sequence[float] = (500, 2000)
    """Range for gripper joint stiffness randomization (per-episode)."""
    gripper_damping_range: Sequence[float] = (50, 200)
    """Range for gripper joint damping randomization (per-episode)."""
    binary_gripper: bool = True
    """Near-binary gripper: threshold the gripper action to its sign each step so it only
    ever drives fully open or fully closed (the arm/rail stay continuous delta control).
    Makes 'release' unambiguously fully-open -- robust to the sim-vs-real gripper opening
    gap that left a real bar gripped -- and needs only two calibrated gripper endpoints.
    The action space stays continuous (SAC unchanged); only the sign matters."""
    arm_stiffness_range: Sequence[float] = (600, 1400)
    """Per-episode stiffness range for the arm + rail joints (nominal 1e3). The real
    STS3215 servos run their own controller; a fixed sim PD overfits its exact response,
    so vary it +-40% for sim2real robustness. Set (1000, 1000) to disable."""
    arm_damping_range: Sequence[float] = (60, 140)
    """Per-episode damping range for the arm + rail joints (nominal 1e2). See above."""
    randomize_lighting: bool = True
    """Whether to randomize ambient lighting."""

    # === Camera FOV (base value; per-episode fov_noise below jitters around it) ===
    wrist_camera_fov: float = np.deg2rad(58)
    """Base vertical FOV of the wrist camera. Default comes from the MJCF twin's `fovy`;
    override with the measured real-camera value (the deploy mapping saved by
    ``examples/calibrate_real_camera.py`` reports it as ``fov_deg``)."""
    overhead_camera_fov: float = np.deg2rad(38)
    """Base vertical FOV of the overhead camera. Calibrated against the real rig
    (2026-07: interactive alignment via examples/move_sim_camera.py with the URDF pose
    held fixed; the deploy-side counterpart lives in overhead_camera_mapping.json). The
    URDF/MJCF twin's fovy said 60, which does not reproduce the real camera's view."""

    # === Camera pose correction (constant, on top of the URDF link pose) ===
    # The URDF camera links place the printed holders (CAD mate connectors); the actual
    # lens can sit closer/tilted relative to that. These offsets are expressed in the
    # camera link's local frame (+X = view direction) and are applied before jitter.
    # Calibrate by comparing a sim render against a rectified real frame
    # (examples/calibrate_real_camera.py) and nudging until they align.
    wrist_camera_pos_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant position offset (meters, camera-local: +X forward along the view)."""
    wrist_camera_rot_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant rotation offset (roll, pitch, yaw in radians, camera-local)."""
    overhead_camera_pos_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant position offset (meters, camera-local: +X forward along the view)."""
    overhead_camera_rot_offset: Sequence[float] = (0.0, 0.0, 0.0)
    """Constant rotation offset (roll, pitch, yaw in radians, camera-local)."""

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

    @classmethod
    def resolve(cls, config: Union["RandomizationConfig", dict]) -> "RandomizationConfig":
        """Merge a dict of overrides into this class's defaults (instances pass through)."""
        if isinstance(config, cls):
            return config
        merged = asdict(cls())
        common.dict_merge(merged, config if isinstance(config, dict) else asdict(config))
        return dacite.from_dict(data_class=cls, data=merged, config=dacite.Config(strict=True))


class BaseRandomEnv(BaseEnv):
    """Base environment with domain randomization and overlay support.

    Subclasses (e.g. ``pick_place.PickPlaceBin``) add the task-specific scene and
    inherit the cameras from ``DualCameraEnv``.
    """

    def __init__(
        self,
        *args,
        domain_randomization_config: Union[RandomizationConfig, dict] = RandomizationConfig(),
        domain_randomization: bool = True,
        **kwargs,
    ):
        self.domain_randomization = domain_randomization
        self.domain_randomization_config = RandomizationConfig.resolve(domain_randomization_config)

        # Action-delay state (see RandomizationConfig.action_delay_steps_range)
        self._action_delay: Optional[torch.Tensor] = None
        self._action_queue: Optional[torch.Tensor] = None

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
        fidelity = self.domain_randomization_config.visual_fidelity
        if fidelity == "raytraced":
            # One overhead softbox (area light), matching the real rig's single box light.
            # SAPIEN lights emit along +x of their pose, so the pose rotates +x straight
            # down; the panel is centered over the spawn regions and oversized so shadows
            # wash out to near-nothing. rt/rt-fast shaders only.
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
            # The lightbox is a diffuse source surrounding the workspace: light arrives
            # from everywhere, which in a rasterizer is ambient, not directional. The
            # boost keeps the arm's VERTICAL surfaces lit (a straight-down directional
            # light leaves them dark; their normals never face it).
            ambient_colors = ambient_colors + 0.25
        for i, scene in enumerate(self.scene.sub_scenes):
            scene.render_system.ambient_light = ambient_colors[i]

        if fidelity == "raster":
            # Rasterizer stand-in for the softbox. The real lightbox leaves the
            # workspace essentially shadow-free (verified against real captures and the
            # "raytraced" area-light render). Directional shadow maps can only cast
            # hard-edged shadows, so softness is approximated by ratio instead: most
            # illumination is the boosted omnidirectional ambient above plus a modest
            # shadowless vertical fill, and the single shadow-casting key is faint,
            # leaving just a subtle darkening where the arm blocks it (shadowed:lit is
            # roughly 0.75:1 on the surface, versus about 0.3:1 with the old strong key
            # light, which cast a heavy shadow blob the real rig never shows).
            # shadow_map_size stays small: shadow maps allocate per env (2048^2 across
            # 1024 envs is ~17 GB of VRAM); 512^2 is plenty for 128 px observations.
            self.scene.add_directional_light(
                [0.15, 0.1, -1], [0.4, 0.39, 0.38], shadow=True, shadow_scale=2.0, shadow_map_size=512,
            )
            self.scene.add_directional_light([0, 0, -1], [0.4, 0.39, 0.38])
            self.scene.add_directional_light([-1, -0.3, -0.6], [0.25, 0.25, 0.28])
            return

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

    def _randomize_arm_gains(self, env_idx: torch.Tensor):
        """Per-episode stiffness/damping on the arm + rail joints (everything but the
        gripper, which has its own randomization). Models the real servos' varying
        response so the policy doesn't overfit sim's fixed PD."""
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
                joint._objs[idx].set_drive_properties(stiffnesses[i], dampings[i], force_limit=100)

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
            # The first non-render sensor camera sets the overlay resolution; with no
            # overlay image configured, a black one is used.
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
        self._randomize_action_delay(env_idx)

    def _randomize_action_delay(self, env_idx: torch.Tensor):
        lo, hi = self.domain_randomization_config.action_delay_steps_range
        if self._action_delay is None:
            self._action_delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if self.domain_randomization and hi > 0:
            self._action_delay[env_idx] = torch.randint(
                int(lo), int(hi) + 1, (len(env_idx),), device=self.device
            )
        else:
            self._action_delay[env_idx] = 0
        if self._action_queue is not None:
            self._action_queue[env_idx] = 0.0

    def step(self, action):
        """Near-binary gripper (threshold to sign) + action delay (apply the action from
        `delay` steps ago per-env, see RandomizationConfig)."""
        cfg = self.domain_randomization_config
        if cfg.binary_gripper:
            gi = self.agent.joint_names.index("gripper")
            act = common.to_tensor(action, device=self.device).clone().float()
            act[..., gi] = torch.where(act[..., gi] > 0, 1.0, -1.0)
            action = act
        max_delay = int(self.domain_randomization_config.action_delay_steps_range[1])
        if self.domain_randomization and max_delay > 0 and self._action_delay is not None:
            act = common.to_tensor(action, device=self.device).float()
            squeeze = act.ndim == 1
            if squeeze:
                act = act.unsqueeze(0)
            if self._action_queue is None or self._action_queue.shape[-1] != act.shape[-1]:
                self._action_queue = torch.zeros(
                    (self.num_envs, max_delay + 1, act.shape[-1]), device=self.device
                )
            self._action_queue = torch.roll(self._action_queue, shifts=1, dims=1)
            self._action_queue[:, 0] = act
            applied = self._action_queue[
                torch.arange(self.num_envs, device=self.device), self._action_delay
            ]
            action = applied.squeeze(0) if squeeze else applied
        return super().step(action)


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
    bar and bin. `FlattenRGBDObservationWrapper` concatenates both cameras' RGB along the
    channel axis (`rgb` becomes H x W x 6); `utils.ColorJitterWrapper` jitters each camera's
    3 channels independently since `torchvision.transforms.ColorJitter` requires exactly 3.

    Both cameras' poses come straight from forward kinematics of the URDF's calibrated
    `frame_wrist_camera`/`frame_overhead_camera` joints, plus small jitter for sim-to-real
    robustness.
    """

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
                fov=config.wrist_camera_fov + fov_noise(config.wrist_camera_fov_noise),
                near=0.01,
                far=100,
                mount=self.wrist_camera_mount,
            ),
            CameraConfig(
                "overhead_camera",
                pose=sapien.Pose(),
                width=128,
                height=128,
                fov=config.overhead_camera_fov + fov_noise(config.overhead_camera_fov_noise),
                near=0.01,
                far=100,
                mount=self.overhead_camera_mount,
            ),
        ]

    @staticmethod
    def _offset_pose(pos_offset, rot_offset) -> sapien.Pose:
        """Constant local-frame correction pose from (xyz, rpy) config values."""
        from transforms3d.euler import euler2quat

        return sapien.Pose(p=list(pos_offset), q=euler2quat(*rot_offset))

    def _update_camera_poses(self):
        """Follow the URDF-calibrated camera links, plus a constant calibration offset
        and small per-episode jitter on each."""
        config = self.domain_randomization_config

        wrist_offset = self._offset_pose(config.wrist_camera_pos_offset, config.wrist_camera_rot_offset)
        wrist_jitter = _sample_jitter_pose(
            self.num_envs, config.wrist_camera_pos_noise, config.wrist_camera_rot_noise,
            self.domain_randomization, self.device,
        )
        self.wrist_camera_mount.set_pose(self.agent.wrist_camera_link.pose * wrist_offset * wrist_jitter)

        overhead_offset = self._offset_pose(config.overhead_camera_pos_offset, config.overhead_camera_rot_offset)
        overhead_jitter = _sample_jitter_pose(
            self.num_envs, config.overhead_camera_pos_noise, config.overhead_camera_rot_noise,
            self.domain_randomization, self.device,
        )
        self.overhead_camera_mount.set_pose(self.agent.overhead_camera_link.pose * overhead_offset * overhead_jitter)

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
