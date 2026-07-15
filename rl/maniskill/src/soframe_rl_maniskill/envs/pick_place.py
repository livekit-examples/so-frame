"""SO-101-on-frame pick-and-place task: pick a cube, place it in a bin.

The ManiSkill3 + vision counterpart to ``rl/mjlab``'s task, adapted from
[Squint](https://github.com/aalmuzairee/squint)'s ``envs/place.py``. Differences from
Squint's version:

- Robot: this repo's frame-mounted ``so101_on_frame`` agent (rail + arm + calibrated
  wrist camera) instead of Squint's bare tabletop SO-101. See ``robot/so101_on_frame.py``.
- Scene: no ``TableSceneBuilder``. The frame (with its lightbox work surface) *is* the
  robot's own URDF, so the scene is just a ground plane plus the frame/arm.
- Cube/bin dimensions default to the same physical sizes already tuned for this rig in
  ``rl/mjlab/src/soframe_rl/assets.py`` (2.5 cm cube, 10 cm bin) rather than Squint's
  ranges, for consistency between the two RL backends.
- Observations default to vision + proprioception only (no ground-truth object poses,
  see ``_get_obs_extra`` below). This is Squint's own default too: its
  ``obs_mode="rgb+segmentation"`` has no privileged "state" component, which already
  satisfies "no state-based observations" for the policy. The privileged block is only
  populated if a caller explicitly asks for an obs_mode including "+state", useful for
  debugging/eval, never for training.

The URDF's root joint lifts the whole rig so the work surface sits at ``WORK_SURFACE_Z = 0``
(``rl/mjlab`` uses the same convention with its own lift amount). The lightbox's four panel
links have box collision matching their mesh bounds. See ``examples/measure_work_surface.py``
for how the geometry was measured.

``_load_scene`` puts the ground plane well below the work surface: it's a safety catch-all
for anything that rolls off, not the resting surface itself.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Union

import dacite
import numpy as np
import sapien
import sapien.render
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils import common
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from .base_random_env import DefaultCameraEnv, DefaultRandomizationConfig
from ..robot.so101_on_frame import SO101OnFrame

# The URDF's root joint already lifts the rig so this is 0 -- see the module docstring.
WORK_SURFACE_Z = 0.0

# Item and bin spawn from separate regions along the rail's travel axis (world -Y), far
# enough apart that the arm alone can't reach both without sliding: dof_slider moves the
# tcp along world Y across a 0.82 m range (y in [-0.878, -0.058] at the rest qpos), and
# these two regions sit on opposite ends of that range with margin to spare.
#
# The 2.5 cm cube gets the far region (y=-0.70): it sits dead-center in the overhead
# camera's frame there, where the near region (y=-0.30) is at the frame's edge and partly
# occluded by the arm -- measured at 0-16 px (128px render), with ~16% of spawns having
# zero pixels. The 10 cm bin is 16x the cube's area, so it stays clearly visible even in
# the edge region.
ITEM_SPAWN_CENTER = (-0.225, -0.70)
BIN_SPAWN_CENTER = (-0.225, -0.30)
SPAWN_HALF_SIZE = 0.1


@dataclass
class PickPlaceRandomizationConfig(DefaultRandomizationConfig):
    """Domain randomization config for the pick-and-place task."""

    # Noisy joint positions for better sim2real.
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Cube randomization (defaults match rl/mjlab's fixed 2.5 cm cube).
    cube_half_size_range: Sequence[float] = (0.0125, 0.0125)
    # Bin randomization (half sizes; defaults match rl/mjlab's fixed 10 cm interior /
    # 5 cm wall height bin -- note rl/mjlab's BIN_RIM_HEIGHT=0.05 is a *full* height, so
    # the z half-size here is half of that (0.025), not 0.05.
    bin_half_size_x_range: Sequence[float] = (0.05, 0.05)
    bin_half_size_y_range: Sequence[float] = (0.05, 0.05)
    bin_half_size_z_range: Sequence[float] = (0.025, 0.025)

    item_friction_range: Sequence[float] = (0.5, 1.0)
    item_density_range: Sequence[float] = (400, 400)  # ~30 g at 2.5 cm cube.
    randomize_item_color: bool = True
    randomize_bin_color: bool = True


@register_env("SOFramePickPlaceBin-v1", max_episode_steps=200)
class PickPlaceBin(DefaultCameraEnv):
    """
    **Task Description:**
    Pick up a cube and place it in a bin.

    **Randomizations:**
    - the cube's xy position is randomized within its own region near one end of the rail
    - the bin's xy position is randomized within its own region near the other end, far
      enough from the cube's region that reaching both requires sliding the rail
    - the cube's z-axis rotation is randomized

    **Success Conditions:**
    - the cube is within the bin's xy footprint
    - the robot is not touching the cube or the bin
    - the robot is (roughly) static
    """

    SUPPORTED_ROBOTS = ["so101_on_frame"]
    SUPPORTED_OBS_MODES = [
        "none", "state", "state_dict", "rgb", "rgb+segmentation",
        "rgb+state", "rgb+segmentation+state",
        "rgb+depth+segmentation", "rgb+depth+segmentation+state",
    ]
    agent: SO101OnFrame

    def __init__(
        self,
        *args,
        robot_uids="so101_on_frame",
        control_mode="pd_joint_target_delta_pos",
        domain_randomization_config: Union[
            PickPlaceRandomizationConfig, dict
        ] = PickPlaceRandomizationConfig(),
        domain_randomization=False,
        item_spawn_center=ITEM_SPAWN_CENTER,
        bin_spawn_center=BIN_SPAWN_CENTER,
        spawn_half_size=SPAWN_HALF_SIZE,
        **kwargs,
    ):
        self.domain_randomization_config = PickPlaceRandomizationConfig()
        merged = self.domain_randomization_config.dict()
        if isinstance(domain_randomization_config, dict):
            common.dict_merge(merged, domain_randomization_config)
            self.domain_randomization_config = dacite.from_dict(
                data_class=PickPlaceRandomizationConfig,
                data=merged,
                config=dacite.Config(strict=True),
            )
        elif isinstance(domain_randomization_config, PickPlaceRandomizationConfig):
            self.domain_randomization_config = domain_randomization_config

        self.item_spawn_center = item_spawn_center
        self.bin_spawn_center = bin_spawn_center
        self.spawn_half_size = spawn_half_size

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            sapien.Pose(),
            build_separate=True
            if self.domain_randomization
            and self.domain_randomization_config.robot_color == "random"
            else False,
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # The lightbox floor at WORK_SURFACE_Z (0) is the real resting surface; this
        # ground plane sits far below it purely as a safety catch-all for anything that
        # rolls off the panels.
        self.ground = build_ground(self.scene, altitude=-1.0)

        cfg = self.domain_randomization_config

        def sample_range(value_range):
            """Per-env uniform sample under domain randomization, midpoint otherwise."""
            if self.domain_randomization:
                return self._batched_episode_rng.uniform(low=value_range[0], high=value_range[1])
            return np.full(self.num_envs, (value_range[0] + value_range[1]) / 2)

        # The work surface's panels render near-white (base color ~0.9). A random color
        # whose every channel lands within ~20/255 of that is indistinguishable from the
        # floor at the squinted training resolution, making the episode unsolvable by
        # perception -- redraw those (full random hues otherwise preserved).
        _WORK_SURFACE_RGB = 0.902

        def sample_visible_colors():
            colors = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
            for _ in range(8):
                too_close = np.abs(colors - _WORK_SURFACE_RGB).max(axis=-1) < 0.08
                if not too_close.any():
                    break
                redraw = self._batched_episode_rng.uniform(low=0, high=1, size=(3,))
                colors[too_close] = redraw[too_close]
            return colors

        half_sizes = sample_range(cfg.cube_half_size_range)
        frictions = sample_range(cfg.item_friction_range)
        densities = sample_range(cfg.item_density_range)

        colors = np.zeros((self.num_envs, 3))
        colors[:, 2] = 1  # Blue cube.
        if self.domain_randomization and cfg.randomize_item_color:
            colors = sample_visible_colors()

        self.item_half_sizes = common.to_tensor(half_sizes, device=self.device)
        self.item_dimensions = torch.stack([self.item_half_sizes] * 3, dim=-1)
        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction, dynamic_friction=friction, restitution=0,
            )
            builder.add_box_collision(
                half_size=[half_sizes[i]] * 3, material=material, density=densities[i]
            )
            item_material = sapien.render.RenderMaterial(base_color=colors[i])
            if self.domain_randomization_config.realism_mode:
                item_material.set_roughness(0.6)
                item_material.set_metallic(0.0)
            builder.add_box_visual(half_size=[half_sizes[i]] * 3, material=item_material)
            builder.initial_pose = sapien.Pose(p=[0.2, 0, half_sizes[i]])
            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        bin_colors = np.ones((self.num_envs, 3)) * [0.55, 0.45, 0.05]  # dark yellow
        if self.domain_randomization and cfg.randomize_bin_color:
            bin_colors = sample_visible_colors()
        bin_colors = np.concatenate([bin_colors, np.ones((self.num_envs, 1))], axis=-1)
        thickness = 0.004
        self.bin_thickness = thickness

        bin_half_sizes_x = sample_range(cfg.bin_half_size_x_range)
        bin_half_sizes_y = sample_range(cfg.bin_half_size_y_range)
        bin_half_sizes_z = sample_range(cfg.bin_half_size_z_range)

        self.bin_half_sizes_x = common.to_tensor(bin_half_sizes_x, device=self.device)
        self.bin_half_sizes_y = common.to_tensor(bin_half_sizes_y, device=self.device)
        self.bin_half_sizes_z = common.to_tensor(bin_half_sizes_z, device=self.device)
        self.bin_dimensions = torch.stack(
            [self.bin_half_sizes_x, self.bin_half_sizes_y, self.bin_half_sizes_z], dim=-1
        )

        bins = []
        for i in range(self.num_envs):
            bin_half_size = [bin_half_sizes_x[i], bin_half_sizes_y[i], bin_half_sizes_z[i]]
            builder = self.scene.create_actor_builder()

            bin_color = sapien.render.RenderMaterial(base_color=bin_colors[i])
            if self.domain_randomization_config.realism_mode:
                bin_color.set_roughness(0.55)
                bin_color.set_metallic(0.0)

            bin_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
            bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
            builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)

            for j in [-1, 1]:
                y = j * bin_center_half_size[1]
                wall_pose = sapien.Pose([0, y, bin_half_size[2]])
                wall_half_size = [bin_half_size[0], thickness / 2, bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)
                x = j * bin_center_half_size[0]
                wall_pose = sapien.Pose([x, 0, bin_half_size[2]])
                wall_half_size = [thickness / 2, bin_half_size[1], bin_half_size[2]]
                builder.add_box_collision(pose=wall_pose, half_size=wall_half_size)
                builder.add_box_visual(pose=wall_pose, half_size=wall_half_size, material=bin_color)

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, bin_half_size[2]])
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)
        self.bin_radius = torch.linalg.norm(self.bin_dimensions[:, :2], dim=-1)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bin)

        self.rest_qpos = common.to_tensor(
            SO101OnFrame.keyframes["rest"].qpos.tolist(), device=self.device
        )

        self._load_camera_mount()
        self._randomize_robot_color()
        if self.domain_randomization_config.realism_mode:
            self.agent.apply_realistic_materials()

        goal_builder = self.scene.create_actor_builder()
        goal_builder.add_sphere_visual(
            radius=0.01, material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])
        )
        goal_builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.goal_site = goal_builder.build_kinematic(name="goal_site")
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)

            self.agent.robot.set_qpos(
                self.rest_qpos
                + torch.randn(size=(b, self.rest_qpos.shape[-1]))
                * self.domain_randomization_config.initial_qpos_noise_scale
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=[0, 0, 0]))

            item_spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.item_spawn_center[0], self.item_spawn_center[1], 0]
            )
            bin_spawn_center = self.agent.robot.pose.p + torch.tensor(
                [self.bin_spawn_center[0], self.bin_spawn_center[1], 0]
            )

            # Item and bin spawn in separate, non-overlapping regions (0.4 m apart along
            # the rail), so their offsets are just independent uniform samples -- no
            # placement-collision handling needed between them.
            item_xy_offset = (torch.rand(b, 2) * 2 - 1) * self.spawn_half_size
            bin_xy_offset = (torch.rand(b, 2) * 2 - 1) * self.spawn_half_size

            item_xyz = torch.zeros((b, 3))
            item_xyz[:, :2] = item_spawn_center[env_idx, :2] + item_xy_offset
            item_xyz[:, 2] = WORK_SURFACE_Z + self.item_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = bin_spawn_center[env_idx, :2] + bin_xy_offset
            bin_xyz[:, 2] = WORK_SURFACE_Z + self.bin_thickness / 2
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, qs))

            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = WORK_SURFACE_Z + self.bin_thickness + self.item_half_sizes[env_idx]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_agent(self):
        qpos = self.agent.robot.get_qpos()
        if self.domain_randomization and self.domain_randomization_config.robot_qpos_noise_std > 0:
            noise = torch.randn_like(qpos) * self.domain_randomization_config.robot_qpos_noise_std
            qpos = qpos + noise
        obs = dict(noisy_qpos=qpos)
        controller_state = self.agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def _get_obs_extra(self, info: dict):
        """Privileged ground-truth state. Only populated for an explicit "+state" obs_mode.

        Training uses the default obs_mode ("rgb+segmentation", no "state" component), so
        this stays empty: the policy sees vision + proprioception only, never ground-truth
        object poses. Useful for offline eval/debugging with `obs_mode="rgb+segmentation+state"`.
        """
        obs = dict()
        if self.obs_mode_struct.state:
            obs.update(
                qvel=self.agent.robot.get_qvel(),
                is_item_grasped=info["is_item_grasped"],
                item_pose=self.item.pose.raw_pose,
                bin_pose=self.bin.pose.raw_pose,
                tcp_pose=self.agent.tcp_pose.raw_pose,
                tcp_to_item_grip_pos=self.item.pose.p - self.agent.tcp_pos,
                tcp_to_bin_pos=self.bin.pose.p - self.agent.tcp_pos,
                item_to_bin_pos=self.bin.pose.p - self.item.pose.p,
            )
            if self.domain_randomization:
                gripper_params = self.get_gripper_params()
                obs.update(
                    clean_qpos=self.agent.robot.get_qpos(),
                    item_dimensions=self.item_dimensions,
                    bin_dimensions=self.bin_dimensions,
                    item_friction=self.item_frictions,
                    item_density=self.item_densities,
                    gripper_stiffness=gripper_params["gripper_stiffness"],
                    gripper_damping=gripper_params["gripper_damping"],
                )
        return obs

    def evaluate(self):
        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        bin_pos[:, 2] = WORK_SURFACE_Z + self.bin_thickness + self.item_half_sizes

        offset = item_pos - bin_pos
        inside_x = torch.abs(offset[:, 0]) < self.bin_half_sizes_x
        inside_y = torch.abs(offset[:, 1]) < self.bin_half_sizes_y
        is_item_above_bin = inside_x & inside_y
        # Actually settled at its resting height inside the bin -- not still falling
        # toward it, and not perched on a wall rim (a rim-rest sits ~4.5 cm higher).
        is_item_in_bin = is_item_above_bin & (offset[:, 2] < 0.01)

        item_lifted = self.item.pose.p[..., -1] >= (WORK_SURFACE_Z + self.item_half_sizes + 1e-3)

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_item_grasped = self.agent.is_grasping(self.item)
        is_robot_static = self.agent.is_static()

        robot_touching_bin = self.agent.is_touching(self.bin)
        robot_touching_item = self.agent.is_touching(self.item)

        success = (
            is_item_in_bin & is_item_static
            & (~robot_touching_item) & is_robot_static & (~robot_touching_bin)
        )

        return {
            "inside_x": inside_x,
            "inside_y": inside_y,
            "item_vel": item_vel,
            "item_lifted": item_lifted,
            "is_item_static": is_item_static,
            "success": success,
            "is_item_above_bin": is_item_above_bin,
            "is_item_in_bin": is_item_in_bin,
            "is_item_grasped": is_item_grasped,
            "is_robot_static": is_robot_static,
            "robot_touching_bin": robot_touching_bin,
            "robot_touching_item": robot_touching_item,
        }

    def _gripper_qpos_openness(self):
        idx = self.agent.joint_names.index("gripper")
        gripper_min, gripper_max = self.agent.robot.get_qlimits()[0, idx, :]
        return (self.agent.robot.get_qpos()[:, idx] - gripper_min) / (gripper_max - gripper_min)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_item_dist = torch.linalg.norm(self.agent.tcp_pos - self.item.pose.p, axis=1)
        reaching_reward = 2 * (1 - torch.tanh(5 * tcp_to_item_dist))
        reward = reaching_reward

        item_pos = self.item.pose.p
        bin_pos = self.bin.pose.p.clone()
        goal_xyz = bin_pos.clone()
        goal_xyz[..., 2] = WORK_SURFACE_Z + self.bin_thickness + self.item_half_sizes

        item_to_goal_dist = torch.linalg.norm(goal_xyz - item_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal_dist)

        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        item_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + (self.bin_dimensions[:, 2:] * 2) + 0.03) - item_pos[..., 2:], dim=1
        )
        item_to_goal_dist_z_close = torch.linalg.norm(goal_xyz[..., 2:] - item_pos[..., 2:], dim=1)
        item_close_to_goal = item_to_goal_dist_xy <= self.bin_radius
        item_to_goal_dist_z = torch.where(item_close_to_goal, item_to_goal_dist_z_close, item_to_goal_dist_z_far)
        place_reward_z = 1 - torch.tanh(10.0 * item_to_goal_dist_z)
        place_reward = place_reward_final + place_reward_z

        gripper_openness = self._gripper_qpos_openness()

        reward[info["is_item_grasped"]] = (3 + place_reward)[info["is_item_grasped"]]

        is_item_dropped = (~info["robot_touching_item"]).float()
        arm_idx = [i for i, n in enumerate(self.agent.joint_names) if n != "gripper"]
        robot_v = torch.linalg.norm(self.agent.robot.get_qvel()[:, arm_idx], axis=1)
        # Only pays out once the item is actually released: holding it steady above the
        # bin forever otherwise nearly matches the success reward, so the policy has no
        # incentive to let go.
        static_robot_reward = (1 - torch.tanh(robot_v * 10)) * is_item_dropped
        reward[info["is_item_above_bin"]] = (
            4 + place_reward + is_item_dropped + gripper_openness + static_robot_reward
        )[info["is_item_above_bin"]]

        reward[info["success"]] = 20

        reward -= 3 * info["robot_touching_bin"].float()
        reward -= 1 * (~info["item_lifted"]).float()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20
