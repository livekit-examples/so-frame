"""SO-101-on-frame pick-and-place task: pick a flat bar, place it in a bin.

The ManiSkill3 + vision counterpart to ``rl/mjlab``'s task, adapted from
[Squint](https://github.com/aalmuzairee/squint)'s ``envs/place.py``: this repo's
frame-mounted ``so101_on_frame`` agent (rail + arm + calibrated cameras, see
``robot/so101_on_frame.py``) replaces Squint's bare tabletop SO-101, and there's no
``TableSceneBuilder`` -- the frame with its lightbox work surface is part of the robot's
own URDF, so the scene is just the frame/arm, the bar, the bin, and a ground plane far
below as a safety catch-all. Observations are vision + proprioception only; privileged
ground-truth state exists solely behind an explicit "+state" obs_mode (see
``_get_obs_extra``).

The URDF's root joint lifts the rig so the lightbox work surface sits at
``WORK_SURFACE_Z = 0``, measured from the loaded articulation plus the panel STL's bounds
(``rl/mjlab`` uses the same convention with its own lift amount).
"""

from dataclasses import dataclass
from typing import Any, Sequence, Union

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

from .base_random_env import DualCameraEnv, RandomizationConfig
from ..robot.so101_on_frame import SO101OnFrame

# The URDF's root joint already lifts the rig so this is 0 -- see the module docstring.
WORK_SURFACE_Z = 0.0

# Item and bin spawn from separate regions along the rail's travel axis (world -Y), far
# enough apart that the arm alone can't reach both without sliding (dof_slider covers
# y in [-0.878, -0.058] at the rest qpos). The small bar gets the far region, dead-center
# in the overhead camera's frame; in the near region it can shrink to zero visible pixels
# behind the arm. The much larger bin stays visible in either.
ITEM_SPAWN_CENTER = (-0.225, -0.70)
BIN_SPAWN_CENTER = (-0.225, -0.30)
SPAWN_HALF_SIZE = 0.1


@dataclass
class PickPlaceRandomizationConfig(RandomizationConfig):
    """Domain randomization config for the pick-and-place task."""

    # Noisy joint positions for better sim2real.
    robot_qpos_noise_std: float = np.deg2rad(5)
    # Item randomization: a flat bar matching the real object, 75 x 25 x 15 mm. Its
    # footprint diagonal (~79 mm) fits the real bin's ~84 mm clear interior at any yaw,
    # and it's grasped across its 25 mm width.
    item_half_size_x_range: Sequence[float] = (0.0375, 0.0375)
    item_half_size_y_range: Sequence[float] = (0.0125, 0.0125)
    item_half_size_z_range: Sequence[float] = (0.0075, 0.0075)
    # Bin randomization (half sizes), matching the real bin: 85 x 85 mm footprint,
    # 35 mm walls, 1 mm wall thickness.
    bin_half_size_x_range: Sequence[float] = (0.0425, 0.0425)
    bin_half_size_y_range: Sequence[float] = (0.0425, 0.0425)
    bin_half_size_z_range: Sequence[float] = (0.0175, 0.0175)

    item_friction_range: Sequence[float] = (0.5, 1.0)
    item_density_range: Sequence[float] = (400, 400)  # ~12 g for the default bar.
    # Fixed colors (blue bar, dark yellow bin) by default: color randomization is
    # supported (with the visibility floor below) but costs substantial sample
    # efficiency, so it's opt-in for a dedicated color-generalization run.
    randomize_item_color: bool = False
    randomize_bin_color: bool = False


@register_env("SOFramePickPlaceBin-v1", max_episode_steps=200)
class PickPlaceBin(DualCameraEnv):
    """
    **Task Description:**
    Pick up a flat bar and place it in a bin.

    **Randomizations:**
    - the bar's xy position is randomized within its own region near one end of the rail
    - the bin's xy position is randomized within its own region near the other end, far
      enough from the bar's region that reaching both requires sliding the rail
    - the bar's z-axis rotation is randomized

    **Success Conditions:**
    - the bar rests settled inside the bin
    - the robot is not touching the bar or the bin
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
        action_rate_penalty=0.0,
        **kwargs,
    ):
        self.domain_randomization_config = PickPlaceRandomizationConfig.resolve(
            domain_randomization_config
        )

        self.item_spawn_center = item_spawn_center
        self.bin_spawn_center = bin_spawn_center
        self.spawn_half_size = spawn_half_size
        # Coefficient for the action-rate (smoothness) penalty: -k * ||a_t - a_{t-1}||^2
        # per step. Off by default; penalizes jerk, not movement, so the long slider
        # traverses the task needs stay untaxed.
        self.action_rate_penalty = action_rate_penalty
        self._prev_action = None

        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            domain_randomization=domain_randomization,
            domain_randomization_config=self.domain_randomization_config,
            **kwargs,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # The lightbox floor at WORK_SURFACE_Z (0) is the real resting surface; this
        # ground plane sits far below it purely as a safety catch-all for anything that
        # rolls off the panels.
        self.ground = build_ground(self.scene, altitude=-1.0)

        cfg = self.domain_randomization_config
        realistic = cfg.visual_fidelity != "flat"  # PBR materials for "raster"/"raytraced"

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

        half_x = sample_range(cfg.item_half_size_x_range)
        half_y = sample_range(cfg.item_half_size_y_range)
        half_z = sample_range(cfg.item_half_size_z_range)
        frictions = sample_range(cfg.item_friction_range)
        densities = sample_range(cfg.item_density_range)

        colors = np.zeros((self.num_envs, 3))
        colors[:, 2] = 1  # Blue bar.
        if self.domain_randomization and cfg.randomize_item_color:
            colors = sample_visible_colors()

        # Vertical half-extent: resting/goal heights and the lifted check key off this.
        self.item_half_heights = common.to_tensor(half_z, device=self.device)
        self.item_dimensions = torch.stack(
            [common.to_tensor(half_x, device=self.device),
             common.to_tensor(half_y, device=self.device),
             self.item_half_heights], dim=-1,
        )
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
            item_half_size = [half_x[i], half_y[i], half_z[i]]
            builder.add_box_collision(
                half_size=item_half_size, material=material, density=densities[i]
            )
            item_material = sapien.render.RenderMaterial(base_color=colors[i])
            if realistic:
                item_material.set_roughness(0.6)
                item_material.set_metallic(0.0)
            builder.add_box_visual(half_size=item_half_size, material=item_material)
            builder.initial_pose = sapien.Pose(p=[0.2, 0, half_z[i]])
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
        # Real bin walls are 1 mm. That's too thin for reliable contact at the 10 ms
        # physics step (a bar at 0.5 m/s crosses 5 mm per step), so walls keep an honest
        # 1 mm visual and get a thicker collision box extended outward, inner faces
        # aligned -- the interior clearance stays true to the real bin.
        thickness = 0.001
        wall_collision_thickness = 0.004
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
            if realistic:
                bin_color.set_roughness(0.55)
                bin_color.set_metallic(0.0)

            bin_center_pose = sapien.Pose([0.0, 0.0, thickness / 2])
            bin_center_half_size = [bin_half_size[0], bin_half_size[1], thickness / 2]
            builder.add_box_collision(pose=bin_center_pose, half_size=bin_center_half_size)
            builder.add_box_visual(pose=bin_center_pose, half_size=bin_center_half_size, material=bin_color)

            # Collision walls sit outward of the visual walls with inner faces aligned
            # (see wall_collision_thickness above).
            col_off_x = bin_half_size[0] - thickness / 2 + wall_collision_thickness / 2
            col_off_y = bin_half_size[1] - thickness / 2 + wall_collision_thickness / 2
            for j in [-1, 1]:
                builder.add_box_visual(
                    pose=sapien.Pose([0, j * bin_center_half_size[1], bin_half_size[2]]),
                    half_size=[bin_half_size[0], thickness / 2, bin_half_size[2]],
                    material=bin_color,
                )
                builder.add_box_collision(
                    pose=sapien.Pose([0, j * col_off_y, bin_half_size[2]]),
                    half_size=[bin_half_size[0], wall_collision_thickness / 2, bin_half_size[2]],
                )
                builder.add_box_visual(
                    pose=sapien.Pose([j * bin_center_half_size[0], 0, bin_half_size[2]]),
                    half_size=[thickness / 2, bin_half_size[1], bin_half_size[2]],
                    material=bin_color,
                )
                builder.add_box_collision(
                    pose=sapien.Pose([j * col_off_x, 0, bin_half_size[2]]),
                    half_size=[wall_collision_thickness / 2, bin_half_size[1], bin_half_size[2]],
                )

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
        if realistic:
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
        if self._prev_action is not None:
            self._prev_action[env_idx] = 0.0
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
            item_xyz[:, 2] = WORK_SURFACE_Z + self.item_half_heights[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.item.set_pose(Pose.create_from_pq(item_xyz, qs))

            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = bin_spawn_center[env_idx, :2] + bin_xy_offset
            bin_xyz[:, 2] = WORK_SURFACE_Z + self.bin_thickness / 2
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, qs))

            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = WORK_SURFACE_Z + self.bin_thickness + self.item_half_heights[env_idx]
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
        bin_pos[:, 2] = WORK_SURFACE_Z + self.bin_thickness + self.item_half_heights

        offset = item_pos - bin_pos
        inside_x = torch.abs(offset[:, 0]) < self.bin_half_sizes_x
        inside_y = torch.abs(offset[:, 1]) < self.bin_half_sizes_y
        is_item_above_bin = inside_x & inside_y
        # In the bin means part of the bar reaches the bin floor: its lowest corner is
        # within a tolerance of the floor's top. A tilted bar leaning on a wall counts
        # (real drops often settle that way); one bridging flat across the rim, or still
        # falling above the bin, does not.
        item_rot = self.item.pose.to_transformation_matrix()[..., :3, :3]
        item_lowest = item_pos[:, 2] - (item_rot[..., 2, :].abs() * self.item_dimensions).sum(-1)
        touches_bottom = item_lowest <= WORK_SURFACE_Z + self.bin_thickness + 0.005
        is_item_in_bin = is_item_above_bin & touches_bottom

        item_lifted = self.item.pose.p[..., -1] >= (WORK_SURFACE_Z + self.item_half_heights + 1e-3)

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
        # The carry target is a DROP height just above the bin's rim, not the resting
        # height on its floor: inside the bin the moving jaw has no room to swing open
        # (84 mm interior), so a policy shaped to insert at depth ends up physically
        # unable to release. From rim + 1 cm the jaw opens freely and gravity finishes
        # the placement. Success itself still requires the bar settled inside the bin.
        goal_xyz[..., 2] = (
            WORK_SURFACE_Z + self.bin_dimensions[:, 2] * 2 + self.item_half_heights + 0.01
        )

        item_to_goal_dist = torch.linalg.norm(goal_xyz - item_pos, axis=1)
        place_reward_final = 1 - torch.tanh(5.0 * item_to_goal_dist)

        item_to_goal_dist_xy = torch.linalg.norm(goal_xyz[..., :2] - item_pos[..., :2], dim=1)
        item_to_goal_dist_z_far = torch.linalg.norm(
            (goal_xyz[..., 2:] + 0.03) - item_pos[..., 2:], dim=1
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

        # Hovering over the bin while still gripping must not be a comfortable resting
        # point: a flat tax drops it to roughly carry-stage value, so releasing is the
        # only way to earn more (release pays ~9, success 20).
        reward -= 1 * (info["is_item_grasped"] & info["is_item_above_bin"]).float()
        # Kept mild: placing a 75 mm bar into the 84 mm interior means the gripper
        # fingers enter the opening, so brushing a wall is part of a good placement,
        # not something to scare the policy away from (success still requires ending
        # clear of the bin).
        reward -= 0.5 * info["robot_touching_bin"].float()
        # Penalize leaving the bar sitting on the work surface, not being low per se: a
        # bar resting inside the bin sits at nearly the same height (1 mm bin floor) and
        # must not be docked for it.
        reward -= 1 * (~info["item_lifted"] & ~info["is_item_above_bin"]).float()

        if self.action_rate_penalty > 0:
            action = common.to_tensor(action, device=self.device)
            if self._prev_action is None or self._prev_action.shape != action.shape:
                self._prev_action = torch.zeros_like(action)
            reward -= self.action_rate_penalty * ((action - self._prev_action) ** 2).sum(-1)
            self._prev_action = action.clone()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20
