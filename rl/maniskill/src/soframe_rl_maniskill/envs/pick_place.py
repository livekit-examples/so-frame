"""SO-101-on-frame pick-and-place task: pick a flat bar, place it in a bin. Adapted from
[Squint](https://github.com/aalmuzairee/squint)'s ``envs/place.py``, with this repo's
frame-mounted ``so101_on_frame`` agent replacing Squint's bare tabletop SO-101. The
lightbox work surface is part of the robot's own URDF (no ``TableSceneBuilder``); a ground
plane far below is a safety catch-all. Observations are vision + proprioception only;
privileged state exists solely behind an explicit "+state" obs_mode (see ``_get_obs_extra``)."""

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

WORK_SURFACE_Z = 0.0

# Reward: a MONOTONIC STAGE LADDER (see compute_dense_reward), deliberately kept small.
#
# Each stage is a fixed rung plus at most 1.0 of bounded shaping, and each rung sits above the
# previous stage's maximum. That spacing is what does the work: progress is strictly monotone,
# a regression drops to a lower rung on its own, and there is nothing to trade off against.
# No penalty terms -- motion limits are enforced structurally instead (the delta action space
# caps per-step speed at the measured real servo rate, and the 3 N.m force limits cap torque
# at the STS3215 stall value), so there is no smoothness or effort term left to tune.
#
#   reach [0,1] < grasped [2,3] < holding [4,5] < released 6 < success 10
#
REWARD_SUCCESS = 10.0     # terminal: bar settled in bin, arm + bar static, gripper clear
RUNG_RELEASED = 6.0       # bar over the bin, released -- must dominate holding
RUNG_HOLDING = 4.0        # bar over the bin, still gripped (leave it by opening the jaw)
RUNG_GRASPED = 2.0        # bar grasped, carried toward the drop point (not yet over the bin)
#                           reach stage base is 0 (shaping only)

# The one shaping term worth keeping past the rungs: how far opening the jaw over the bin pays
# BEFORE contact breaks. Without it the holding rung is a flat plateau and the policy sits on
# it holding the bar, because releasing is a blind leap to the next rung. Stays at 1.0 so the
# most-open still-holding pose (5.0) is still strictly worse than an actual release (6.0).
SHAPE_HOLD_OPEN = 1.0

SHARP = 5.0               # single tanh sharpness (~1/length-scale, m^-1) for every distance term
REACH_XY_ALIGNED = 0.03   # tcp within this xy of the bar is "aligned" and may descend
# Drop point height above the bin RIM (not floor): the jaw can't open at depth in the 84 mm
# interior. Carry here, open, let gravity finish; success still requires the bar settled IN the bin.
GOAL_CLEARANCE = 0.05

# Item and bin spawn in separate regions along the rail, far enough apart that reaching both
# requires sliding. The small bar gets the far region (dead-center in the overhead frame; it
# can vanish behind the arm in the near region); the larger bin stays visible in either.
ITEM_SPAWN_CENTER = (-0.225, -0.70)
BIN_SPAWN_CENTER = (-0.225, -0.30)
SPAWN_HALF_SIZE = 0.1


@dataclass
class PickPlaceRandomizationConfig(RandomizationConfig):
    """Domain randomization config for the pick-and-place task."""

    robot_qpos_noise_std: float = np.deg2rad(5)
    # Flat bar matching the real object, 75 x 25 x 15 mm (footprint diagonal ~79 mm fits the
    # real bin's ~84 mm interior at any yaw; grasped across its 25 mm width).
    item_half_size_x_range: Sequence[float] = (0.0375, 0.0375)
    item_half_size_y_range: Sequence[float] = (0.0125, 0.0125)
    item_half_size_z_range: Sequence[float] = (0.0075, 0.0075)
    # Bin half sizes, matching the real bin: 85 x 85 mm footprint, 35 mm walls, 1 mm thick.
    bin_half_size_x_range: Sequence[float] = (0.0425, 0.0425)
    bin_half_size_y_range: Sequence[float] = (0.0425, 0.0425)
    bin_half_size_z_range: Sequence[float] = (0.0175, 0.0175)

    item_friction_range: Sequence[float] = (0.5, 1.0)
    item_density_range: Sequence[float] = (400, 400)  # ~12 g for the default bar.
    # Fixed colors (purple bar, yellow bin) matched to real captures. Color randomization
    # is supported but costs sample efficiency, so it's opt-in.
    randomize_item_color: bool = False
    randomize_bin_color: bool = False


# 300 steps (30 s at 10 Hz): the slow real-servo-tracking speeds need this much runway.
@register_env("SOFramePickPlaceBin-v1", max_episode_steps=300)
class PickPlaceBin(DualCameraEnv):
    """Pick up a flat bar and place it in a bin. Bar and bin spawn in separate rail regions
    (bar z-rotation also randomized); success = bar settled in the bin, robot clear and static."""

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
        self.domain_randomization_config = PickPlaceRandomizationConfig.resolve(
            domain_randomization_config
        )

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
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        # Safety catch-all far below the lightbox floor for anything that rolls off the panels.
        self.ground = build_ground(self.scene, altitude=-1.0)

        cfg = self.domain_randomization_config
        realistic = cfg.visual_fidelity != "flat"  # PBR materials for "raster"/"raytraced"

        def sample_range(value_range):
            """Per-env uniform sample under domain randomization, midpoint otherwise."""
            if self.domain_randomization:
                return self._batched_episode_rng.uniform(low=value_range[0], high=value_range[1])
            return np.full(self.num_envs, (value_range[0] + value_range[1]) / 2)

        # Work surface panels render near-white (~0.9); a random color too close to that is
        # indistinguishable from the floor at training resolution, so redraw those.
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

        # Purple bar, matched to real captures (median sRGB decoded to linear and scaled by the
        # bar-to-surface brightness ratio, so it holds under sim lighting not the real exposure).
        colors = np.tile([0.28, 0.19, 0.57], (self.num_envs, 1))
        if self.domain_randomization and cfg.randomize_item_color:
            colors = sample_visible_colors()

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

        # Yellow bin, matched to real captures the same way as the bar's albedo above.
        bin_colors = np.ones((self.num_envs, 3)) * [0.90, 0.47, 0.01]
        if self.domain_randomization and cfg.randomize_bin_color:
            bin_colors = sample_visible_colors()
        bin_colors = np.concatenate([bin_colors, np.ones((self.num_envs, 1))], axis=-1)
        # Real bin walls are 1 mm, too thin for reliable contact at the 10 ms step, so walls
        # keep a 1 mm visual but get a thicker collision box extended outward (inner faces
        # aligned), so the interior clearance stays true to the real bin.
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

            # Collision walls sit outward of the visual walls, inner faces aligned.
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

            # Separate non-overlapping regions, so offsets are independent uniform samples.
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
        """The actor's proprio: noisy measured qpos (7) then the controller's target qpos (7).

        This is the deploy contract -- 14 values the real rig can always produce, in this
        order. Anything added here changes what every deployed checkpoint expects, so keep it
        to quantities the robot actually measures or commands. (A previous version inserted
        action-delay fields here, which silently broke deploy; see nets/README.md.)
        """
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
        """Privileged ground-truth state, only populated for an explicit "+state" obs_mode
        (empty under the default training obs_mode). For offline eval/debugging."""
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
        # In the bin = the bar's lowest corner is within tolerance of the bin floor. A tilted
        # bar leaning on a wall counts; one bridging flat across the rim, or still falling, does not.
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
        """Monotonic stage ladder, no penalties. Stages are mutually exclusive, and each rung
        sits above the previous stage's maximum, so progress is monotone and a regression falls
        to a lower rung without needing an explicit anti-regression term."""
        tcp = self.agent.tcp_pos
        item_p = self.item.pose.p

        # Drop point: above the bin rim (GOAL_CLEARANCE). The carry term shapes toward it.
        goal_xyz = self.bin.pose.p.clone()
        goal_xyz[..., 2] = (
            WORK_SURFACE_Z + self.bin_dimensions[:, 2] * 2 + self.item_half_heights + GOAL_CLEARANCE
        )

        # Stage 0 [0, 1]: top-down reach. xy alignment first; the z term only pays once aligned,
        # so the tool descends from ABOVE rather than scooping from the side (which wrecked grasps).
        reach_d_xy = torch.linalg.norm(tcp[:, :2] - item_p[:, :2], axis=1)
        reach_d_z = torch.abs(tcp[:, 2] - item_p[:, 2])
        aligned_xy = reach_d_xy < REACH_XY_ALIGNED
        reward = 0.5 * (1 - torch.tanh(SHARP * reach_d_xy)) + torch.where(
            aligned_xy,
            0.5 * (1 - torch.tanh(SHARP * reach_d_z)),
            torch.zeros_like(reach_d_z),
        )

        # Stage masks. is_item_above_bin is horizontal-only; holding vs released splits on contact.
        above_bin = info["is_item_above_bin"]
        holding = info["robot_touching_item"]
        grasped_only = info["is_item_grasped"] & ~above_bin
        holding_above = above_bin & holding
        released_above = above_bin & ~holding

        # Stage 1 [2, 3]: grasped, carrying toward the drop point (single 3D distance; the drop
        # point's 5 cm rim clearance already shapes the over-the-top approach).
        item_to_goal = torch.linalg.norm(goal_xyz - item_p, axis=1)
        carry = RUNG_GRASPED + (1 - torch.tanh(SHARP * item_to_goal))
        reward = torch.where(grasped_only, carry, reward)

        # Stage 2 [4, 5]: holding over the bin -- rises with how far the jaw is opened, so
        # unclamping is a continuous climb instead of a blind jump to the released rung.
        # Openness pays HERE only, never while carrying, so opening early (and dropping the bar
        # short of the bin) is still worth strictly less than carrying on.
        openness = self._gripper_qpos_openness()
        reward = torch.where(holding_above, RUNG_HOLDING + SHAPE_HOLD_OPEN * openness, reward)

        # Stage 3 [6]: released over the bin. Flat -- success is what pays for a clean settle.
        reward = torch.where(released_above, torch.full_like(reward, RUNG_RELEASED), reward)

        # Terminal success (bar settled in the bin, arm + bar static, gripper clear).
        reward = torch.where(info["success"], torch.full_like(reward, REWARD_SUCCESS), reward)

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / REWARD_SUCCESS
