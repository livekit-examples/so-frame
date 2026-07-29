"""SO-101-on-frame pick-and-place task: pick a cube, place it in a bin. Adapted from
[Squint](https://github.com/aalmuzairee/squint)'s ``envs/place.py``, with this repo's
frame-mounted ``so101_on_frame`` agent replacing Squint's bare tabletop SO-101. The
lightbox work surface is part of the robot's own URDF (no ``TableSceneBuilder``); a ground
plane far below is a safety catch-all. Observations are vision + proprioception only;
privileged state exists solely behind an explicit "+state" obs_mode (see ``_get_obs_extra``)."""

import math
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
from ..robot.so101_on_frame import SO101OnFrame, _REPO_ROOT
from .. import config

# CAD meshes for the task objects, exported from the 3MF sources by
# simulation/assets/objects/convert.py (which also asserts the dimensions in config.py).
_OBJECTS_ROOT = _REPO_ROOT / config.OBJECTS_ROOT
CUBE_MESH = _OBJECTS_ROOT / "cube.obj"
BIN_MESH = _OBJECTS_ROOT / "bin.obj"
for _mesh in (CUBE_MESH, BIN_MESH):
    assert _mesh.exists(), (
        f"task mesh not found at {_mesh}; run simulation/assets/objects/convert.py"
    )

# Every tunable constant lives in ../config.py; imported by name here so the reward reads
# cleanly. See that file for the reasoning behind the ladder spacing and the motion limits.
from ..config import (  # noqa: E402
    GOAL_CLEARANCE,
    REACH_XY_ALIGNED,
    REACH_Z_ALIGNED,
    REWARD_SUCCESS,
    RUNG_GRASPED,
    RUNG_HOLDING,
    RUNG_RELEASED,
    SHAPE_HOLD_OPEN,
    SHAPE_REACH_CLOSE,
    SHARP,
    WORK_SURFACE_Z,
)


@dataclass
class PickPlaceRandomizationConfig(RandomizationConfig):
    """Domain randomization config for the pick-and-place task.

    Randomization *ranges* live here rather than in config.py: they are per-run and
    CLI-overridable, whereas config.py holds the fixed physical constants of the rig."""

    robot_qpos_noise_std: float = np.deg2rad(5)

    # Object SIZE is no longer randomized: both the cube and the bin come from fixed CAD meshes
    # (simulation/assets/objects/), so their dimensions live in config.py. Only material
    # properties and colour vary per episode.
    item_friction_range: Sequence[float] = (0.5, 1.0)
    item_density_range: Sequence[float] = (400, 400)  # ~3.2 g for the 20 mm cube.
    # Fixed colors (purple cube, yellow bin) matched to real captures. Color randomization
    # is supported but costs sample efficiency, so it's opt-in.
    randomize_item_color: bool = False
    randomize_bin_color: bool = False


# 300 steps (30 s at 10 Hz): the slow real-servo-tracking speeds need this much runway.
@register_env("SOFramePickPlaceBin-v1", max_episode_steps=config.EPISODE_HORIZON)
class PickPlaceBin(DualCameraEnv):
    """Pick up a 20 mm cube and place it in a bin, both from the CAD meshes in
    simulation/assets/objects/.

    Cube and bin spawn anywhere in ONE workspace zone (bin first, then the cube at least
    config.SPAWN_MIN_GAP clear of it), each with a random z-rotation. Success = cube settled in
    the bin, robot clear and static."""

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
        workspace_center=config.WORKSPACE_CENTER,
        workspace_half=config.WORKSPACE_HALF,
        **kwargs,
    ):
        self.domain_randomization_config = PickPlaceRandomizationConfig.resolve(
            domain_randomization_config
        )

        self.workspace_center = workspace_center
        self.workspace_half = workspace_half

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
        # Deliberately beyond config.SENSOR_FAR so the policy cameras never render it; see
        # config.GROUND_ALTITUDE for why that matters.
        self.ground = build_ground(self.scene, altitude=config.GROUND_ALTITUDE)

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

        frictions = sample_range(cfg.item_friction_range)
        densities = sample_range(cfg.item_density_range)

        # Cube colour from config.COLOR_CUBE (linear base colour).
        colors = np.tile(config.COLOR_CUBE, (self.num_envs, 1))
        if self.domain_randomization and cfg.randomize_item_color:
            colors = sample_visible_colors()
        colors = np.concatenate([colors, np.ones((self.num_envs, 1))], axis=-1)

        # Fixed geometry from the CAD meshes, so these are scalars broadcast per env rather than
        # per-env samples. Kept as tensors because evaluate() and the reward index them by env.
        ones = torch.ones(self.num_envs, device=self.device)
        self.item_half_heights = ones * config.CUBE_HALF
        self.item_dimensions = torch.stack([ones * config.CUBE_HALF] * 3, dim=-1)
        self.item_frictions = common.to_tensor(frictions, device=self.device)
        self.item_densities = common.to_tensor(densities, device=self.device)

        items = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            friction = frictions[i]
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction, dynamic_friction=friction, restitution=0,
            )
            # Exact box collision: the mesh IS a 20 mm cube, so a box is identical and cheaper
            # than a convex hull, and gives PhysX flat faces to rest on.
            builder.add_box_collision(
                half_size=[config.CUBE_HALF] * 3, material=material, density=densities[i]
            )
            item_material = sapien.render.RenderMaterial(base_color=colors[i])
            if realistic:
                item_material.set_roughness(0.6)
                item_material.set_metallic(0.0)
            # Visual from the CAD mesh. Its base is at z=0 while the collision box is centred on
            # the origin, so the mesh is lowered by a half-height to line the two up.
            builder.add_visual_from_file(
                filename=str(CUBE_MESH),
                pose=sapien.Pose(p=[0, 0, -config.CUBE_HALF]),
                material=item_material,
            )
            builder.initial_pose = sapien.Pose(p=[0.2, 0, config.CUBE_HALF])
            builder.set_scene_idxs([i])
            item = builder.build(name=f"item-{i}")
            items.append(item)
            self.remove_from_state_dict_registry(item)

        self.item = Actor.merge(items, name="item")
        self.add_to_state_dict_registry(self.item)

        # Bin colour from config.COLOR_BIN.
        bin_colors = np.tile(config.COLOR_BIN, (self.num_envs, 1))
        if self.domain_randomization and cfg.randomize_bin_color:
            bin_colors = sample_visible_colors()
        bin_colors = np.concatenate([bin_colors, np.ones((self.num_envs, 1))], axis=-1)

        self.bin_thickness = config.BIN_FLOOR_THICKNESS
        # [interior_half_x, interior_half_y, rim_height]. The z entry is the FULL rim height
        # above the work surface (the mesh base sits on it), not a half-extent.
        self.bin_dimensions = torch.stack([
            ones * config.BIN_INTERIOR_HALF,
            ones * config.BIN_INTERIOR_HALF,
            ones * config.BIN_HEIGHT,
        ], dim=-1)

        bins = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()

            bin_material = sapien.render.RenderMaterial(base_color=bin_colors[i])
            if realistic:
                bin_material.set_roughness(0.55)
                bin_material.set_metallic(0.0)

            # Visual: the CAD mesh, filleted corners and true 2 mm walls. Its base is at z=0 and
            # the actor origin is the base too, so no offset.
            builder.add_visual_from_file(filename=str(BIN_MESH), material=bin_material)

            # Collision: floor slab + 4 walls. The mesh is concave, and a convex hull would seal
            # the opening, so collision is built from boxes instead (convex decomposition would
            # also work but costs build time across 1024 envs for no benefit on a box tray).
            floor_h = config.BIN_FLOOR_THICKNESS / 2
            builder.add_box_collision(
                pose=sapien.Pose([0, 0, floor_h]),
                half_size=[config.BIN_FOOTPRINT_HALF, config.BIN_FOOTPRINT_HALF, floor_h],
            )
            # Walls span floor top -> rim, inner faces on the true interior, thickened outward.
            wall_t = config.BIN_WALL_COLLISION_THICKNESS
            wall_half_h = (config.BIN_HEIGHT - config.BIN_FLOOR_THICKNESS) / 2
            wall_cz = config.BIN_FLOOR_THICKNESS + wall_half_h
            wall_off = config.BIN_INTERIOR_HALF + wall_t / 2
            for j in (-1, 1):
                builder.add_box_collision(
                    pose=sapien.Pose([0, j * wall_off, wall_cz]),
                    half_size=[wall_off, wall_t / 2, wall_half_h],
                )
                builder.add_box_collision(
                    pose=sapien.Pose([j * wall_off, 0, wall_cz]),
                    half_size=[wall_t / 2, wall_off, wall_half_h],
                )

            builder.initial_pose = sapien.Pose(p=[-0.2, 0, 0])
            builder.set_scene_idxs([i])
            bin_actor = builder.build(name=f"bin-{i}")
            bins.append(bin_actor)
            self.remove_from_state_dict_registry(bin_actor)

        self.bin = Actor.merge(bins, name="bin")
        self.add_to_state_dict_registry(self.bin)
        self.bin_radius = ones * config.BIN_FOOTPRINT_HALF * math.sqrt(2)

        if self.apply_greenscreen:
            self.remove_object_from_greenscreen(self.agent.robot)
            self.remove_object_from_greenscreen(self.item)
            self.remove_object_from_greenscreen(self.bin)

        self.rest_qpos = common.to_tensor(
            SO101OnFrame.keyframes["rest"].qpos.tolist(), device=self.device
        )

        # One pass: colour scheme always, PBR relief maps only when realistic.
        self.agent.apply_materials(realistic=realistic)

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

            # -- Spawn: one zone for both objects, bin placed first --------------------------
            #
            # The bin goes anywhere in the workspace; the cube then goes anywhere in the same
            # workspace that clears the bin by at least SPAWN_MIN_GAP. So the pair can appear in
            # any relative arrangement, and the policy cannot learn "bar is always far, bin is
            # always near" from the old split regions.
            #
            # Each object samples from the zone inset by its own footprint circumradius plus
            # SPAWN_PADDING, so it lands fully inside the reachable area and away from the frame
            # edge. The bin is inset much further than the cube because it is far larger.
            origin = self.agent.robot.pose.p[env_idx, :2] + torch.tensor(
                [self.workspace_center[0], self.workspace_center[1]]
            )
            zone_half = torch.tensor([self.workspace_half[0], self.workspace_half[1]])

            # Yaw is sampled up front so the edge inset can use each object's ACTUAL rotated
            # footprint. A square of half-size h at yaw t spans h*(|cos t| + |sin t|) per axis,
            # between h and h*sqrt(2). Insetting by the worst case instead would cost almost all
            # of the bin's x freedom: the zone is only 200 mm wide and the bin's diagonal is 141.
            item_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            bin_qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)

            def aabb_half(quats, half_size):
                """Axis-aligned half-extent of a square footprint at these yaws."""
                # z-only rotation: yaw = 2*atan2(qz, qw)
                yaw = 2.0 * torch.atan2(quats[:, 3], quats[:, 0])
                spread = yaw.cos().abs() + yaw.sin().abs()
                return (half_size * spread).unsqueeze(-1).expand(-1, 2)

            def sample_xy(centre, footprint):
                """Uniform in the zone, inset so the rotated footprint stays inside it."""
                inset = (zone_half - footprint - config.SPAWN_PADDING).clamp_min(0.0)
                return centre + (torch.rand(footprint.shape[0], 2) * 2 - 1) * inset

            bin_fp = aabb_half(bin_qs, config.BIN_FOOTPRINT_HALF)
            cube_fp = aabb_half(item_qs, config.CUBE_HALF)

            # The pairwise test stays on circumradii: conservative is the right side to err on
            # for "do not overlap", and it is independent of both yaws.
            bin_r = config.BIN_FOOTPRINT_HALF * math.sqrt(2)
            cube_r = config.CUBE_HALF * math.sqrt(2)
            min_centre_dist = bin_r + cube_r + config.SPAWN_MIN_GAP

            bin_xy = sample_xy(origin, bin_fp)
            cube_xy = sample_xy(origin, cube_fp)

            # Rejection-sample only the cubes that are too close to their bin.
            for _ in range(config.SPAWN_MAX_ATTEMPTS):
                too_close = torch.linalg.norm(cube_xy - bin_xy, dim=-1) < min_centre_dist
                if not too_close.any():
                    break
                cube_xy[too_close] = sample_xy(origin[too_close], cube_fp[too_close])
            else:
                # Deterministic fallback so a spawn is never left overlapping: push the stragglers
                # to whichever end of the long (y) axis is further from their bin, at the bin's x.
                too_close = torch.linalg.norm(cube_xy - bin_xy, dim=-1) < min_centre_dist
                if too_close.any():
                    limit = (zone_half[1] - cube_r - config.SPAWN_PADDING).clamp_min(0.0)
                    far_end = torch.where(bin_xy[:, 1] > origin[:, 1], -limit, limit)
                    cube_xy[too_close] = torch.stack(
                        [bin_xy[too_close, 0], (origin[:, 1] + far_end)[too_close]], dim=-1
                    )

            # The cube's collision box is centred on the actor origin, so it rests a half-height
            # above the surface. The bin's mesh and collision are both based at its origin.
            # MUST reuse item_qs / bin_qs, the yaws the edge inset above was computed from.
            # Drawing fresh ones here would let an object rotate into the zone edge it was
            # placed to clear.
            item_xyz = torch.zeros((b, 3))
            item_xyz[:, :2] = cube_xy
            item_xyz[:, 2] = WORK_SURFACE_Z + self.item_half_heights[env_idx]
            self.item.set_pose(Pose.create_from_pq(item_xyz, item_qs))

            bin_xyz = torch.zeros((b, 3))
            bin_xyz[:, :2] = bin_xy
            bin_xyz[:, 2] = WORK_SURFACE_Z
            self.bin.set_pose(Pose.create_from_pq(bin_xyz, bin_qs))

            # Goal marker: resting height of the cube on the bin's interior floor.
            goal_xyz = bin_xyz.clone()
            goal_xyz[:, 2] = (
                WORK_SURFACE_Z + config.BIN_FLOOR_THICKNESS + self.item_half_heights[env_idx]
            )
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_agent(self):
        """The actor's proprio: noisy measured qpos (7) then the controller's target qpos (7).

        This is the deploy contract -- 14 values the real rig can always produce, in this
        order. Anything added here changes what every deployed checkpoint expects, so keep it
        to quantities the robot actually measures or commands. (A previous version inserted
        action-delay fields here, which silently broke deploy; see policy/README.md.)
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
        bin_pos[:, 2] = WORK_SURFACE_Z + config.BIN_FLOOR_THICKNESS + self.item_half_heights

        # Over the OPENING, not the outer footprint: the walls are part of the footprint but a
        # cube above a wall is not above the hole.
        offset = item_pos - bin_pos
        inside_x = torch.abs(offset[:, 0]) < config.BIN_INTERIOR_HALF
        inside_y = torch.abs(offset[:, 1]) < config.BIN_INTERIOR_HALF
        is_item_above_bin = inside_x & inside_y
        # In the bin = the cube's lowest corner is within tolerance of the bin's interior floor.
        # A cube tilted against a wall counts; one bridging the rim, or still falling, does not.
        item_rot = self.item.pose.to_transformation_matrix()[..., :3, :3]
        item_lowest = item_pos[:, 2] - (item_rot[..., 2, :].abs() * self.item_dimensions).sum(-1)
        touches_bottom = item_lowest <= WORK_SURFACE_Z + config.BIN_FLOOR_THICKNESS + 0.005
        is_item_in_bin = is_item_above_bin & touches_bottom

        item_lifted = self.item.pose.p[..., -1] >= (WORK_SURFACE_Z + self.item_half_heights + 1e-3)

        item_vel = torch.linalg.norm(self.item.linear_velocity, axis=-1)
        is_item_static = item_vel <= 2e-2
        is_robot_static = self.agent.is_static()

        # Two pairwise contact queries per object, fetched once and shared. Asking the agent for
        # `is_grasping(item)` and `is_touching(item)` separately ran the same two queries twice.
        item_forces = self.agent.jaw_contact_forces(self.item)
        bin_forces = self.agent.jaw_contact_forces(self.bin)
        is_item_grasped = self.agent.is_grasping(self.item, forces=item_forces)
        robot_touching_item = self.agent.is_touching(self.item, forces=item_forces)
        robot_touching_bin = self.agent.is_touching(self.bin, forces=bin_forces)

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
            WORK_SURFACE_Z + config.BIN_HEIGHT + self.item_half_heights + GOAL_CLEARANCE
        )

        # How far the jaw is open, in [0, 1]. Used at both ends of the pick: closing pays in the
        # reach stage, opening pays in the holding stage.
        openness = self._gripper_qpos_openness()

        # Stage 0 [0, 1.5]: top-down reach, then clamp. xy alignment first; the z term only pays
        # once aligned, so the tool descends from ABOVE rather than scooping from the side (which
        # wrecked grasps).
        reach_d_xy = torch.linalg.norm(tcp[:, :2] - item_p[:, :2], axis=1)
        reach_d_z = torch.abs(tcp[:, 2] - item_p[:, 2])
        aligned_xy = reach_d_xy < REACH_XY_ALIGNED
        reward = 0.5 * (1 - torch.tanh(SHARP * reach_d_xy)) + torch.where(
            aligned_xy,
            0.5 * (1 - torch.tanh(SHARP * reach_d_z)),
            torch.zeros_like(reach_d_z),
        )
        # The mirror of the openness term in stage 2: once the tool is ON the cube, squeezing the
        # jaw shut is a continuous climb toward the grasped rung instead of a blind jump from a
        # flat plateau. Gated on BOTH axes being aligned, so closedness pays nothing on the way
        # in -- a jaw that shuts before it surrounds the cube cannot grasp it.
        on_item = aligned_xy & (reach_d_z < REACH_Z_ALIGNED)
        reward = reward + torch.where(
            on_item, SHAPE_REACH_CLOSE * (1 - openness), torch.zeros_like(openness)
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
        reward = torch.where(holding_above, RUNG_HOLDING + SHAPE_HOLD_OPEN * openness, reward)

        # Stage 3 [6]: released over the bin. Flat -- success is what pays for a clean settle.
        reward = torch.where(released_above, torch.full_like(reward, RUNG_RELEASED), reward)

        # Terminal success (bar settled in the bin, arm + bar static, gripper clear).
        reward = torch.where(info["success"], torch.full_like(reward, REWARD_SUCCESS), reward)

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / REWARD_SUCCESS
