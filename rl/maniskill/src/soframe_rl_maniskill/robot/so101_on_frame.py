"""SO-101-on-frame robot definition for ManiSkill3, modeled on
[Squint](https://github.com/aalmuzairee/squint)'s ``envs/robot/so101.py`` (a bare tabletop
SO-101). Loads ``simulation/urdf/so101_on_frame.urdf`` as a ManiSkill ``BaseAgent``, adds
the frame's rail joint (``dof_slider``) as a 7th DOF, and computes the ``tcp`` point from a
fixed offset off ``gripper_link`` (this URDF has no fingertip links)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent, RenderTexture2D

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *  # noqa: F403 (PDJointPosControllerConfig, etc.)
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

_REPO_ROOT = Path(__file__).resolve().parents[5]
SO101_ON_FRAME_URDF = (
    _REPO_ROOT / "simulation" / "urdf" / "so101_on_frame.urdf"
)
assert SO101_ON_FRAME_URDF.exists(), f"SO-101-on-frame URDF not found at {SO101_ON_FRAME_URDF}"

# Grasp point between the finger pads, in gripper_link's frame. Matches the `grasp_site`
# MJCF site added for rl/mjlab (simulation/mjcf/so101_on_frame.xml).
GRASP_SITE_OFFSET = sapien.Pose(p=[0.012, 0.0, -0.07])

# Per-joint delta-position step limits (rad for revolute joints, m for the rail), keyed by
# name so joint order from the URDF loader doesn't matter. Caps track measured real servo
# speeds (arm ~0.5 rad/s, rail ~7 cm/s) so the motion profile transfers to the real rig.
_JOINT_DELTA_LIMITS = {
    "dof_slider": 0.007,
    "shoulder_pan": 0.05,
    "shoulder_lift": 0.05,
    "elbow_flex": 0.05,
    "wrist_flex": 0.05,
    "wrist_roll": 0.05,
    "gripper": 0.2,
}

# Per-joint force limits: N*m for the six STS3215 revolute servos, N for the rail. Capping
# at the STS3215's ~3 N*m stall torque keeps the sim arm from pressing harder than the real
# servo can, closing the sim2real gap. Matches simulation/mjcf/sts3215.xml (forcerange -3 3).
# The rail is a separate belt-driven actuator, left at a functional value.
STS3215_STALL_TORQUE = 3.0  # N*m (30 kg*cm @ 12V)
_JOINT_FORCE_LIMITS = {
    "dof_slider": 100.0,
    "shoulder_pan": STS3215_STALL_TORQUE,
    "shoulder_lift": STS3215_STALL_TORQUE,
    "elbow_flex": STS3215_STALL_TORQUE,
    "wrist_flex": STS3215_STALL_TORQUE,
    "wrist_roll": STS3215_STALL_TORQUE,
    "gripper": STS3215_STALL_TORQUE,
}

# Multiplier on the arm/rail delta limits (NOT the gripper) for arm-speed ablations.
ARM_SPEED_SCALE = 1.0

# PBR texture sets for `apply_realistic_materials` (see `RandomizationConfig.visual_fidelity`).
_TEXTURES_ROOT = _REPO_ROOT / "simulation" / "assets" / "textures"

# The URDF only defines flat `<color rgba>` materials and SAPIEN's loader keeps only the
# resolved rgba, so `apply_realistic_materials` matches each part's baked base_color back to
# these known URDF colors to pick a texture set.
_ALUMINUM_BASE_COLORS = [
    (0.4, 0.4, 0.4),                 # extrusion_gray (2020 extrusion bars)
    (0.749020, 0.749020, 0.749020),  # nuts/screws/wheels/gantry plate
    (0.960784, 0.960784, 0.964706),  # metal servo horn links
]
_WHITE_PLASTIC_BASE_COLORS = [
    (1.0, 1.0, 1.0),                 # 3d_printed, camera holders/mounts
    (0.901961, 0.901961, 0.901961),  # light bracket parts
]

# The lightbox's diffusing side panels: large matte surfaces, not small 3D-printed parts,
# though they share a base color with some (see `apply_realistic_materials`).
_LIGHTBOX_PANEL_LINKS = {"part_1", "part_1_1", "part_1_2", "part_1_3"}

_ORANGE_PLASTIC_BASE_COLORS = [
    (0.94, 0.49, 0.06),  # gripper_orange, accent_orange
]


def _base_color_matches(color, targets, tol: float = 0.02) -> bool:
    return any(
        abs(color[0] - t[0]) < tol and abs(color[1] - t[1]) < tol and abs(color[2] - t[2]) < tol
        for t in targets
    )


def _load_texture_set(name: str) -> dict:
    """Load one texture set from ``simulation/assets/textures/<name>/`` (data maps use srgb=False)."""
    root = _TEXTURES_ROOT / name
    textures = {"base_color": RenderTexture2D(str(root / "base_color.jpg"), srgb=True)}
    for key, filename in (("roughness", "roughness.jpg"), ("metallic", "metalness.jpg"), ("normal", "normal.jpg")):
        path = root / filename
        if path.exists():
            textures[key] = RenderTexture2D(str(path), srgb=False)
    return textures


@register_agent()
class SO101OnFrame(BaseAgent):
    uid = "so101_on_frame"

    urdf_path = str(SO101_ON_FRAME_URDF)
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            gripper_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            moving_jaw_so101_v1_link=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    # qpos order matches `self.robot.active_joints`:
    #   [dof_slider, shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, 0.0, -0.5, 0.8, 0.6, 0.0, 1.2]),
            pose=sapien.Pose(),
        ),
        zero=Keyframe(
            qpos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            pose=sapien.Pose(),
        ),
    )

    @property
    def _controller_configs(self):
        joint_names = self.joint_names
        # Scale arm/rail limits by ARM_SPEED_SCALE (gripper unscaled).
        def _limit(name):
            lim = _JOINT_DELTA_LIMITS[name]
            return lim if name == "gripper" else lim * ARM_SPEED_SCALE
        delta_lower = [-_limit(name) for name in joint_names]
        delta_upper = [_limit(name) for name in joint_names]
        force_limits = [_JOINT_FORCE_LIMITS[name] for name in joint_names]

        pd_joint_pos = PDJointPosControllerConfig(  # noqa: F405
            joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=force_limits,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(  # noqa: F405
            joint_names,
            delta_lower,
            delta_upper,
            stiffness=[1e3] * len(joint_names),
            damping=[1e2] * len(joint_names),
            force_limit=force_limits,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return {k: copy.deepcopy(v) for k, v in controller_configs.items()}

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.joint_names = [joint.name for joint in self.robot.active_joints]
        self.finger1_link = self.robot.links_map["gripper_link"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1_link"]
        # Calibrated camera links baked into the URDF (see simulation/urdf/README.md).
        self.wrist_camera_link = self.robot.links_map["frame_wrist_camera"]
        self.overhead_camera_link = self.robot.links_map["frame_overhead_camera"]

    @property
    def tcp_pos(self):
        """Grasp point between the finger pads, offset from gripper_link (no fingertip links in this URDF)."""
        return (self.finger1_link.pose * GRASP_SITE_OFFSET).p

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_touching(self, object: Actor):
        """Check if either gripper jaw is touching `object`."""
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        return torch.logical_or(lforce >= 1e-2, rforce >= 1e-2)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if both gripper jaws are pressing into `object` from opposing sides."""
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.15):
        """Check if the arm (excluding the gripper joint) is roughly still."""
        arm_idx = [i for i, name in enumerate(self.joint_names) if name != "gripper"]
        qvel = self.robot.get_qvel()[:, arm_idx]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    def apply_realistic_materials(self):
        """Wire real PBR textures onto the flat-color URDF materials for the "raster"/
        "raytraced" fidelity modes; untextured parts keep their flat color. The lightbox
        panels (``_LIGHTBOX_PANEL_LINKS``) get a plain matte material rather than the
        plastic grain, which would catch light unevenly across their large flat surfaces."""
        aluminum = _load_texture_set("aluminum")
        plastic_white = _load_texture_set("plastic_white")
        plastic_orange = _load_texture_set("plastic_orange")

        for link in self.robot.links:
            is_lightbox_panel = link.name in _LIGHTBOX_PANEL_LINKS
            for obj in link._objs:
                render_body_component = obj.entity.find_component_by_type(RenderBodyComponent)
                if render_body_component is None:
                    continue
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        if is_lightbox_panel:
                            part.material.set_roughness(0.9)
                            part.material.set_metallic(0.0)
                            continue

                        color = part.material.get_base_color()
                        if _base_color_matches(color, _ALUMINUM_BASE_COLORS):
                            textures = aluminum
                        elif _base_color_matches(color, _WHITE_PLASTIC_BASE_COLORS):
                            textures = plastic_white
                        elif _base_color_matches(color, _ORANGE_PLASTIC_BASE_COLORS):
                            textures = plastic_orange
                        else:
                            continue

                        part.material.set_base_color_texture(textures["base_color"])
                        if "roughness" in textures:
                            part.material.set_roughness_texture(textures["roughness"])
                        if "metallic" in textures:
                            part.material.set_metallic_texture(textures["metallic"])
                        if "normal" in textures:
                            part.material.set_normal_texture(textures["normal"])
