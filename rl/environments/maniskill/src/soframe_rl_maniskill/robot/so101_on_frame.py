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

# rl/environments/maniskill/src/soframe_rl_maniskill/robot/... -> repo root is parents[6].
_REPO_ROOT = Path(__file__).resolve().parents[6]
SO101_ON_FRAME_URDF = (
    _REPO_ROOT / "simulation" / "urdf" / "so101_on_frame.urdf"
)
assert SO101_ON_FRAME_URDF.exists(), f"SO-101-on-frame URDF not found at {SO101_ON_FRAME_URDF}"

# Imported as a module (not `from config import ...`) so a runtime override of e.g.
# config.ARM_SPEED_SCALE is picked up when the controller is built.
from .. import config

GRASP_SITE_OFFSET = config.GRASP_SITE_OFFSET

# PBR texture sets for `apply_materials` (see `RandomizationConfig.visual_fidelity`).
_TEXTURES_ROOT = _REPO_ROOT / "simulation" / "assets" / "textures"

# The URDF only defines flat `<color rgba>` materials and SAPIEN's loader keeps only the resolved
# rgba, so `apply_materials` matches each part's baked base_color against these known URDF colours
# to pick a texture set.
_ALUMINUM_BASE_COLORS = [
    (0.4, 0.4, 0.4),                 # extrusion_gray (2020 extrusion bars)
    (0.749020, 0.749020, 0.749020),  # nuts/screws/wheels/gantry plate
    (0.960784, 0.960784, 0.964706),  # metal servo horn links
]
_WHITE_PLASTIC_BASE_COLORS = [
    (1.0, 1.0, 1.0),                 # 3d_printed, camera holders/mounts
    (0.901961, 0.901961, 0.901961),  # light bracket parts
]

# The lightbox's diffusing side panels: large matte surfaces, not small 3D-printed parts, though
# they share a base colour with some (see `apply_materials`).
_LIGHTBOX_PANEL_LINKS = {"part_1", "part_1_1", "part_1_2", "part_1_3"}

# PBR relief maps per colour group. Only roughness/metallic/normal come from these; albedo comes
# from the colour scheme.
_GROUP_TEXTURE_SET = {
    "extrusion": "aluminum",
    "slider": "aluminum",
    "arm": "plastic_white",
    "gripper": "plastic_white",
}

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
    # The rest pose is config.REST_QPOS; the deploy-side reset ramps to the same pose.
    _JOINT_ORDER = ("dof_slider", "shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll", "gripper")
    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([config.REST_QPOS[j] for j in _JOINT_ORDER]),
            pose=sapien.Pose(),
        ),
        zero=Keyframe(
            qpos=np.zeros(len(_JOINT_ORDER)),
            pose=sapien.Pose(),
        ),
    )

    @property
    def _controller_configs(self):
        joint_names = self.joint_names
        # Scale arm/rail limits by config.ARM_SPEED_SCALE (gripper unscaled). MUST be read here
        # rather than at import, so --arm_speed_scale can override it before the env is built.
        def _limit(name):
            lim = config.JOINT_DELTA_LIMITS[name]
            return lim if name == "gripper" else lim * config.ARM_SPEED_SCALE
        delta_lower = [-_limit(name) for name in joint_names]
        delta_upper = [_limit(name) for name in joint_names]
        force_limits = [config.JOINT_FORCE_LIMITS[name] for name in joint_names]

        pd_joint_pos = PDJointPosControllerConfig(  # noqa: F405
            joint_names,
            lower=None,
            upper=None,
            stiffness=config.JOINT_STIFFNESS,
            damping=config.JOINT_DAMPING,
            force_limit=force_limits,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(  # noqa: F405
            joint_names,
            delta_lower,
            delta_upper,
            stiffness=[config.JOINT_STIFFNESS] * len(joint_names),
            damping=[config.JOINT_DAMPING] * len(joint_names),
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

    def jaw_contact_forces(self, object: Actor):
        """Per-jaw contact force vectors against `object`, as (fixed_jaw, moving_jaw).

        The two pairwise queries are the expensive part of every contact test below, so a caller
        needing more than one flag for the same object should fetch this once and pass it in rather
        than calling `is_touching` and `is_grasping` back to back."""
        return (
            self.scene.get_pairwise_contact_forces(self.finger1_link, object),
            self.scene.get_pairwise_contact_forces(self.finger2_link, object),
        )

    def is_touching(self, object: Actor, forces=None):
        """Check if either gripper jaw is touching `object`.

        `forces` is an already-fetched `jaw_contact_forces(object)` pair; omit it to fetch."""
        if forces is None:
            forces = self.jaw_contact_forces(object)
        l_contact_forces, r_contact_forces = forces
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        return torch.logical_or(lforce >= 1e-2, rforce >= 1e-2)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110, forces=None):
        """Check if both gripper jaws are pressing into `object` from opposing sides.

        `forces` is an already-fetched `jaw_contact_forces(object)` pair; omit it to fetch."""
        if forces is None:
            forces = self.jaw_contact_forces(object)
        l_contact_forces, r_contact_forces = forces
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

    @staticmethod
    def _color_group(link_name: str):
        """Which colour group a link belongs to, or None to keep its URDF colour."""
        for group, patterns in config.COLOR_GROUPS:
            if any(p in link_name for p in patterns):
                return group
        return None

    def apply_materials(self, realistic: bool):
        """Set every render material in ONE pass: the colour scheme plus, when `realistic`, the PBR
        relief maps.

        Must stay a single pass: each material mutation makes that part's material unique, so
        touching a grouped part twice doubles the descriptor sets and exhausts the Vulkan descriptor
        pool under "raster" once both the train and eval env pools exist.

        Grouped links (config.COLOR_GROUPS) get a FLAT scheme colour and, when realistic, only the
        roughness / metallic / normal maps. No albedo map: SAPIEN's base-colour texture replaces
        base_color rather than modulating it, so it would overwrite the scheme colour.

        Ungrouped links keep their baked URDF colour and get the full texture set matched from it.
        That covers the lightbox panels (the work surface) and the camera holders.
        """
        textures = {}
        if realistic:
            for name in ("aluminum", "plastic_white", "plastic_orange"):
                textures[name] = _load_texture_set(name)

        def relief_only(material, texture_set):
            """Surface detail without touching albedo, so the flat scheme colour survives."""
            if "roughness" in texture_set:
                material.set_roughness_texture(texture_set["roughness"])
            if "metallic" in texture_set:
                material.set_metallic_texture(texture_set["metallic"])
            if "normal" in texture_set:
                material.set_normal_texture(texture_set["normal"])

        for link in self.robot.links:
            is_lightbox_panel = link.name in _LIGHTBOX_PANEL_LINKS
            group = self._color_group(link.name)
            for obj in link._objs:
                render_body_component = obj.entity.find_component_by_type(RenderBodyComponent)
                if render_body_component is None:
                    continue
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
                        material = part.material

                        if is_lightbox_panel:
                            # Large flat diffusers: plain matte, no grain to catch light unevenly.
                            if realistic:
                                material.set_roughness(0.9)
                                material.set_metallic(0.0)
                            continue

                        if group is not None:
                            # A grouped link is printed structure PLUS motor hardware. Only the
                            # structure takes the scheme colour; the motor keeps the URDF's own
                            # colour and no texture.
                            if _base_color_matches(material.get_base_color(),
                                                   config.MOTOR_BASE_COLORS):
                                material.set_base_color([*config.COLOR_MOTOR, 1.0])
                                continue
                            material.set_base_color([*config.COLOR_BY_GROUP[group], 1.0])
                            if realistic:
                                relief_only(material, textures[_GROUP_TEXTURE_SET[group]])
                            continue

                        if not realistic:
                            continue
                        color = material.get_base_color()
                        if _base_color_matches(color, _ALUMINUM_BASE_COLORS):
                            texture_set = textures["aluminum"]
                        elif _base_color_matches(color, _WHITE_PLASTIC_BASE_COLORS):
                            texture_set = textures["plastic_white"]
                        elif _base_color_matches(color, _ORANGE_PLASTIC_BASE_COLORS):
                            texture_set = textures["plastic_orange"]
                        else:
                            continue
                        material.set_base_color_texture(texture_set["base_color"])
                        relief_only(material, texture_set)
