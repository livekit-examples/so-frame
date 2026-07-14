"""SO-101-on-frame robot definition for ManiSkill3.

Loads the repo's existing, unmodified ``simulation/urdf/so101_on_frame.urdf`` (the
frame-mounted arm with its rail, lightbox, and already-calibrated wrist/overhead camera
mounts) as a ManiSkill ``BaseAgent``. Modeled on the reference
[Squint](https://github.com/aalmuzairee/squint) implementation's ``envs/robot/so101.py``,
which targets a bare tabletop SO-101 with no rail.

This version adds the frame's linear rail joint (``dof_slider``) as an extra controllable
DOF, matching the 7-actuator action space used by this repo's ``rl/mjlab`` implementation.
It also computes the end-effector (``tcp``) point from a fixed local offset off
``gripper_link`` instead of two fingertip links, since this URDF has no
``finger1_tip``/``finger2_tip`` links the way Squint's does. The offset
(``0.012, 0, -0.07``) matches the ``grasp_site`` added to the MJCF for ``rl/mjlab``.
"""

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

# so101_on_frame.py -> robot -> soframe_rl_maniskill -> src -> rl/maniskill -> rl -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[5]
SO101_ON_FRAME_URDF = (
    _REPO_ROOT / "simulation" / "urdf" / "so101_on_frame.urdf"
)
assert SO101_ON_FRAME_URDF.exists(), f"SO-101-on-frame URDF not found at {SO101_ON_FRAME_URDF}"

# Fixed offset (in gripper_link's frame) to the grasp point between the finger pads.
# Matches the `grasp_site` MJCF site added for rl/mjlab (simulation/mjcf/so101_on_frame.xml).
GRASP_SITE_OFFSET = sapien.Pose(p=[0.012, 0.0, -0.07])

# Per-joint delta-position step limits (rad for revolute joints, m for the rail).
# Keyed by name (not position) so the controller config below is robust regardless of
# the joint order SAPIEN's URDF loader assigns to `robot.active_joints`.
_JOINT_DELTA_LIMITS = {
    "dof_slider": 0.02,
    "shoulder_pan": 0.1,
    "shoulder_lift": 0.1,
    "elbow_flex": 0.1,
    "wrist_flex": 0.1,
    "wrist_roll": 0.1,
    "gripper": 0.2,
}

# PBR texture sets for `apply_realistic_materials` below (visualization only, see
# `envs/base_random_env.py`'s `RandomizationConfig.realism_mode` and
# `examples/render_realistic.py`).
_TEXTURES_ROOT = _REPO_ROOT / "simulation" / "assets" / "textures"

# The URDF (simulation/urdf/so101_on_frame.urdf) only defines flat `<color rgba="...">`
# materials, and SAPIEN's URDF loader doesn't preserve the original `<material name="...">`
# string on the loaded render parts -- only the resolved rgba. So `apply_realistic_materials`
# matches each part's baked base_color back to the URDF's known material colors to decide
# which texture set applies.
_ALUMINUM_BASE_COLORS = [
    (0.4, 0.4, 0.4),                 # extrusion_gray -- the 2020 aluminum extrusion bars
    (0.749020, 0.749020, 0.749020),  # unnamed hardware: nuts/screws/wheels/gantry plate
    (0.960784, 0.960784, 0.964706),  # "metal servo horn" links (金属舵盘)
]
_WHITE_PLASTIC_BASE_COLORS = [
    (1.0, 1.0, 1.0),                 # 3d_printed, camera_holder_white, camera_mount_white
    (0.901961, 0.901961, 0.901961),  # unnamed light bracket parts
]
_ORANGE_PLASTIC_BASE_COLORS = [
    (0.94, 0.49, 0.06),  # gripper_orange, accent_orange
]


def _base_color_matches(color, targets, tol: float = 0.02) -> bool:
    return any(
        abs(color[0] - t[0]) < tol and abs(color[1] - t[1]) < tol and abs(color[2] - t[2]) < tol
        for t in targets
    )


def _load_texture_set(name: str) -> dict:
    """Load one texture set from ``simulation/assets/textures/<name>/``. Roughness/
    metalness/normal maps use ``srgb=False`` (they're data, not color), per
    ``RenderTexture2D``'s own guidance."""
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

    # qpos order matches `self.robot.active_joints` (rail -> arm -> gripper):
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
        delta_lower = [-_JOINT_DELTA_LIMITS[name] for name in joint_names]
        delta_upper = [_JOINT_DELTA_LIMITS[name] for name in joint_names]

        pd_joint_pos = PDJointPosControllerConfig(  # noqa: F405
            joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(  # noqa: F405
            joint_names,
            delta_lower,
            delta_upper,
            stiffness=[1e3] * len(joint_names),
            damping=[1e2] * len(joint_names),
            force_limit=100,
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
        # Already-calibrated camera links baked into the URDF (see simulation/urdf/README.md).
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
        """Wire real PBR textures (``simulation/assets/textures/``) onto the robot's flat-
        color URDF materials, for one-off visualization renders only (see
        ``RandomizationConfig.realism_mode`` and ``examples/render_realistic.py``) --
        never called during training. Parts we don't have a texture for (the near-black
        servo casings) are left with their original flat color.
        """
        aluminum = _load_texture_set("aluminum")
        plastic_white = _load_texture_set("plastic_white")
        plastic_orange = _load_texture_set("plastic_orange")

        for link in self.robot.links:
            for obj in link._objs:
                render_body_component = obj.entity.find_component_by_type(RenderBodyComponent)
                if render_body_component is None:
                    continue
                for render_shape in render_body_component.render_shapes:
                    for part in render_shape.parts:
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
