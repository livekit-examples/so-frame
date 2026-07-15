"""SO-101-on-frame robot definition for mjlab.

Wraps the existing, unmodified ``simulation/mjcf/so101_on_frame.xml`` as an mjlab
``EntityCfg``. Following mjlab's own robot assets (see i2rt_yam), the model's XML
actuators are stripped and re-declared here in Python so their gains/effort limits
are first-class, tunable, sim-to-real knobs.

The model is a *fixed-base* articulated entity (the frame is welded to the world;
only the rail slider + the arm move). mjlab auto-wraps fixed-base entities in a
mocap body, which also carries the frame's loose worldbody geoms, so the whole rig
positions correctly per environment. The only edits to the XML are the ``grasp_site``
added to ``gripper_link`` and collision boxes added to the lightbox's panel geoms
(they're otherwise visual-only).
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# Paths.
##

# rl/mjlab/src/soframe_rl/so101_constants.py -> repo root is parents[4].
_REPO_ROOT = Path(__file__).resolve().parents[4]
SO101_XML = _REPO_ROOT / "simulation" / "mjcf" / "so101_on_frame.xml"
assert SO101_XML.exists(), f"SO-101 model not found at {SO101_XML}"

# Site added to gripper_link that marks the grasp point between the finger pads.
GRASP_SITE = "grasp_site"

##
# Joint groups.
##

SLIDER_JOINT = "dof_slider"
ARM_JOINTS = (
  "shoulder_pan",
  "shoulder_lift",
  "elbow_flex",
  "wrist_flex",
  "wrist_roll",
)
GRIPPER_JOINT = "gripper"


# World-space height (m) of the lightbox bottom panel's top surface (its collision box,
# see simulation/mjcf/so101_on_frame.xml, is centered ~(-0.245, -0.5)).
WORK_SURFACE_Z = 0.083


def get_spec() -> mujoco.MjSpec:
  """Load the SO-101 model and strip its XML actuators.

  mjlab adds actuators from the ``EntityArticulationInfoCfg`` below; leaving the
  XML's ``<position>`` actuators in place would double them up.
  """
  spec = mujoco.MjSpec.from_file(str(SO101_XML))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


##
# Actuators.
#
# NOTE: These are reasonable *starting* PD gains, mirroring the generic values in
# the XML's <default> classes. They are NOT calibrated STS3215 (arm/gripper) or
# rail-drive (slider) parameters. Tune them and randomize them via domain-
# randomization events before trusting sim-to-real transfer. See the repo's
# MJCF README, which flags the same caveat.
##

_ARM_STIFFNESS = 18.0
_ARM_DAMPING = 0.6
_ARM_ARMATURE = 0.02
_ARM_EFFORT = 10.0  # matches the joints' actuatorfrcrange in the XML.

_SLIDER_STIFFNESS = 400.0
_SLIDER_DAMPING = 8.0
_SLIDER_ARMATURE = 0.1
_SLIDER_EFFORT = 60.0

# `damping` is the active PD derivative gain (kv). `viscous_damping=0.0` zeroes the
# XML's passive <joint damping> so all damping is controlled here (clean for DR).
ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=ARM_JOINTS,
  stiffness=_ARM_STIFFNESS,
  damping=_ARM_DAMPING,
  armature=_ARM_ARMATURE,
  effort_limit=_ARM_EFFORT,
  viscous_damping=0.0,
)

GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(GRIPPER_JOINT,),
  stiffness=_ARM_STIFFNESS,
  damping=_ARM_DAMPING,
  armature=_ARM_ARMATURE,
  effort_limit=_ARM_EFFORT,
  viscous_damping=0.0,
)

SLIDER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(SLIDER_JOINT,),
  stiffness=_SLIDER_STIFFNESS,
  damping=_SLIDER_DAMPING,
  armature=_SLIDER_ARMATURE,
  effort_limit=_SLIDER_EFFORT,
  viscous_damping=0.0,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(SLIDER_ACTUATOR, ARM_ACTUATOR, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

##
# Initial state (home keyframe).
#
# pos z=0.09 sets the frame's leg bottoms on mjlab's terrain plane (z=0) with the
# lightbox body raised on them, matching the real rig. Objects rest on the added
# white work floor at WORK_SURFACE_Z (0.08), inside the lightbox and in reach.
# joint_pos is a gentle "arm folded toward the workspace, gripper open" pose.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.09),
  joint_pos={
    SLIDER_JOINT: 0.0,
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.5,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    GRIPPER_JOINT: 1.2,  # open-ish (range -0.17..1.75); confirm sign in viewer.
  },
  joint_vel={".*": 0.0},
)


def get_so101_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


##
# Per-actuator action scale (delta-position step size), following mjlab's yam
# heuristic: scale = 0.25 * effort_limit / stiffness. Arm ~0.14 rad, slider
# ~0.04 m, gripper ~0.14 rad per step.
##

SO101_ACTION_SCALE: dict[str, float] = {
  name: 0.25 * act.effort_limit / act.stiffness
  for act in ARTICULATION.actuators
  for name in act.target_names_expr
}


##
# Workspace (env-relative, on the terrain plane).
#
# !!! THESE ARE ESTIMATES. Validate reachability in the viewer and adjust. !!!
# Ranges are relative to each environment's origin. The arm base sits near
# (0, -0.49) env-relative; these bracket a patch in front of it on the plane.
##

WORKSPACE_X = (-0.35, -0.10)
WORKSPACE_Y = (-0.65, -0.30)
CUBE_Z = (0.02, 0.03)  # spawn just above the plane so it settles.
