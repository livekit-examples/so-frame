"""Task, robot and reward constants -- the single place to edit them.

Everything here is a plain module-level constant, deliberately: several are *derived* from
others (the per-step motion caps are measured real speed divided by the control rate), and a
data file would either lose that relationship or silently let the two drift apart. Edit the
measured quantity and the derived one follows.

What lives elsewhere, and why:

- **Training hyperparameters** -> ``sac/args.py``. They are per-run rather than per-rig
  (learning rates, batch size, update ratio), and tyro exposes every one as a CLI flag, so
  they are overridden per experiment rather than edited in place.
- **Domain randomization** -> ``RandomizationConfig`` / ``PickPlaceRandomizationConfig`` in
  ``envs/``. Already dataclasses, already per-instance, and randomization *ranges* are
  conceptually different from the fixed physical constants here.
- **Network architecture** -> ``nets/`` (the shared ``soframe_nets`` package). Changing those
  invalidates existing checkpoints, so they are pinned by tests rather than exposed as knobs.

The mjlab twin has its own copy of the control/effort numbers in
``rl/mjlab/src/soframe_rl/so101_constants.py``; keep the two in step by hand.
"""

import numpy as np
import sapien

# =====================================================================================
# Control rate
# =====================================================================================
# The sim's control frequency, and therefore the rate the deployed policy is driven at. The
# deploy loop (sim2real, PORTAL_FPS) must match: a policy trained at one rate and stepped at
# another sees a different amount of world motion per decision.
CONTROL_HZ = 10.0
SIM_HZ = 100.0   # physics substeps per second; SIM_HZ / CONTROL_HZ substeps per control step


# =====================================================================================
# Motion limits -- the speed half of the sim2real contract
# =====================================================================================
# Measured on the real rig: arm joints 29-34 deg/s (0.5 rad/s is 28.6 deg/s, at the low end of
# the measured band), rail max ~7 cm/s off the control UI. The gripper is deliberately quicker,
# since the jaw has to open and close within a single reach.
#
# This is what actually enforces real speed. The effort limits below bound TORQUE, not
# velocity: 3 N.m against the servo's damping reaches terminal velocity around 4.9 rad/s
# (~280 deg/s), an order of magnitude past the real arm. The real arm is slow because the
# servo's commanded speed profile limits it, so sim has to impose that rate explicitly.
REAL_JOINT_SPEED = {          # rad/s for revolute joints, m/s for the rail
    "dof_slider":    0.07,
    "shoulder_pan":  0.5,
    "shoulder_lift": 0.5,
    "elbow_flex":    0.5,
    "wrist_flex":    0.5,
    "wrist_roll":    0.5,
    "gripper":       2.0,
}

# Per-step position deltas at full command. Derived, so changing CONTROL_HZ keeps real speed.
JOINT_DELTA_LIMITS = {name: speed / CONTROL_HZ for name, speed in REAL_JOINT_SPEED.items()}

# Multiplier on the arm/rail deltas (NOT the gripper), for arm-speed ablations only.
# 1.0 = the measured real speed. Overridden at runtime by --arm_speed_scale.
ARM_SPEED_SCALE = 1.0


# =====================================================================================
# Effort limits -- the torque half of the sim2real contract
# =====================================================================================
# The six arm joints are Feetech STS3215 servos, which stall around 3 N.m (30 kg*cm at 12 V).
# Capping here stops the policy learning pushes the real arm cannot deliver, which showed up on
# hardware as the arm stalling mid-lift. Matches simulation/mjcf/sts3215.xml's
# forcerange="-3.0 3.0" and the URDF's effort="3".
#
# NOTE: domain randomization re-writes drive properties per episode and must pass these back
# through; see BaseRandomEnv._force_limit. Passing a constant there silently overrode this.
STS3215_STALL_TORQUE = 3.0    # N*m

# The rail is a separate belt drive, not an STS3215; linear force in N, functional not measured.
RAIL_FORCE_LIMIT = 100.0      # N

JOINT_FORCE_LIMITS = {
    "dof_slider":    RAIL_FORCE_LIMIT,
    "shoulder_pan":  STS3215_STALL_TORQUE,
    "shoulder_lift": STS3215_STALL_TORQUE,
    "elbow_flex":    STS3215_STALL_TORQUE,
    "wrist_flex":    STS3215_STALL_TORQUE,
    "wrist_roll":    STS3215_STALL_TORQUE,
    "gripper":       STS3215_STALL_TORQUE,
}

# PD gains for the position controller (nominal; randomized per episode around these).
JOINT_STIFFNESS = 1e3
JOINT_DAMPING = 1e2


# =====================================================================================
# Robot geometry
# =====================================================================================
# Grasp point between the finger pads, in gripper_link's frame. Matches the `grasp_site` MJCF
# site added for rl/mjlab (simulation/mjcf/so101_on_frame.xml).
GRASP_SITE_OFFSET = sapien.Pose(p=[0.012, 0.0, -0.07])

# Rest pose (the "rest" keyframe). The deploy-side reset ramps to this same pose.
REST_QPOS = {
    "dof_slider": 0.0, "shoulder_pan": 0.0, "shoulder_lift": -0.5,
    "elbow_flex": 0.8, "wrist_flex": 0.6, "wrist_roll": 0.0, "gripper": 1.2,
}


# =====================================================================================
# Task objects
# =====================================================================================
# Geometry mirrored from the CAD sources in simulation/assets/objects/. That directory's
# convert.py re-exports the OBJs the env loads AND asserts these numbers still match the 3MF,
# so editing the CAD without updating here fails loudly instead of changing the task silently.
OBJECTS_ROOT = "simulation/assets/objects"

# Cube: a 20 mm cube, base at z=0, centred in xy. Collision is an exact box.
CUBE_SIZE = 0.020
CUBE_HALF = CUBE_SIZE / 2

# Bin: 100x100x30 mm outer with 2 mm walls and a 2 mm floor, so a 96 mm square opening 28 mm
# deep. Base at z=0, centred in xy, corners filleted. The 20 mm cube has 76 mm of clearance in
# the opening, so it drops in at any yaw.
BIN_FOOTPRINT_HALF = 0.050
BIN_INTERIOR_HALF = 0.048
BIN_HEIGHT = 0.030
BIN_FLOOR_THICKNESS = 0.002
BIN_WALL_THICKNESS = 0.002
# The real walls are 2 mm, thin enough that a fast cube can tunnel through in one 10 ms step.
# Collision walls are thickened OUTWARD with their inner faces left on the true interior, so the
# opening the cube must fit through stays honest while contact stays reliable. This makes the
# collision footprint wider than the visual by (this - BIN_WALL_THICKNESS) per side.
BIN_WALL_COLLISION_THICKNESS = 0.005


# =====================================================================================
# Scene layout
# =====================================================================================
WORK_SURFACE_Z = 0.0

# ONE spawn zone spanning the workspace, rather than a separate region per object.
#
# Widened to the measured maximum: the intersection of three independently measured limits,
# x [-0.270, -0.012], y [-0.760, -0.050].
#
#   1. Top-down graspable reach. Sampling the arm's joint space at a fixed rail position and
#      keeping poses with the TCP 0-50 mm up and the approach axis within 30 deg of straight
#      down gives good coverage over x [-0.27, +0.06]. Past x = -0.27 only a narrow sliver of
#      configurations reach, so grasps there would be fragile. The rail translates the arm along
#      y (820 mm of travel), so y reach is not the binding constraint -- x is.
#   2. Overhead camera footprint. The overhead camera is static, so its view of the work surface
#      is a fixed trapezoid; this is the largest axis-aligned rectangle inside it, evaluated at
#      BOTH the cube's height and the bin's rim (the taller object sees a smaller footprint).
#      Vision-only policy, so anything outside this is unlearnable.
#   3. Physical clearance. Probing where the bin actually settles on the surface: clear over
#      x [-0.34, 0.00], y [-0.80, -0.05]. Only y >= 0 is blocked, by the frame's near edge.
#
# This is 258 x 710 mm, 1.5x the area of the 200 x 600 mm it replaced. NOTE the old zone ran to
# y = -0.80, which is 40 mm PAST the overhead camera's far edge at -0.760 -- objects spawned in
# that strip were out of frame, and the policy could not have seen them.
#
# Known consequence of one zone spanning the workspace: the ARM is in the workspace too, so at
# reset the cube is occluded by it in ~12% of spawns (measured by segmentation over 300 spawns;
# the bin, being far larger, is never fully hidden). Every one of those clears once the gantry
# moves -- checked by teleporting it aside, after which 0 remain hidden -- so this is transient
# occlusion the policy can resolve, not an unobservable state. It is concentrated in
# y [-0.37, -0.10], the near end where the arm parks. The old split-region layout avoided it by
# keeping the item in the far region only, which is also what made position predictable.
WORKSPACE_CENTER = (-0.141, -0.405)
WORKSPACE_HALF = (0.129, 0.355)

# Extra inset on top of each object's own footprint. The zone bounds above already guarantee
# visibility, but they were measured at the NOMINAL camera pose, and domain randomization jitters
# the overhead camera by a few mm and up to a degree of FOV each episode. This is the margin for
# that, so an object never sits exactly on the frame edge.
SPAWN_PADDING = 0.02

# Minimum GAP (not centre distance) between the cube's and the bin's footprints. The cube is
# rejection-sampled against the bin until it clears this. Both objects get a random yaw, so the
# test uses circumradii and is therefore yaw-invariant and slightly conservative.
SPAWN_MIN_GAP = 0.05
# How many resample rounds before falling back to a deterministic placement at the far end of
# the workspace. Never silently emits an overlapping spawn.
SPAWN_MAX_ATTEMPTS = 32

# Drop point height above the bin RIM (not the floor): the jaw cannot open at depth inside the
# bin. Carry to here, open, let gravity finish; success still requires the cube settled IN the bin.
GOAL_CLEARANCE = 0.05


# =====================================================================================
# Reward ladder
# =====================================================================================
# A monotonic stage ladder. Each stage is a fixed rung plus at most 1.0 of bounded shaping, and
# each rung sits above the previous stage's maximum. That spacing is the whole mechanism:
# progress is strictly monotone, a regression drops to a lower rung on its own, and there is
# nothing to trade off against.
#
#   reach [0,1] < grasped [2,3] < holding [4,5] < released 6 < success 10
#
# There are no penalty terms. Motion limits are enforced structurally by the delta action space
# and the effort limits above, so there is no smoothness or effort weight to balance against
# the task reward. If you add a rung, keep the spacing strictly increasing -- the env asserts it.
REWARD_SUCCESS = 10.0     # terminal: bar settled in bin, arm + bar static, gripper clear
RUNG_RELEASED = 6.0       # bar over the bin, released -- must dominate holding
RUNG_HOLDING = 4.0        # bar over the bin, still gripped (leave it by opening the jaw)
RUNG_GRASPED = 2.0        # bar grasped, carried toward the drop point (not yet over the bin)
#                           the reach stage's base is 0 (shaping only)

# The one shaping term beyond the rungs: how far opening the jaw over the bin pays BEFORE
# contact breaks. Without it the holding rung is a flat plateau and the policy sits on it
# holding the bar, because releasing is a blind leap to the next rung. Keep it below
# (RUNG_RELEASED - RUNG_HOLDING) so the most-open still-holding pose stays strictly worse than
# an actual release.
SHAPE_HOLD_OPEN = 1.0

SHARP = 5.0               # single tanh sharpness (~1/length-scale, m^-1) for every distance term
REACH_XY_ALIGNED = 0.03   # tcp within this xy of the bar is "aligned" and may descend


# =====================================================================================
# Cameras
# =====================================================================================
# Calibrated against the real rig; the deploy side rectifies each real camera to match, using a
# mapping fitted in sim2real/utils/. Nothing in this tree reads those mappings.
WRIST_CAMERA_FOV = np.deg2rad(58)      # from the MJCF twin's fovy
OVERHEAD_CAMERA_FOV = np.deg2rad(38)   # calibrated against the real rig
SENSOR_RESOLUTION = 128                # square render size for both sensor cameras
