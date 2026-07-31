"""Task, robot and reward constants: the single place to edit them.

Plain module-level constants, because several are derived from others (the per-step motion caps
are measured real speed divided by the control rate).

Elsewhere: training hyperparameters in ``sac/args.py``, randomization ranges in
``RandomizationConfig`` / ``PickPlaceRandomizationConfig`` under ``envs/``, network architecture
in ``policy/``.

The mjlab twin has its own copy of the control/effort numbers in
``rl/environments/mjlab/src/soframe_rl/so101_constants.py``; keep the two in step by hand.
"""

import numpy as np
import sapien

from soframe_policy import rig

# =====================================================================================
# Joint contract -- SHARED with deploy, defined in soframe_policy.rig
# =====================================================================================
# Joint order, control rate, measured joint speeds, per-step motion caps, joint limits, force
# limits and the rest pose live in ``soframe_policy/rig.py``, because the real robot needs the
# same numbers and cannot import this package. Re-exported here for readability; EDIT THEM THERE
# so sim and deploy change together.
CONTROL_HZ = rig.CONTROL_HZ
SIM_HZ = 100.0   # physics substeps per second; SIM_HZ / CONTROL_HZ substeps per control step

REAL_JOINT_SPEED = rig.REAL_JOINT_SPEED
JOINT_DELTA_LIMITS = rig.JOINT_DELTA_LIMITS
JOINT_LIMITS = rig.JOINT_LIMITS
REST_QPOS = rig.REST_QPOS

STS3215_STALL_TORQUE = rig.STS3215_STALL_TORQUE
RAIL_FORCE_LIMIT = rig.RAIL_FORCE_LIMIT
JOINT_FORCE_LIMITS = rig.JOINT_FORCE_LIMITS

# Multiplier on the arm/rail deltas (NOT the gripper), for arm-speed ablations only.
# 1.0 = the measured real speed. Overridden at runtime by --arm_speed_scale.
ARM_SPEED_SCALE = 1.0

# PD gains for the position controller (nominal; randomized per episode around these). Sim-only:
# the real servos run their own internal loop.
JOINT_STIFFNESS = 1e3
JOINT_DAMPING = 1e2


# =====================================================================================
# Solver cost
# =====================================================================================
# PhysX solver effort: pure cost/fidelity knobs, no physical constant, set below ManiSkill's
# defaults of 15 position / 1 velocity iterations with friction every iteration.
#
# 8 is the floor. What bounds it is the residual speed of a cube that should be sitting still:
# 1.33 mm/s here, 5.29 at 4 iterations (where it starts buzzing). Lower these only after
# re-checking that plus cube tunnelling through the bin's 5 mm collision walls.
SOLVER_POSITION_ITERATIONS = 8
SOLVER_VELOCITY_ITERATIONS = 0
# Off: it would matter for an object held by friction alone, but the cube is gripped between two
# jaws at static_friction 2.0 and weighs 3.2 g.
FRICTION_EVERY_ITERATION = False


# =====================================================================================
# Render cost
# =====================================================================================
# OFF: a shadow-casting light costs a whole extra geometry pass per camera per step, and the sim's
# cast shadow is a sim-only artifact anyway. The real rig's overhead captures
# (rl/deploy/utils/captures/) show only soft contact darkening at an object's base, never a
# directional shadow displaced from its caster. SHADOW_MAP_SIZE matters only when this is True.
RASTER_SHADOWS = False
SHADOW_MAP_SIZE = 512


# =====================================================================================
# Robot geometry
# =====================================================================================
# Grasp point between the finger pads, in gripper_link's frame. Matches the `grasp_site` MJCF
# site added for rl/environments/mjlab (simulation/mjcf/so101_on_frame.xml).
GRASP_SITE_OFFSET = sapien.Pose(p=[0.012, 0.0, -0.07])


# =====================================================================================
# Task objects
# =====================================================================================
# Geometry mirrored from the CAD sources in simulation/assets/objects/. That directory's
# convert.py re-exports the OBJs the env loads AND asserts these numbers still match the 3MF, so
# editing the CAD without updating here fails loudly instead of changing the task silently.
OBJECTS_ROOT = "simulation/assets/objects"

# Cube: a 20 mm cube, base at z=0, centred in xy. Collision is an exact box.
CUBE_SIZE = 0.020
CUBE_HALF = CUBE_SIZE / 2

# Bin: 100x100x30 mm outer with 2 mm walls and a 2 mm floor, so a 96 mm square opening 28 mm
# deep. Base at z=0, centred in xy, corners filleted. The 20 mm cube drops in at any yaw.
BIN_FOOTPRINT_HALF = 0.050
BIN_INTERIOR_HALF = 0.048
BIN_HEIGHT = 0.030
BIN_FLOOR_THICKNESS = 0.002
# The true 2 mm walls are thin enough that a fast cube tunnels through in one 10 ms step, so the
# collision walls are thickened OUTWARD from them, inner faces left on the true interior: the
# opening stays honest, the collision footprint is wider than the visual by (this - 2 mm) per side.
BIN_WALL_COLLISION_THICKNESS = 0.005


# =====================================================================================
# Scene layout
# =====================================================================================
WORK_SURFACE_Z = 0.0

# ONE spawn zone shared by both objects: x [-0.374, -0.016], y [-0.766, -0.038], i.e. 358 x 728 mm.
#
# Three edges are the overhead camera's footprint, the largest axis-aligned rectangle inside its view
# measured at both the cube's height and the taller bin's rim: the policy is vision-only, so a spawn
# out of frame is unobservable rather than merely hard. The far -x edge is pulled 100 mm inside that
# (-0.474 -> -0.374), where the jaw has to come in tilted.
#
# Reach at the cube's height, measured by sweeping the joint limits: x [-0.247, +0.050] with the jaw
# within 15 deg of vertical, x [-0.378, +0.117] at any jaw angle. The zone sits inside the latter, and
# each object's own footprint inset pulls it further in (cube centres to x [-0.340, -0.050]), so every
# spawn is workable, just not all of them straight down. Do NOT quote the top-down number as the
# task's reach limit: v6_dino hits 1.00 success_at_end and sustains 0.86-0.97.
#
# The arm is in the zone too, so at reset it occludes the cube in a minority of spawns,
# concentrated where the arm parks. All of them clear once the gantry moves, so the occlusion is
# transient, not unobservable state.
WORKSPACE_CENTER = (-0.195, -0.402)
WORKSPACE_HALF = (0.179, 0.364)

# Extra inset on top of each object's own footprint. The zone bounds are measured at the NOMINAL
# camera pose, and randomization jitters the overhead camera a few mm and up to a degree of FOV
# per episode; this is the margin, so an object never sits exactly on the frame edge.
SPAWN_PADDING = 0.02

# Minimum GAP (not centre distance) between the cube's and the bin's footprints; the cube is
# rejection-sampled against the bin until it clears this. Both get a random yaw, so the test uses
# circumradii: yaw-invariant and slightly conservative.
SPAWN_MIN_GAP = 0.05
# Resample rounds before falling back to a deterministic placement at the far end of the
# workspace. Never silently emits an overlapping spawn.
SPAWN_MAX_ATTEMPTS = 32

# Drop point height above the bin RIM (not the floor): the jaw cannot open at depth inside the
# bin. Carry to here, open, let gravity finish; success still requires the cube settled IN the bin.
GOAL_CLEARANCE = 0.05

# Steps per episode, passed to register_env. Replay sizing is expressed in EPISODES rather than
# transitions (sac/args.py: --replay_episodes), so the two must agree: a buffer quoted as "2
# episodes of retention" is 2 * this * num_envs transitions.
#
# 200 steps = 20 s per attempt at CONTROL_HZ, sized against rail travel, which dominates: park to
# cube then cube to bin costs about 110 steps at a p90 spawn, leaving the worst decile 45% of the
# episode to descend, grasp, lift, release and settle.
EPISODE_HORIZON = 200


# =====================================================================================
# Reward ladder
# =====================================================================================
# A monotonic stage ladder: each stage is a fixed rung plus at most 1.0 of bounded shaping, and
# each rung sits above the previous stage's maximum, so a regression drops to a lower rung on its
# own.
#
#   reach [0,1.5] < grasped [2,3] < holding [4,5] < released 6 < success 10
#
# No penalty terms: motion limits are structural (delta action space + effort limits above). If
# you add a rung, keep the spacing strictly increasing; the assert below checks it.
REWARD_SUCCESS = 10.0     # terminal: bar settled in bin, arm + bar static, gripper clear
RUNG_RELEASED = 6.0       # bar over the bin, released -- must dominate holding
RUNG_HOLDING = 4.0        # bar over the bin, still gripped (leave it by opening the jaw)
RUNG_GRASPED = 2.0        # bar grasped, carried toward the drop point (not yet over the bin)
#                           the reach stage's base is 0 (shaping only)

# Two jaw-motion shaping terms beyond the rungs, one per end of the pick, so neither jaw motion is
# a blind leap off a flat plateau.
#
# Opening: how far opening the jaw over the bin pays BEFORE contact breaks. Keep below
# (RUNG_RELEASED - RUNG_HOLDING) so the most-open still-holding pose stays worse than a release.
SHAPE_HOLD_OPEN = 1.0
# Closing: how far closing the jaw pays while the tool is ON the cube (xy aligned AND within
# REACH_Z_ALIGNED in z) but has not caught it. Keep below (RUNG_GRASPED - 1.0), the headroom above
# the reach stage's shaping maximum, so a shut jaw with no cube stays worse than a grasp.
SHAPE_REACH_CLOSE = 0.5

SHARP = 5.0               # single tanh sharpness (~1/length-scale, m^-1) for every distance term
REACH_XY_ALIGNED = 0.03   # tcp within this xy of the bar is "aligned" and may descend
REACH_Z_ALIGNED = 0.02    # ... and within this z of it as well, so the jaw is around it and may close

# Reach and grasped each carry 1.0 of distance shaping on top of their base; each jaw term adds to
# the stage it pays in.
assert (
    1.0 + SHAPE_REACH_CLOSE < RUNG_GRASPED
    and RUNG_GRASPED + 1.0 < RUNG_HOLDING
    and RUNG_HOLDING + SHAPE_HOLD_OPEN < RUNG_RELEASED < REWARD_SUCCESS
), "every reward stage's maximum must stay strictly below the next stage's rung"


# =====================================================================================
# Colours
# =====================================================================================
# One place for every colour in the scene. These are LINEAR base colours (not sRGB), which is what
# SAPIEN's RenderMaterial takes.
#
# Never pure 0 or 1: pure black returns no light and renders as a flat silhouette with no shape
# cues, pure white clips under the softbox and loses its edges the same way. Real matte black
# plastic reflects ~4-5% anyway.
BLACK = (0.04, 0.04, 0.04)
WHITE = (0.88, 0.88, 0.88)

# Hue-separated task-object colours, selected by --object_colors distinct. The squint CNN needs
# them: at res 32 the 20 mm cube is a SINGLE pixel, so hue is its only cue. dino_patch resolves
# the cube as a shape at res 112/168 and does not. Same no-pure-0-or-1 rule as above.
YELLOW = (0.80, 0.62, 0.04)
BLUE = (0.04, 0.12, 0.60)

# Task objects. Overridden per-run by --object_colors; see sac/args.py.
COLOR_CUBE = BLACK
COLOR_BIN = BLACK

# Rig. "extrusion" is the 2020 aluminium frame and its brackets; "slider" is the gantry carriage
# the arm rides on, plus the rail drive gear; "arm" is the arm itself INCLUDING its own base;
# "gripper" is the jaw assembly.
COLOR_EXTRUSION = BLACK
COLOR_SLIDER = BLACK
COLOR_GRIPPER = WHITE
COLOR_ARM = WHITE

# Motor hardware is NOT part of the scheme: a link is a printed shell plus an STS3215 in a dark
# grey case, and recolouring the whole link paints the motor too. Only the printed structure takes
# COLOR_ARM / COLOR_GRIPPER / COLOR_SLIDER; anything matching a motor colour below keeps this,
# which is the URDF's own value.
COLOR_MOTOR = (0.1, 0.1, 0.1)

# Baked URDF colours that identify motor hardware. Matched against a part's ORIGINAL colour, so
# apply_materials must read it before recolouring (it does; it is a single pass).
MOTOR_BASE_COLORS = (
    (0.1, 0.1, 0.1),        # STS3215 case on the arm links
    (0.098, 0.098, 0.098),  # carriage motor + electronics (sg_ziji, xg_ziji, pcb, motor_1723, ...)
)

# Link-name patterns -> colour group. Matched as substrings against the URDF link names, first
# match wins. Any link that matches nothing keeps the colour baked into the URDF -- that is
# deliberate for the lightbox panels (the work surface, near-white) and the camera holders.
COLOR_GROUPS = (
    # (group name, link-name substrings)
    # The metal servo horns (金属舵盘) are absent on purpose: hardware, not printed structure, so
    # they keep the URDF's bright-metal colour.
    ("gripper",   ("gripper_link", "moving_jaw")),
    # `base_link` is the ARM's own base, the block bolted on top of the carriage. Do NOT add
    # `base_mount` here: despite the name it is the SLIDER BASE, the tray that rides the rail, and
    # it belongs below.
    ("arm",       ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link",
                   "wrist_link")),
    # The carriage: `base_mount` is its tray, `pinion` its drive gear.
    ("slider",    ("base_mount", "pinion", "20mm_gantry_plate", "v_wheel", "m5_nylock",
                   "m5x25_low_profile_screw", "handle", "sg_ziji", "xg_ziji", "pcb_chazuo",
                   "ge_27", "zk_122", "motor_1723")),
    ("extrusion", ("100cm", "50cm", "inside_corner_bracket", "angle_bracket")),
)

COLOR_BY_GROUP = {
    "gripper": COLOR_GRIPPER,
    "arm": COLOR_ARM,
    "slider": COLOR_SLIDER,
    "extrusion": COLOR_EXTRUSION,
}


# =====================================================================================
# Cameras
# =====================================================================================
# Calibrated against the real rig; the deploy side rectifies each real camera to match, using a
# mapping fitted in rl/deploy/utils/. Nothing in this tree reads those mappings.
WRIST_CAMERA_FOV = np.deg2rad(58)      # from the MJCF twin's fovy
OVERHEAD_CAMERA_FOV = np.deg2rad(38)   # calibrated against the real rig
SENSOR_RESOLUTION = 128                # square render size for both sensor cameras

# Sensor near/far clip. The rig fits in ~1.5 m, so a 3 m far plane covers every surface either
# camera can see and CULLS THE GROUND PLANE (see GROUND_ALTITUDE), which is what lets the policy
# obs_mode be plain "rgb" with no segmentation pass. Raising this past -GROUND_ALTITUDE puts the
# ground back in frame.
SENSOR_NEAR = 0.01
SENSOR_FAR = 3.0

# Safety catch-all plane for anything that rolls off the panels. MUST sit beyond SENSOR_FAR so
# neither policy camera renders it: the background is then the renderer's black clear colour.
GROUND_ALTITUDE = -5.0

# The third-person video camera; the policy never sees it. Its render target is allocated per
# sub-scene, so the size costs VRAM per env. Raise it for a --capture_video eval; training does not
# read it.
HUMAN_RENDER_RESOLUTION = 128
