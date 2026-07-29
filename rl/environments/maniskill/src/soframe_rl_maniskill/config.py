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
- **Network architecture** -> ``policy/`` (the shared ``soframe_policy`` package). Changing those
  invalidates existing checkpoints, so they are pinned by tests rather than exposed as knobs.

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
# limits and the rest pose all live in ``soframe_policy/rig.py``, because the real robot needs
# the same numbers and cannot import this package. Re-exported here so this file still reads as
# the whole picture -- but EDIT THEM THERE, so sim and deploy change together.
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
# PhysX solver settings. These are pure cost/fidelity knobs -- they change no physical constant,
# only how hard the solver works per substep -- and the physics stage is roughly a third of a
# step, so they are the cheapest thing to trade.
#
# ManiSkill's defaults are 15 position / 1 velocity iterations with friction recomputed every
# iteration, which is tuned for tasks far more contact-rich than a 3.2 g cube resting on a flat
# panel. Dropping to these cut the physics stage from 105 to 65 ms/step at 512 envs on an
# RTX 5090.
#
# The measurement that bounds how low they can go is the residual speed of a cube that should be
# sitting still: 0.37 mm/s at 15 position iterations, 1.33 at 8, 5.29 at 4. So 4 is where a cube
# left alone starts buzzing, and 8 is the floor. A cube thrown into the bin at 2 m/s never
# tunnelled its 5 mm collision walls at any setting tried. If you lower these further, re-check
# both of those -- and note that steady-state arm droop tells you nothing here, since at ~2
# degrees it is set by the real 3 N.m STS3215 effort limit rather than by solver quality.
SOLVER_POSITION_ITERATIONS = 8
SOLVER_VELOCITY_ITERATIONS = 0
# Recomputing friction on every solver iteration made no measurable difference to either check,
# so it is off: free. It would matter for an object held by friction alone; the cube is gripped
# between two jaws at static_friction 2.0 and weighs 3.2 g.
FRICTION_EVERY_ITERATION = False


# =====================================================================================
# Render cost
# =====================================================================================
# A shadow-casting light costs a whole extra geometry pass per camera per step -- the shadow map
# is rendered from the light's point of view -- and the render stage is geometry-bound (at 512
# envs, rendering at 32 px instead of 128 px changes the step by 1.3%, so it is transforming
# vertices, not filling pixels). Measured: 96.6 -> 67.4 ms/step, a 30% cut, the largest single
# render lever there is.
#
# OFF, because the shadow it cast was the wrong shadow. Measured on the real rig's own overhead
# capture (rl/deploy/utils/captures/): objects there show soft contact darkening hugging their
# base -- about -10/255 for the bin, -26/255 for the bar, and some of even that is edge bleed
# from a 640x480 UVC sensor, since it scales with the object's darkness rather than with lighting
# geometry. What the real lightbox does NOT produce anywhere in frame is a directional cast
# shadow displaced from its caster. The sim's key light produced exactly that, and a large moving
# one: the arm threw a silhouette well to its left across the work surface.
#
# So this was a sim-only artifact, and dropping it narrows the sim2real gap rather than widening
# it. The real rig's contact occlusion is a genuine feature sim does not reproduce, but sim never
# did -- boosted ambient plus two shadowless fills generate no ambient occlusion -- so that gap
# is unchanged either way. For scale, the difference this makes to a rendered frame is 4.2/255
# mean and 17/255 peak, against a jitter/augmentation stack that already swings brightness and
# contrast by +-30% and gamma over 0.7-1.4 every single step.
#
# Set True to get it back; SHADOW_MAP_SIZE only matters then. Note this also makes
# --visual_fidelity flat nearly pointless as a speed option: flat measured 87.0 ms/step against
# raster-without-shadow at 87.3, so raster now costs the same and keeps the PBR materials.
RASTER_SHADOWS = False
SHADOW_MAP_SIZE = 512


# =====================================================================================
# Robot geometry
# =====================================================================================
# Grasp point between the finger pads, in gripper_link's frame. Matches the `grasp_site` MJCF
# site added for rl/environments/mjlab (simulation/mjcf/so101_on_frame.xml).
GRASP_SITE_OFFSET = sapien.Pose(p=[0.012, 0.0, -0.07])
# (REST_QPOS is part of the shared joint contract above -- soframe_policy.rig.)


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
#   reach [0,1.5] < grasped [2,3] < holding [4,5] < released 6 < success 10
#
# There are no penalty terms. Motion limits are enforced structurally by the delta action space
# and the effort limits above, so there is no smoothness or effort weight to balance against
# the task reward. If you add a rung, keep the spacing strictly increasing -- the env asserts it.
REWARD_SUCCESS = 10.0     # terminal: bar settled in bin, arm + bar static, gripper clear
RUNG_RELEASED = 6.0       # bar over the bin, released -- must dominate holding
RUNG_HOLDING = 4.0        # bar over the bin, still gripped (leave it by opening the jaw)
RUNG_GRASPED = 2.0        # bar grasped, carried toward the drop point (not yet over the bin)
#                           the reach stage's base is 0 (shaping only)

# Two jaw-motion shaping terms beyond the rungs, one per end of the pick. Each exists for the
# same reason: without it the rung below is a flat plateau the policy can sit on, because the jaw
# motion that leaves it is a blind leap to the next rung.
#
# Opening: how far opening the jaw over the bin pays BEFORE contact breaks. Keep it below
# (RUNG_RELEASED - RUNG_HOLDING) so the most-open still-holding pose stays strictly worse than an
# actual release.
SHAPE_HOLD_OPEN = 1.0
# Closing: the mirror of the above, how far closing the jaw pays while the tool is ON the cube
# but the grasp has not caught yet. Gated on being on the cube (xy aligned AND within
# REACH_Z_ALIGNED in z), so closing on the way down -- shutting the jaw before it surrounds the
# cube -- pays nothing. Keep it below (RUNG_GRASPED - 1.0), the headroom above the reach stage's
# shaping maximum, so a shut jaw that has not caught the cube stays strictly worse than a grasp.
SHAPE_REACH_CLOSE = 0.5

SHARP = 5.0               # single tanh sharpness (~1/length-scale, m^-1) for every distance term
REACH_XY_ALIGNED = 0.03   # tcp within this xy of the bar is "aligned" and may descend
REACH_Z_ALIGNED = 0.02    # ... and within this z of it as well, so the jaw is around it and may close

# The spacing rule above, made real. Reach and grasped each carry 1.0 of bounded distance shaping
# on top of their base, and each jaw term adds to the stage it pays in.
assert (
    1.0 + SHAPE_REACH_CLOSE < RUNG_GRASPED
    and RUNG_GRASPED + 1.0 < RUNG_HOLDING
    and RUNG_HOLDING + SHAPE_HOLD_OPEN < RUNG_RELEASED < REWARD_SUCCESS
), "every reward stage's maximum must stay strictly below the next stage's rung"


# =====================================================================================
# Colours
# =====================================================================================
# One place for every colour in the scene. These are LINEAR base colours (not sRGB), which is
# what SAPIEN's RenderMaterial takes.
#
# Neither "black" nor "white" is set to a pure 0 or 1. Pure black returns no light at all and
# renders as a flat silhouette with no shape cues, and pure white clips under the softbox and
# loses its edges the same way; 0.04 and 0.88 read as black and white while keeping the shading
# gradients the vision policy localizes from. Real matte black plastic reflects ~4-5% anyway.
BLACK = (0.04, 0.04, 0.04)
WHITE = (0.88, 0.88, 0.88)

# Task objects.
COLOR_CUBE = BLACK
COLOR_BIN = BLACK

# Rig. "extrusion" is the 2020 aluminium frame and its brackets; "slider" is the gantry carriage
# the arm rides on, plus the rail drive gear; "arm" is the arm itself INCLUDING its own base;
# "gripper" is the jaw assembly.
COLOR_EXTRUSION = BLACK
COLOR_SLIDER = BLACK
COLOR_GRIPPER = WHITE
COLOR_ARM = WHITE

# Motor hardware is NOT part of the scheme. Each arm link is a printed plastic shell plus an
# STS3215 whose case is moulded dark grey, and the carriage carries dark electronics; recolouring
# a whole link paints the motors too, which made the arm read as one featureless white mass. Only
# the printed structure takes COLOR_ARM / COLOR_GRIPPER / COLOR_SLIDER; anything matching a motor
# colour below keeps this instead. The default is the URDF's own value, i.e. unchanged.
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
    # The metal servo horns (金属舵盘) are deliberately absent: they are hardware, not printed
    # structure, so they keep the URDF's bright-metal colour.
    ("gripper",   ("gripper_link", "moving_jaw")),
    # `base_link` is the ARM's own base -- the block bolted on top of the carriage -- so it takes
    # COLOR_ARM. Do NOT add `base_mount` here: despite the name it is the SLIDER BASE, the tray
    # that rides the rail on the v-wheels and carries the pinion, and it belongs below.
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

# Sensor near/far clip. The whole rig fits in ~1.5 m, so a 3 m far plane covers every surface
# either camera can see and additionally CULLS THE GROUND PLANE (see GROUND_ALTITUDE), which is
# what lets the policy obs_mode be plain "rgb" with no segmentation pass. Raising this past
# -GROUND_ALTITUDE puts the ground back in frame.
SENSOR_NEAR = 0.01
SENSOR_FAR = 3.0

# The safety catch-all plane, well below the lightbox floor, for anything that rolls off the
# panels. Sits beyond SENSOR_FAR so neither policy camera renders it: the background is then the
# renderer's black clear colour, which is exactly what the greenscreen used to paint it (its
# overlay image is 128x128 of pure black) at the cost of a whole extra segmentation render pass
# per camera per step. It was at -1.0, inside the far plane, which is why that pass existed.
GROUND_ALTITUDE = -5.0

# The third-person video camera. Not an input to anything -- the policy sees only the two sensor
# cameras -- but its render target is allocated per sub-scene, so it was reserving ~1 GB of VRAM
# at 512x512 and 1024 envs. Raise it for a --capture_video eval; training does not read it.
HUMAN_RENDER_RESOLUTION = 128
