"""The physical joint contract, shared by sim and the real robot.

These are the numbers both sides must agree on to drive the same policy: joint order, joint
limits, how far a joint may move in one control step, the control rate, and the rest pose.

They live here rather than in ``rl/environments/maniskill/config.py`` because deploy needs them too and
cannot import the training package (that would pull in mani_skill). The training config
re-exports them, so it stays the one file you read to understand the task -- but if you change
a number here, both sides change together, which is the point. The old deploy tree kept its own
copy of all of this in ``policy/bridge.py``.

Everything is in SIM units: radians for the revolute joints, metres for the rail. Converting to
the wire's units (degrees, and 0..100 for the rail) is deploy's job -- see rl/deploy/utils/bridge.py.
"""

from __future__ import annotations

# Joint order. Matches the URDF's active joints, the sim's action vector, and the wire contract
# in rl/deploy/portal.yaml. The rail comes first.
JOINT_NAMES = (
    "dof_slider",
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Control rate (Hz). The sim steps at this rate and the deploy loop ticks at it, so a policy
# sees the same amount of world motion per decision in both. Changing it rescales the per-step
# deltas below, which is why they are derived rather than written out.
CONTROL_HZ = 10.0

# Measured on the real rig: arm joints 29-34 deg/s (0.5 rad/s = 28.6 deg/s, the low end of the
# band), rail max ~7 cm/s from the control UI. The gripper is deliberately quicker, since the jaw
# has to open and close within a single reach.
#
# This is what enforces real speed. Torque limits do NOT: 3 N.m against the servo's damping
# reaches ~4.9 rad/s (~280 deg/s), an order of magnitude past the real arm. The real arm is slow
# because the servo's commanded speed profile limits it, so it has to be imposed explicitly.
REAL_JOINT_SPEED = {          # rad/s, or m/s for the rail
    "dof_slider":    0.07,
    "shoulder_pan":  0.5,
    "shoulder_lift": 0.5,
    "elbow_flex":    0.5,
    "wrist_flex":    0.5,
    "wrist_roll":    0.5,
    "gripper":       2.0,
}

# Per-step position delta at full command. A normalized action a in [-1, 1] moves the target by
# a * JOINT_DELTA_LIMITS[joint] each tick, on both sides.
JOINT_DELTA_LIMITS = {name: speed / CONTROL_HZ for name, speed in REAL_JOINT_SPEED.items()}

# Joint limits from the URDF. Deploy clamps its integrated target to these so a runaway cannot
# command past a joint stop.
JOINT_LIMITS = {
    "dof_slider":    (-0.40286, 0.41714),
    "shoulder_pan":  (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex":    (-1.69, 1.69),
    "wrist_flex":    (-1.65806, 1.65806),
    "wrist_roll":    (-2.74385, 2.84121),
    "gripper":       (-0.17453, 1.74533),
}

# Rest pose: the sim's "rest" keyframe and the pose deploy's reset ramps to.
REST_QPOS = {
    "dof_slider": 0.0,
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.5,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": 1.2,
}

# The six arm joints are Feetech STS3215 servos, stalling around 3 N.m (30 kg*cm at 12 V).
# Capping sim here stops the policy learning pushes the real arm cannot deliver.
STS3215_STALL_TORQUE = 3.0    # N*m
RAIL_FORCE_LIMIT = 100.0      # N; the rail is a separate belt drive, functional not measured

JOINT_FORCE_LIMITS = {
    name: (RAIL_FORCE_LIMIT if name == "dof_slider" else STS3215_STALL_TORQUE)
    for name in JOINT_NAMES
}

assert set(JOINT_NAMES) == set(REAL_JOINT_SPEED) == set(JOINT_LIMITS) == set(REST_QPOS), \
    "every rig table must cover exactly JOINT_NAMES"


def clamp(qpos: dict) -> dict:
    """Clamp a sim-unit joint dict to the joint limits."""
    out = {}
    for name, value in qpos.items():
        lo, hi = JOINT_LIMITS[name]
        out[name] = min(hi, max(lo, float(value)))
    return out
