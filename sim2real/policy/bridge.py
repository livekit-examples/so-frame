"""sim <-> real unit/coordinate bridge for the SO-101-on-frame policy.

Sim units: arm/gripper in radians, rail (dof_slider) in metres, zeros at the sim
URDF calibration. Wire units: arm/gripper in degrees; rail NORMALIZED 0..100,
0=far, 100=near camera (matches the leslider follower's slider.pos range). Every
boundary value passes through here so the policy's tensors stay in sim space.

The rail mapping IS calibrated (endpoints measured from sim; see override below).
The arm/gripper joints are NEEDS-CALIBRATION: SCALE is the rad<->deg conversion
but OFFSET_REAL/SIGN are identity, i.e. they assume real joint zero == sim joint
zero, which real servos rarely satisfy. Measure each arm joint's real .pos at the
sim-zero pose into OFFSET_REAL (flip SIGN if reversed). Until then treat any real
rollout as unsafe -- verify with a slow, supervised move first.
"""
from __future__ import annotations

import math
from typing import Mapping

# Joint order MUST match portal.yaml and the sim's active_joints (rail first).
JOINT_KEYS = (
    "dof_slider.pos",
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

_RAD2DEG = 180.0 / math.pi

# real = SIGN * sim * SCALE + OFFSET_REAL  (inverse for real -> sim).
# Arm/gripper: SCALE is rad<->deg; OFFSET_REAL/SIGN are NEEDS-CALIBRATION (identity).
# The rail overrides both below with its measured [0,1] mapping.
SCALE: dict[str, float] = {
    "dof_slider.pos":    1.0,       # placeholder; set to the 0..100 map below
    "shoulder_pan.pos":  _RAD2DEG,  # sim rad -> real deg
    "shoulder_lift.pos": _RAD2DEG,
    "elbow_flex.pos":    _RAD2DEG,
    "wrist_flex.pos":    _RAD2DEG,
    "wrist_roll.pos":    _RAD2DEG,
    "gripper.pos":       _RAD2DEG,
}
SIGN: dict[str, float] = {k: 1.0 for k in JOINT_KEYS}          # flip per rig
OFFSET_REAL: dict[str, float] = {k: 0.0 for k in JOINT_KEYS}   # real .pos at sim zero

# Sim joint limits (rad/m) from the URDF. Clamp the integrated target here so a
# runaway can't command past the joint stop.
SIM_LIMITS: dict[str, tuple[float, float]] = {
    "dof_slider.pos":    (-0.40286, 0.41714),
    "shoulder_pan.pos":  (-1.91986, 1.91986),
    "shoulder_lift.pos": (-1.74533, 1.74533),
    "elbow_flex.pos":    (-1.69, 1.69),
    "wrist_flex.pos":    (-1.65806, 1.65806),
    "wrist_roll.pos":    (-2.74385, 2.84121),
    "gripper.pos":       (-0.17453, 1.74533),
}

# --- Rail (dof_slider): NORMALIZED 0..100, 0=FAR, 100=near camera. ---
# Matches the leslider follower's slider.pos range (0..100). Measured on the sim
# twin: sim MIN (-0.40286) is farthest from the overhead camera -> 0, sim MAX
# (+0.41714) is closest -> 100. Affine, no sign flip: real = 100*(sim-lo)/(hi-lo).
# Calibrated from geometry, unlike the arm zeros.
_S_LO, _S_HI = SIM_LIMITS["dof_slider.pos"]
SCALE["dof_slider.pos"] = 100.0 / (_S_HI - _S_LO)
OFFSET_REAL["dof_slider.pos"] = -100.0 * _S_LO / (_S_HI - _S_LO)

# Per-step delta-position limits (sim units), == so101_on_frame._JOINT_DELTA_LIMITS
# at arm_speed_scale=1.0 (v31 recipe). A normalized action a in [-1,1] moves the
# target by a * DELTA_LIMIT each tick.
DELTA_LIMIT: dict[str, float] = {
    "dof_slider.pos":    0.007,   # matched to the real rail's measured max (~8.9 units/s)
    "shoulder_pan.pos":  0.05,
    "shoulder_lift.pos": 0.05,
    "elbow_flex.pos":    0.05,
    "wrist_flex.pos":    0.05,
    "wrist_roll.pos":    0.05,
    "gripper.pos":       0.2,
}


def real_to_sim(real_state: Mapping[str, float]) -> dict[str, float]:
    """Wire state (deg / mm) -> sim qpos (rad / m)."""
    return {
        k: (float(real_state[k]) - OFFSET_REAL[k]) / (SIGN[k] * SCALE[k])
        for k in JOINT_KEYS if k in real_state
    }


def sim_to_real(sim_state: Mapping[str, float]) -> dict[str, float]:
    """Sim qpos (rad / m) -> wire targets (deg / mm)."""
    return {
        k: SIGN[k] * float(sim_state[k]) * SCALE[k] + OFFSET_REAL[k]
        for k in JOINT_KEYS if k in sim_state
    }


def clamp_sim(sim_state: Mapping[str, float]) -> dict[str, float]:
    """Clamp a sim-unit target to the joint limits."""
    out = {}
    for k in JOINT_KEYS:
        if k not in sim_state:
            continue
        lo, hi = SIM_LIMITS[k]
        out[k] = min(hi, max(lo, float(sim_state[k])))
    return out
