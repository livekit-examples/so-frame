"""sim <-> real unit and coordinate bridge.

UNIT CONTRACT: sim is radians (arm, gripper) and metres (rail); the wire is degrees (arm,
gripper) and 0..100 for the rail (0 = far from the camera, 100 = near).

Joint order, joint limits, per-step delta caps and the rest pose come from
``soframe_policy.rig``, which training reads too. This file owns only the unit conversion.

``OFFSET_REAL`` is 0 for most joints because the follower runs lerobot's per-motor calibration
(``leslider/configs/calibrate_follower_pos.yaml``), which places real ``.pos`` zero at the URDF zero
pose. ``wrist_roll`` is the measured exception, see below.

Two assumptions to re-check against a live arm with rl/calibrate's ``uv run calibrate`` after
re-homing a servo: that each joint's calibrated zero is the URDF zero pose (an offset), and that
``SIGN`` is identity (a flipped joint drives the arm the wrong way, which no offset would rescue).
"""
from __future__ import annotations

import math
from typing import Mapping

from soframe_policy import rig

# Wire keys: the follower names every joint "<name>.pos". JOINT ORDER is rig.JOINT_NAMES, which
# is also portal.yaml's order and the policy's action order.
JOINT_KEYS = tuple(f"{name}.pos" for name in rig.JOINT_NAMES)

_RAD2DEG = 180.0 / math.pi

# real = SIGN * sim * SCALE + OFFSET_REAL  (inverse for real -> sim).
SCALE: dict[str, float] = {f"{n}.pos": _RAD2DEG for n in rig.JOINT_NAMES}
SIGN: dict[str, float] = {k: 1.0 for k in JOINT_KEYS}          # flip per rig
OFFSET_REAL: dict[str, float] = {k: 0.0 for k in JOINT_KEYS}   # real .pos at sim zero

# Measured on the real arm: at sim wrist_roll 0 the real wrist sat 90 deg round from where sim
# showed it, so this joint's calibrated zero is NOT the URDF zero. The rest of the arm checked out.
OFFSET_REAL["wrist_roll.pos"] = 90.0

# Rail: normalized 0..100 over its travel, affine, no sign flip, calibrated from geometry.
# Its sim low limit is exactly wire 0 by construction, the end of travel you park the
# carriage at to re-zero it.
RAIL = "dof_slider.pos"
_S_LO, _S_HI = rig.JOINT_LIMITS["dof_slider"]
SCALE[RAIL] = 100.0 / (_S_HI - _S_LO)
OFFSET_REAL[RAIL] = -100.0 * _S_LO / (_S_HI - _S_LO)

# Per-step delta caps and the rest pose, keyed by wire name for convenience.
DELTA_LIMIT: dict[str, float] = {f"{n}.pos": v for n, v in rig.JOINT_DELTA_LIMITS.items()}
SIM_REST: dict[str, float] = {f"{n}.pos": v for n, v in rig.REST_QPOS.items()}
SIM_LIMITS: dict[str, tuple[float, float]] = {
    f"{n}.pos": v for n, v in rig.JOINT_LIMITS.items()
}

# PARK is not REST. SIM_REST above is where a training episode starts, so it is what you stage a
# rollout from. PARK is a folded pose measured on the real arm, for leaving it idle: gripper open
# so it drops anything it is holding. Both the robot's reset RPC and the policy's park key read it.
#
# Wire units, because that is what was measured. lift and elbow sit ON their joint limits, so a
# position controller parked here holds current against a mechanical stop.
PARK_REAL: dict[str, float] = {
    RAIL:                 49.0,   # mid-travel (0..100)
    "shoulder_pan.pos":    0.0,
    "shoulder_lift.pos": -100.0,
    "elbow_flex.pos":     96.8,
    "wrist_flex.pos":     88.2,
    "wrist_roll.pos":     89.8,
    "gripper.pos":        68.75,  # ~69% open
}


def real_to_sim(real_state: Mapping[str, float]) -> dict[str, float]:
    """Wire state (deg, rail 0..100) -> sim qpos (rad, m)."""
    return {
        k: (float(real_state[k]) - OFFSET_REAL[k]) / (SIGN[k] * SCALE[k])
        for k in JOINT_KEYS if k in real_state
    }


def sim_to_real(sim_state: Mapping[str, float]) -> dict[str, float]:
    """Sim qpos (rad, m) -> wire targets (deg, rail 0..100)."""
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
