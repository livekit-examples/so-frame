"""sim <-> real unit and coordinate bridge.

UNIT CONTRACT: sim is radians (arm, gripper) and metres (rail); the wire is degrees (arm,
gripper) and 0..100 for the rail (0 = far from the camera, 100 = near).

Joint order, joint limits, per-step delta caps and the rest pose come from
``soframe_policy.rig``, which training reads too. This file owns only the unit conversion.

``OFFSET_REAL`` is 0 for the arm and gripper joints by design, not by omission: the follower runs
lerobot's per-motor calibration (``leslider/configs/calibrate_follower_pos.yaml``), which already
places real ``.pos`` zero at the same physical pose as the URDF zero. So real == sim after the
unit scale, and there is nothing to measure here.

That leaves two assumptions, both cheap to check with ``debug_policy.py --bridge`` against a live
arm: the calibrated zero really is the URDF zero pose, and ``SIGN`` is identity (a flipped joint
would drive the arm the wrong way, which no offset would rescue). Re-check after re-homing a servo.
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
