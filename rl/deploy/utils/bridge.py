"""sim <-> real unit and coordinate bridge.

Sim units are radians for the arm and gripper, metres for the rail. Wire units are degrees for
the arm and gripper, 0..100 for the rail (0 = far from the camera, 100 = near).

Joint order, joint limits, per-step delta caps and the rest pose are NOT defined here -- they
come from ``soframe_policy.rig``, which the training side reads too. This file owns only the
unit conversion, which is deploy-specific.

CALIBRATION STATUS: the rail mapping is derived from geometry. The arm and gripper mappings
assume real zero == sim zero (``OFFSET_REAL`` and ``SIGN`` are identity). Measure each arm
joint's real ``.pos`` at the sim-zero pose into ``OFFSET_REAL`` before trusting a real rollout.
"""
from __future__ import annotations

import math
from typing import Mapping

from soframe_policy import rig

# Wire keys: the follower names every joint "<name>.pos". Order follows rig.JOINT_NAMES, which
# is also portal.yaml's order and the policy's action order.
JOINT_KEYS = tuple(f"{name}.pos" for name in rig.JOINT_NAMES)

_RAD2DEG = 180.0 / math.pi

# real = SIGN * sim * SCALE + OFFSET_REAL  (inverse for real -> sim).
SCALE: dict[str, float] = {f"{n}.pos": _RAD2DEG for n in rig.JOINT_NAMES}
SIGN: dict[str, float] = {k: 1.0 for k in JOINT_KEYS}          # flip per rig
OFFSET_REAL: dict[str, float] = {k: 0.0 for k in JOINT_KEYS}   # real .pos at sim zero

# Rail: normalized 0..100 over its travel. Affine, no sign flip. Calibrated from geometry,
# unlike the arm zeros above.
_RAIL = "dof_slider.pos"
_S_LO, _S_HI = rig.JOINT_LIMITS["dof_slider"]
SCALE[_RAIL] = 100.0 / (_S_HI - _S_LO)
OFFSET_REAL[_RAIL] = -100.0 * _S_LO / (_S_HI - _S_LO)

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
