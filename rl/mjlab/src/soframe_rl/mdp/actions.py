"""Delta-from-target joint position action.

mjlab ships two joint-position terms and neither matches how this rig is driven:

- ``JointPositionActionCfg``: ``target = action * scale + default_qpos``. An ABSOLUTE target
 measured from the home pose, so nothing bounds per-step motion.
- ``RelativeJointPositionActionCfg``: ``target = current_qpos + action * scale``. A real delta,
 but measured from where the joint actually IS.

The maniskill twin uses ``pd_joint_target_delta_pos`` and the deploy loop integrates the same
way: the delta is added to the PREVIOUS TARGET, not to the measured position. The difference
matters under load. If a joint lags its target -- carrying the cube, pressing into the bin, or
just at the servo's speed limit -- delta-from-current can never get ahead of the lag, so the
commanded target collapses onto the measured pose and the arm stalls. Delta-from-target keeps
integrating, exactly as the real servo's position loop is driven.

  target_t = clamp(target_{t-1} + action * scale, soft joint limits)

On reset the running target is reseeded from the measured pose, so the first action of an episode
is a no-jump delta -- the same thing sim2real/policy/run.py does when it (re)claims control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.actuator.actuator import TransmissionType


class TargetRelativeJointPositionAction(BaseAction):
  """Integrate the action into a running joint-position target."""

  def __init__(self, cfg: "TargetRelativeJointPositionActionCfg", env) -> None:
    super().__init__(cfg=cfg, env=env)
    self._target = torch.zeros_like(self._raw_actions)
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self._lower, self._upper = limits[..., 0], limits[..., 1]
    # Seed before the first step, in case apply_actions runs before any reset.
    self._target[:] = self._entity.data.joint_pos[:, self._target_ids]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    # Reseed from the measured pose so a new episode starts with no commanded jump.
    current = self._entity.data.joint_pos[:, self._target_ids]
    if env_ids is None or isinstance(env_ids, slice):
      self._target[:] = current
    else:
      self._target[env_ids] = current[env_ids]

  def apply_actions(self) -> None:
    # Integrate, then clamp: a runaway policy cannot wind the target past a joint stop, and
    # the clamp keeps it from accumulating an unreachable offset it would have to unwind.
    self._target = torch.clamp(self._target + self._processed_actions,
                 self._lower, self._upper)
    self._entity.set_joint_position_target(self._target, joint_ids=self._target_ids)

  @property
  def target(self) -> torch.Tensor:
    """The running target, for observation terms that expose it (deploy feeds it as proprio)."""
    return self._target


@dataclass(kw_only=True)
class TargetRelativeJointPositionActionCfg(BaseActionCfg):
  """``target = previous_target + action * scale``, clamped to the soft joint limits.

  ``scale`` is therefore a hard per-step motion cap, in joint units. Set it from measured real
  joint speed divided by the control rate -- see soframe_policy.rig.JOINT_DELTA_LIMITS.
  """

  def __post_init__(self):
    self.transmission_type = TransmissionType.JOINT
    if self.offset != 0.0:
      raise ValueError(
        "TargetRelativeJointPositionActionCfg does not support 'offset'. The target is "
        "previous_target + action * scale; a fixed offset has no meaning."
      )

  def build(self, env) -> TargetRelativeJointPositionAction:
    return TargetRelativeJointPositionAction(self, env)
