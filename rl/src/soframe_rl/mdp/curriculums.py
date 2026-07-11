"""Curriculum terms for the pick-and-place task.

``command_spread_curriculum`` adapts ``PlaceInBinCommand.spread`` (0 = fixed nominal
layout with the cube next to the bin; 1 = full workspace randomization) to the
policy's competence: it raises spread only while a smoothed success rate is high and
lowers it when success drops. Starting at spread 0 with success 0, it naturally holds
the fixed easy layout until the policy learns to place, then widens randomization at
whatever pace the policy can keep up with (ADR-style automatic curriculum), instead of
a fixed schedule that can outrun the policy and erode success.

``reward_weight_ramp_curriculum`` linearly interpolates a reward term's weight across
``(step, weight)`` milestones. mjlab's built-in ``reward_curriculum`` applies stages
as discrete jumps; an abrupt several-fold change to a penalty can yank the reward
landscape out from under PPO and collapse a trained policy. A per-step linear ramp
changes the landscape slowly enough for the policy to track it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class command_spread_curriculum:
  """Performance-gated randomization coefficient.

  Params (via ``CurriculumTermCfg.params``):
    command_name: the PlaceInBin command to drive.
    up_threshold: raise spread while smoothed success >= this.
    down_threshold: lower spread while smoothed success <= this.
    step: per-env-step change in spread when raising/lowering.
    ema_alpha: smoothing for the success estimate (per env step).
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._succ_ema: float = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    up_threshold: float = 0.5,
    down_threshold: float = 0.25,
    step: float = 1.0e-4,
    ema_alpha: float = 0.01,
  ) -> dict[str, Any]:
    del env_ids, command_name
    command = env.command_manager.get_term(self._command_name)
    success = float(command.metrics["episode_success"].mean().item())
    self._succ_ema += ema_alpha * (success - self._succ_ema)

    spread = float(command.spread)  # type: ignore[attr-defined]
    if self._succ_ema >= up_threshold:
      spread = min(1.0, spread + step)
    elif self._succ_ema <= down_threshold:
      spread = max(0.0, spread - step)
    command.spread = spread  # type: ignore[attr-defined]

    return {
      "spread": torch.tensor(spread),
      "success_ema": torch.tensor(self._succ_ema),
    }


class reward_weight_ramp_curriculum:
  """Linearly interpolate a reward term's weight across ``(step, weight)`` milestones.

  Unlike discrete stage jumps, the weight changes a tiny amount every env step, so
  the policy adapts continuously instead of facing a reward-landscape cliff.

  Example::

    CurriculumTermCfg(
      func=reward_weight_ramp_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "milestones": [
          {"step": 3000 * 24, "weight": -0.01},  # flat before the first milestone
          {"step": 7000 * 24, "weight": -0.15},  # linear ramp in between
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["reward_name"])
    self._milestones = sorted(cfg.params["milestones"], key=lambda m: m["step"])

  def _weight_at(self, step: int) -> float:
    ms = self._milestones
    if step <= ms[0]["step"]:
      return float(ms[0]["weight"])
    if step >= ms[-1]["step"]:
      return float(ms[-1]["weight"])
    for a, b in zip(ms, ms[1:]):
      if a["step"] <= step < b["step"]:
        t = (step - a["step"]) / max(b["step"] - a["step"], 1)
        return float(a["weight"] + t * (b["weight"] - a["weight"]))
    return float(ms[-1]["weight"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_name: str,
    milestones: list[dict],
  ) -> dict[str, Any]:
    del env_ids, reward_name, milestones
    self._term_cfg.weight = self._weight_at(env.common_step_counter)
    return {"weight": torch.tensor(self._term_cfg.weight)}
