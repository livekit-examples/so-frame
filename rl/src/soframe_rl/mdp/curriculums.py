"""Curriculum terms for the pick-and-place task.

``command_spread_curriculum`` ramps the ``PlaceInBinCommand.spread`` from 0 (fixed
nominal layout, cube next to the bin — easy) toward 1 (full workspace randomization)
as training progresses, linearly interpolating between step milestones. This lets the
policy first learn the whole pick->carry->place motion on a fixed scene, then
generalize to randomized cube/bin positions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class SpreadStage(TypedDict):
  step: int
  spread: float


class command_spread_curriculum:
  """Linearly ramp a command's ``spread`` across ``(step, spread)`` milestones.

  Example::

    CurriculumTermCfg(
      func=command_spread_curriculum,
      params={
        "command_name": "place",
        "stages": [
          {"step": 0,      "spread": 0.0},
          {"step": 30000,  "spread": 0.0},   # hold fixed while it learns to place
          {"step": 120000, "spread": 1.0},   # then ramp to full randomization
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._command_name: str = cfg.params["command_name"]
    self._stages: list[SpreadStage] = sorted(
      cfg.params["stages"], key=lambda s: s["step"]
    )

  def _spread_at(self, step: int) -> float:
    stages = self._stages
    if step <= stages[0]["step"]:
      return float(stages[0]["spread"])
    if step >= stages[-1]["step"]:
      return float(stages[-1]["spread"])
    for a, b in zip(stages, stages[1:]):
      if a["step"] <= step < b["step"]:
        span = max(b["step"] - a["step"], 1)
        t = (step - a["step"]) / span
        return float(a["spread"] + t * (b["spread"] - a["spread"]))
    return float(stages[-1]["spread"])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[SpreadStage],
  ) -> dict[str, Any]:
    del env_ids, command_name, stages
    command = env.command_manager.get_term(self._command_name)
    command.spread = self._spread_at(env.common_step_counter)  # type: ignore[attr-defined]
    return {"spread": torch.tensor(command.spread)}
