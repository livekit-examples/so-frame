"""Task-specific rewards for the pick-and-place task.

The base reach->bring reward is reused from mjlab. These terms add the rest of the
staged signal: ``grasp_lift_reward`` pays for cube height above the work surface, gated
on the gripper being right at the cube, so it is earned only by closing on the cube and
lifting it; ``transport_reward`` then carries a lifted cube toward the bin; and
``in_bin_bonus`` rewards the final placement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def grasp_lift_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  surface_z: float,
  reach_std: float = 0.05,
  max_lift: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward = (gripper-at-cube) * (cube height above the work surface).

  ``reach_std`` is small so the gating term is ~1 only when the grasp site is
  essentially on the cube; lifting is then rewarded up to ``max_lift`` metres.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach = torch.exp(
    -torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1) / reach_std**2
  )
  lift = (obj_pos_w[:, 2] - surface_z).clamp(min=0.0, max=max_lift)
  return reach * lift


def transport_reward(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  surface_z: float,
  lift_ref: float = 0.03,
  xy_std: float = 0.15,
) -> torch.Tensor:
  """Reward = (cube is lifted) * (cube is horizontally near the bin).

  The lift gate (0->1 as the cube rises ``lift_ref`` off the surface) means only a
  *carried* cube earns this, and the generous ``xy_std`` gives a smooth gradient that
  pulls the lifted cube across the workspace toward the bin.
  """
  obj: Entity = env.scene[object_name]
  command = env.command_manager.get_term(command_name)
  cube = obj.data.root_link_pos_w
  lifted = ((cube[:, 2] - surface_z) / lift_ref).clamp(min=0.0, max=1.0)
  horiz = torch.norm(cube[:, :2] - command.target_pos[:, :2], dim=-1)
  proximity = torch.exp(-torch.square(horiz) / xy_std**2)
  return lifted * proximity


def in_bin_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Milestone reward: 1.0 while the cube is inside the bin (from the command metric)."""
  command = env.command_manager.get_term(command_name)
  return command.metrics["in_bin"]
