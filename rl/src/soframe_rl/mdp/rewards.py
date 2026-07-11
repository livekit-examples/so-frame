"""Task-specific rewards for the pick-and-place task.

The base reach->bring reward is reused from mjlab; the missing piece that left the
first policy stuck at "reach only" is a signal for actually grasping and lifting the
cube. ``grasp_lift_reward`` provides it: it pays for cube height above the work
surface, gated on the gripper being right at the cube, so the only way to earn it is
to close on the cube and lift it.
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
