"""Observation terms specific to the pick-and-place task.

Only ``cube_to_goal_distance`` is needed: mjlab's ``object_to_goal_distance``
hard-asserts a ``LiftingCommand``, so we provide a command-agnostic version that
reads ``target_pos`` from any command term. The end-effector-to-cube distance is
reused directly from mjlab (``manipulation.mdp.ee_to_object_distance``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def cube_to_goal_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from the object to the command's target, in the robot base frame."""
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  distance_vec_w = command.target_pos - obj.data.root_link_pos_w
  base_quat_w = robot.data.root_link_quat_w
  return quat_apply(quat_inv(base_quat_w), distance_vec_w)
