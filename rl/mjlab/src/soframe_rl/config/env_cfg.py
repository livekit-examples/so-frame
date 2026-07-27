"""SO-101-specific specialization of the pick-and-place env config."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from ..mdp.actions import TargetRelativeJointPositionActionCfg

from soframe_rl.assets import get_bin_spec, get_cube_spec
from soframe_rl.pick_place_env_cfg import make_pick_place_env_cfg
from soframe_rl.so101_constants import (
  GRASP_SITE,
  SO101_ACTION_SCALE,
  WORK_SURFACE_Z,
  get_so101_robot_cfg,
)
from soframe_rl.train_params import PARAMS


def so101_pick_place_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_pick_place_env_cfg()
  cfg.scene.num_envs = PARAMS["runner"]["num_envs"]

  # Entities: imported (unmodified) robot + programmatic cube + bin.
  cfg.scene.entities = {
    "robot": get_so101_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
    "bin": EntityCfg(spec_fn=get_bin_spec),
  }

  # Per-actuator delta-position action scale (measured real joint speed / CONTROL_HZ).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, TargetRelativeJointPositionActionCfg)
  joint_pos_action.scale = SO101_ACTION_SCALE

  # Point the end-effector observation + rewards at the grasp site.
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    GRASP_SITE,
  )
  cfg.rewards["reach_and_bring"].params["asset_cfg"].site_names = (GRASP_SITE,)
  cfg.rewards["grasp_lift"].params["asset_cfg"].site_names = (GRASP_SITE,)

  # Objects rest on the lightbox bottom panel (the added collision pad height).
  cfg.rewards["grasp_lift"].params["surface_z"] = WORK_SURFACE_Z
  cfg.rewards["transport"].params["surface_z"] = WORK_SURFACE_Z
  cfg.commands["place"].surface_z = WORK_SURFACE_Z

  cfg.viewer.body_name = "gripper_link"

  if play:
    cfg.scene.num_envs = 16
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # No curriculum in play -> show the hard case: full randomization.
    cfg.commands["place"].initial_spread = 1.0

  return cfg
