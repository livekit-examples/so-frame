"""Base env config for the SO-101 pick-a-cube-place-in-bin task.

Robot-agnostic wiring of the mjlab managers. It reuses mjlab's manipulation
rewards/observations wherever possible and only swaps in our ``PlaceInBin``
command and the command-agnostic ``cube_to_goal`` observation. The SO-101-specific
bits (entities, action scale, grasp-site name) are filled in by
``config/env_cfg.py``.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from soframe_rl import mdp as task_mdp

# Effectively "never resample mid-episode": cube+bin are randomized only on reset.
_NO_MIDEPISODE_RESAMPLE = (1.0e9, 1.0e9)


def make_pick_place_env_cfg() -> ManagerBasedRlEnvCfg:
  # -- Observations --------------------------------------------------------
  actor_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "ee_to_cube": ObservationTermCfg(
      func=manipulation_mdp.ee_to_object_distance,
      params={
        "object_name": "cube",
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # set per-robot.
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "cube_to_goal": ObservationTermCfg(
      func=task_mdp.cube_to_goal_distance,
      params={"object_name": "cube", "command_name": "place"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
  }

  # -- Actions (TRUE delta joint position; scale set per-robot) ------------
  #
  # RelativeJointPositionActionCfg gives target = current_joint_pos + action * scale, so `scale`
  # is a hard per-step motion cap and the real servo's speed is enforced structurally.
  #
  # This was JointPositionActionCfg(use_default_offset=True), which is target = action * scale +
  # default_joint_pos -- an ABSOLUTE target measured from the home pose, not a delta. Nothing
  # bounded per-step motion: one step could command the full +-scale swing from home, and
  # consecutive steps could cross the whole 2*scale range instantly. The comment here called it
  # "delta joint position", which is what it was standing in for but never was. The smoothness
  # penalties in `rewards` below existed to paper over exactly that, and are gone with it.
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.05,  # overridden per-robot from SO101_ACTION_SCALE (measured real speed / Hz).
    )
  }

  # -- Command: randomize cube + bin, goal = cube in bin -------------------
  commands: dict[str, CommandTermCfg] = {
    "place": task_mdp.PlaceInBinCommandCfg(
      cube_name="cube",
      bin_name="bin",
      resampling_time_range=_NO_MIDEPISODE_RESAMPLE,
      debug_vis=True,
    )
  }

  # -- Events --------------------------------------------------------------
  # NOTE: cube/bin placement is handled by the command's reset resample.
  # Fingertip-friction domain randomization is intentionally omitted here
  # because it needs named gripper collision geoms (a small XML follow-up);
  # add it once those exist for sim-to-real robustness.
  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  # -- Rewards (staged: reach -> bring cube to bin, + regularizers) --------
  rewards = {
    "reach_and_bring": RewardTermCfg(
      func=manipulation_mdp.staged_position_reward,
      weight=1.0,
      params={
        "command_name": "place",
        "object_name": "cube",
        "reaching_std": 0.2,
        "bringing_std": 0.3,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # set per-robot.
      },
    ),
    "place_precise": RewardTermCfg(
      func=manipulation_mdp.bring_object_reward,
      weight=5.0,
      params={"command_name": "place", "object_name": "cube", "std": 0.05},
    ),
    # Pays for lifting the cube only while gripping it (forms the grasp).
    "grasp_lift": RewardTermCfg(
      func=task_mdp.grasp_lift_reward,
      weight=15.0,
      params={
        "object_name": "cube",
        "surface_z": 0.083,  # overridden per-robot.
        "reach_std": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # set per-robot.
      },
    ),
    # Potential-based carry: rewards a lifted cube for moving closer to the bin.
    "transport": RewardTermCfg(
      func=task_mdp.transport_reward,
      weight=40.0,
      params={
        "object_name": "cube",
        "command_name": "place",
        "surface_z": 0.083,  # overridden per-robot.
        "lift_ref": 0.06,  # ~rim height: clear the rim before carry pays.
      },
    ),
    # Milestone: cube actually inside the bin.
    "in_bin_bonus": RewardTermCfg(
      func=task_mdp.in_bin_bonus,
      weight=10.0,
      params={"command_name": "place"},
    ),
    # No speed or smoothness penalties. They were standing in for a rate limit the action space
    # did not have: joint_vel_hinge's max_vel was 0.5, literally the real arm's 0.5 rad/s, and
    # action_rate_l2 taxed jerk. Both are now enforced structurally by the relative action
    # space's per-step cap, which needs no weight tuned against the task rewards.
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  # Curriculum. `common_step_counter` increments once per env step, i.e.
  # iterations * num_steps_per_env (24). Thresholds below are in that unit
  # (iteration count shown in comments), independent of num_envs.
  curriculum = {
    # Placement curriculum: hold a fixed cube-next-to-bin layout while the policy
    # learns pick->carry->place, then ramp to full workspace randomization.
    "placement_spread": CurriculumTermCfg(
      func=task_mdp.command_spread_curriculum,
      params={
        "command_name": "place",
        # Performance-gated: hold the fixed layout until the policy places well,
        # then widen randomization only as fast as success allows (back off if it
        # drops). Self-paces to competence instead of a fixed schedule.
        "up_threshold": 0.4,
        "down_threshold": 0.2,
        "step": 1.0e-4,
        "ema_alpha": 0.01,
      },
    ),
    # The smoothness-penalty ramp that used to live here is gone with the penalty it tuned.
    # Its own comment was the argument against the approach: the weight had to be ramped
    # linearly because a step change "yanks the reward landscape and can collapse a trained
    # policy", and it had to stay under ~-0.06 or "the penalty starts outbidding the task
    # rewards and carrying stops paying". A per-step cap in the action space has no weight, no
    # ramp, and no window to fall out of.
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # set per-robot.
      distance=1.2,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=60,
      njmax=600,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    # 0.005 s timestep x 20 = 10 Hz control, matching so101_constants.CONTROL_HZ, the maniskill
    # twin, and the deploy loop's 10 Hz portal tick. This was decimation=4 (50 Hz): the action
    # cadence a policy trains at is part of the sim2real contract, and a 50 Hz policy driven at
    # 10 Hz on hardware sees 5x its trained step size per decision.
    decimation=20,
    # 30 s at 10 Hz. The real-servo speeds need this much runway to reach, carry and place;
    # the maniskill twin uses the same 300-step horizon.
    episode_length_s=30.0,
  )
