"""Registers the SO-101 pick-and-place task with mjlab's task registry.

Imported for its side effect by ``soframe_rl/__init__.py``.
"""

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from soframe_rl.config.env_cfg import so101_pick_place_env_cfg
from soframe_rl.config.rl_cfg import so101_pick_place_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Pick-Place-Bin-SO101",
  env_cfg=so101_pick_place_env_cfg(),
  play_env_cfg=so101_pick_place_env_cfg(play=True),
  rl_cfg=so101_pick_place_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
