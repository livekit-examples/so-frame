"""Task-specific MDP terms for the SO-101 pick-and-place-into-bin task.

Most terms (rewards, observations) are reused directly from mjlab's manipulation
task; only the command — which randomizes both the cube and the bin and defines
the goal as "cube inside the bin" — is custom.
"""

from soframe_rl.mdp.commands import (
  PlaceInBinCommand as PlaceInBinCommand,
)
from soframe_rl.mdp.commands import (
  PlaceInBinCommandCfg as PlaceInBinCommandCfg,
)
from soframe_rl.mdp.observations import (
  cube_to_goal_distance as cube_to_goal_distance,
)
from soframe_rl.mdp.rewards import (
  grasp_lift_reward as grasp_lift_reward,
)
