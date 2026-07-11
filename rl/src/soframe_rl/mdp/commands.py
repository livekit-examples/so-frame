"""PlaceInBin command: randomize the cube and the bin, goal = cube in the bin.

Modeled on mjlab's ``LiftingCommand`` (mjlab/tasks/manipulation/mdp/commands.py),
but instead of sampling an abstract 3D target it:

  * samples a bin (x, y) on the plane and teleports the (fixed/mocap) bin there,
  * samples a cube (x, y) elsewhere (rejecting placements too close to the bin)
    and teleports the (free) cube there,
  * exposes ``target_pos`` = a point just above the bin opening, so the reused
    mjlab reach/bring rewards drive the cube up over the rim and into the bin.

Success = cube horizontally inside the bin footprint and below the rim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from soframe_rl import assets

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class PlaceInBinCommand(CommandTerm):
  cfg: PlaceInBinCommandCfg

  def __init__(self, cfg: PlaceInBinCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.cube: Entity = env.scene[cfg.cube_name]
    self.bin: Entity = env.scene[cfg.bin_name]

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.bin_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["cube_to_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  # -- CommandTerm API -------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    cube_pos = self.cube.data.root_link_pos_w
    horiz = torch.norm(cube_pos[:, :2] - self.bin_pos[:, :2], dim=-1)
    self.metrics["cube_to_bin"] = torch.norm(self.target_pos - cube_pos, dim=-1)

    # Cube center inside the footprint and below the rim = in the bin.
    rim_top = self.bin_pos[:, 2] + 2.0 * assets.BIN_WALL_HALF + assets.BIN_RIM_HEIGHT
    inside = (
      (horiz < (assets.BIN_INNER_HALF - assets.CUBE_HALF_SIZE))
      & (cube_pos[:, 2] < rim_top)
      & (cube_pos[:, 2] > self.bin_pos[:, 2])
    ).float()
    self.episode_success = torch.maximum(self.episode_success, inside)
    self.metrics["in_bin"] = inside
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["in_bin"] > 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    origins = self._env.scene.env_origins[env_ids]
    r = self.cfg

    lo = torch.tensor([r.workspace_x[0], r.workspace_y[0]], device=self.device)
    hi = torch.tensor([r.workspace_x[1], r.workspace_y[1]], device=self.device)

    # Sample bin + cube xy, rejecting cube placements too close to the bin.
    bin_xy = sample_uniform(lo, hi, (n, 2), device=self.device)
    cube_xy = sample_uniform(lo, hi, (n, 2), device=self.device)
    for _ in range(r.max_reject_iters):
      too_close = torch.norm(cube_xy - bin_xy, dim=-1) < r.min_separation
      if not too_close.any():
        break
      cube_xy[too_close] = sample_uniform(lo, hi, (n, 2), device=self.device)[too_close]

    z0 = torch.zeros(n, device=self.device)
    cube_z = sample_uniform(r.cube_z[0], r.cube_z[1], (n,), device=self.device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)

    bin_pos_w = torch.stack([bin_xy[:, 0], bin_xy[:, 1], z0], dim=-1) + origins
    cube_pos_w = torch.stack([cube_xy[:, 0], cube_xy[:, 1], cube_z], dim=-1) + origins
    bin_quat = quat_from_euler_xyz(z0, z0, z0)  # upright, identity.
    cube_quat = quat_from_euler_xyz(z0, z0, yaw)

    # Bin is fixed (mocap) -> mocap pose; cube is free -> pose + zero velocity.
    self.bin.write_mocap_pose_to_sim(
      torch.cat([bin_pos_w, bin_quat], dim=-1), env_ids=env_ids
    )
    self.cube.write_root_link_pose_to_sim(
      torch.cat([cube_pos_w, cube_quat], dim=-1), env_ids=env_ids
    )
    self.cube.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )

    # Cache bin pose; goal = a point just above the bin opening.
    self.bin_pos[env_ids] = bin_pos_w
    target = bin_pos_w.clone()
    target[:, 2] += (
      2.0 * assets.BIN_WALL_HALF + assets.BIN_RIM_HEIGHT + assets.CUBE_HALF_SIZE
    )
    self.target_pos[env_ids] = target

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=0.02,
        color=(0.2, 0.9, 0.2, 0.6),
        label=f"bin_target_{batch}",
      )


@dataclass(kw_only=True)
class PlaceInBinCommandCfg(CommandTermCfg):
  cube_name: str = "cube"
  bin_name: str = "bin"

  # Workspace sampling ranges (env-relative), on the plane. VALIDATE in viewer.
  workspace_x: tuple[float, float] = (-0.35, -0.10)
  workspace_y: tuple[float, float] = (-0.65, -0.30)
  cube_z: tuple[float, float] = (0.02, 0.03)

  # Keep the cube from spawning under/next to the bin.
  min_separation: float = 0.14
  max_reject_iters: int = 10

  def build(self, env: ManagerBasedRlEnv) -> PlaceInBinCommand:
    return PlaceInBinCommand(self, env)
