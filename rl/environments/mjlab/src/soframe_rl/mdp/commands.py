"""PlaceInBin command: randomize the cube and the bin, goal = cube in the bin.

Modeled on mjlab's ``LiftingCommand`` (mjlab/tasks/manipulation/mdp/commands.py),
but instead of sampling an abstract 3D target it:

  * samples a bin (x, y) on the plane and teleports the (fixed/mocap) bin there,
  * samples a cube (x, y) elsewhere (rejecting placements too close to the bin)
    and teleports the (free) cube there,
  * exposes ``target_pos`` = where the cube comes to rest on the bin floor, so
    dropping the cube in raises the reused mjlab reach/bring rewards. An
    above-the-rim goal would fall off as the cube settles.

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

    # Curriculum spread in [0, 1]: 0 = fixed nominal layout (cube next to bin),
    # 1 = full workspace randomization. Set by the curriculum term each step.
    self.spread = float(cfg.initial_spread)

    # Horizontal cube->bin distance and its per-step reduction (for the
    # potential-based transport reward: only *moving closer* pays).
    self.cube_bin_h = torch.zeros(self.num_envs, device=self.device)
    self.carry_progress = torch.zeros(self.num_envs, device=self.device)

    self.metrics["cube_to_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["in_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["spread"] = torch.zeros(self.num_envs, device=self.device)

  # -- CommandTerm API -------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    cube_pos = self.cube.data.root_link_pos_w
    horiz = torch.norm(cube_pos[:, :2] - self.bin_pos[:, :2], dim=-1)
    self.metrics["cube_to_bin"] = torch.norm(self.target_pos - cube_pos, dim=-1)

    # Per-step reduction in horizontal cube->bin distance (potential shaping).
    self.carry_progress = self.cube_bin_h - horiz
    self.cube_bin_h = horiz

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
    self.metrics["spread"][:] = self.spread

  def compute_success(self) -> torch.Tensor:
    return self.metrics["in_bin"] > 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    origins = self._env.scene.env_origins[env_ids]
    r = self.cfg

    c = self.spread
    lo = torch.tensor([r.workspace_x[0], r.workspace_y[0]], device=self.device)
    hi = torch.tensor([r.workspace_x[1], r.workspace_y[1]], device=self.device)
    bin_nom = torch.tensor(r.bin_nominal, device=self.device)
    cube_nom = torch.tensor(r.cube_nominal, device=self.device)
    dev = torch.tensor(r.spread_xy, device=self.device)  # max deviation at spread=1.

    def _sample_around(nom: torch.Tensor) -> torch.Tensor:
      # nominal + spread-scaled uniform deviation, clamped to the workspace.
      xy = nom + c * dev * sample_uniform(-1.0, 1.0, (n, 2), device=self.device)
      return torch.clamp(xy, lo, hi)

    # spread=0 -> both at their (fixed) nominals, cube a short hop from the bin.
    bin_xy = _sample_around(bin_nom)
    cube_xy = _sample_around(cube_nom)
    if c > 0.0:  # only meaningful once positions vary.
      for _ in range(r.max_reject_iters):
        too_close = torch.norm(cube_xy - bin_xy, dim=-1) < r.min_separation
        if not too_close.any():
          break
        cube_xy[too_close] = _sample_around(cube_nom)[too_close]

    z0 = torch.zeros(n, device=self.device)
    surf = r.surface_z
    bin_z = torch.full((n,), surf, device=self.device)  # bin base on the work floor.
    cube_z = surf + sample_uniform(r.cube_z[0], r.cube_z[1], (n,), device=self.device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device=self.device)

    bin_pos_w = torch.stack([bin_xy[:, 0], bin_xy[:, 1], bin_z], dim=-1) + origins
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

    # Cache bin pose; goal = cube resting inside the bin (on the bin floor). Placing
    # the cube in now *increases* the place reward, unlike an above-the-rim goal.
    self.bin_pos[env_ids] = bin_pos_w
    target = bin_pos_w.clone()
    target[:, 2] += 2.0 * assets.BIN_WALL_HALF + assets.CUBE_HALF_SIZE
    self.target_pos[env_ids] = target

    # Reset transport tracking for these envs (no spurious progress at reset).
    self.cube_bin_h[env_ids] = torch.norm(cube_xy - bin_xy, dim=-1)
    self.carry_progress[env_ids] = 0.0

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

  # Height (world z) of the work surface the bin sits on and the cube spawns above.
  surface_z: float = 0.083

  # Workspace bounds (env-relative) that sampled positions are clamped to. The arm base sits
  # near (0, -0.49) env-relative; these bracket a patch in front of it.
  # !!! THESE ARE ESTIMATES. Validate reachability in the viewer and adjust. !!!
  workspace_x: tuple[float, float] = (-0.35, -0.10)
  workspace_y: tuple[float, float] = (-0.65, -0.30)
  cube_z: tuple[float, float] = (0.02, 0.04)  # spawn clearance above surface_z.

  # Curriculum: at spread=0 the bin and cube sit at these fixed nominals (cube a
  # short hop from the bin -> easy place); spread in (0,1] scales uniform deviation
  # up to +/- spread_xy, reaching full workspace coverage at spread=1.
  initial_spread: float = 0.0
  bin_nominal: tuple[float, float] = (-0.225, -0.475)
  cube_nominal: tuple[float, float] = (-0.225, -0.315)  # ~0.16 m from bin in +y.
  spread_xy: tuple[float, float] = (0.12, 0.16)

  # Keep the cube from spawning under/next to the bin (applied once spread>0).
  min_separation: float = 0.12
  max_reject_iters: int = 10

  def build(self, env: ManagerBasedRlEnv) -> PlaceInBinCommand:
    return PlaceInBinCommand(self, env)
