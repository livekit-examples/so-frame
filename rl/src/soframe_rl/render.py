"""Fleet render: roll out a trained policy across many environments and render a
camera move that starts on one robot and pulls back to the whole grid.

Both a hero shot ("hundreds of arms training in parallel") and a debug tool (watch
what one robot actually does). All ``num_envs`` robots live in one MuJoCo world on a
grid, so this is a scripted free camera over a policy rollout: the focus env is drawn
in full and every other env's moving parts are stamped in with ``mjv_addGeoms``.

Usage (from rl/, on a GPU machine):

    uv run soframe-render --checkpoint-file <path/to/model_XXXX.pt> \
        --num-envs 256 --seconds 14 --out fleet.mp4
"""

from __future__ import annotations

import os

# Headless GL for offscreen rendering; must be set before mujoco imports a context.
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import asdict, dataclass  # noqa: E402

import mediapy  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import soframe_rl  # noqa: E402,F401  (registers the task)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import (  # noqa: E402
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)


@dataclass
class Args:
  checkpoint_file: str
  """Path to a trained checkpoint (model_XXXX.pt)."""
  task: str = "Mjlab-Pick-Place-Bin-SO101"
  num_envs: int = 400
  seconds: float = 14.0
  fps: int = 30
  width: int = 1280
  height: int = 720
  out: str = "fleet.mp4"
  device: str = "cuda:0"
  # Camera: fixed viewpoint behind the arm, easing straight back from the center
  # robot to a wide, filled field of robots that runs off the frame edges.
  azimuth: float = 180.0
  """Fixed camera azimuth (no panning/orbit); 180 sits behind the arm, looking in."""
  elevation_close: float = -20.0
  elevation_wide: float = -14.0
  wide_dist_frac: float = 0.28
  """Wide-shot distance as a fraction of the grid radius. <1 keeps the grid
  overflowing the frame (robots to every edge) so it reads as endless."""


def _ease(t: float) -> float:
  """Smootherstep (ease-in-out): slow start and slow finish, no abrupt motion."""
  t = float(np.clip(t, 0.0, 1.0))
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _add_skybox(spec: mujoco.MjSpec) -> None:
  """Add a gradient skybox so the background isn't black at low camera angles.

  mjlab's plane scene has no skybox, so anything above the horizon renders black.
  """
  if any(t.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX for t in spec.textures):
    return
  spec.add_texture(
    name="render_skybox",
    type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
    rgb1=[0.45, 0.6, 0.8],
    rgb2=[0.1, 0.13, 0.2],
    width=512,
    height=512,
  )


def main() -> None:
  args = tyro.cli(Args)

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.scene.spec_fn = _add_skybox  # nice sky instead of a black background.
  agent_cfg = load_rl_cfg(args.task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    args.checkpoint_file, load_cfg={"actor": True}, strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  # Host model/data for rendering; big geom budget for the whole fleet.
  model = env.sim.mj_model
  # Size the offscreen framebuffer to the requested resolution, else MuJoCo renders
  # into a smaller default buffer and letterboxes the rest with a black bar.
  model.vis.global_.offwidth = args.width
  model.vis.global_.offheight = args.height
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(
    model, height=args.height, width=args.width,
    max_geom=args.num_envs * model.ngeom + 1000,
  )
  opt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()
  dyn = mujoco.mjtCatBit.mjCAT_DYNAMIC.value

  cam = mujoco.MjvCamera()
  mujoco.mjv_defaultFreeCamera(model, cam)

  # Grid geometry from the actual env origins.
  origins = env.scene.env_origins.detach().cpu().numpy()
  center = origins.mean(axis=0)
  radius = float(np.linalg.norm(origins[:, :2] - center[:2], axis=1).max())
  # Focus on the env nearest the grid center (not a corner).
  focus = int(np.argmin(np.linalg.norm(origins[:, :2] - center[:2], axis=1)))

  ws = np.array([-0.15, -0.45, 0.08])  # rough workspace offset within an env.
  close_look = origins[focus] + ws
  wide_look = center.copy()
  close_dist = 1.1
  wide_dist = args.wide_dist_frac * radius + 2.0

  # Larger extent so the free camera's near/far planes and lighting cover the grid.
  model.stat.extent = max(float(model.stat.extent), 2.0 * radius + 1.0)
  model.light_castshadow[:] = False  # shadows over a huge grid are costly + noisy.

  n_frames = int(args.seconds * args.fps)
  frames: list[np.ndarray] = []
  obs, _ = wrapped.reset()

  for f in range(n_frames):
    with torch.inference_mode():
      action = policy(obs)
    obs, _, _, _ = wrapped.step(action)

    # Camera: fixed azimuth, one smooth eased dolly-back over the whole clip.
    s = _ease(f / max(n_frames - 1, 1))
    cam.lookat[:] = (1 - s) * close_look + s * wide_look
    cam.distance = (1 - s) * close_dist + s * wide_dist
    cam.elevation = (1 - s) * args.elevation_close + s * args.elevation_wide
    cam.azimuth = args.azimuth

    # Pull all envs' state to host once, then draw.
    qpos = env.sim.data.qpos.detach().cpu().numpy()
    qvel = env.sim.data.qvel.detach().cpu().numpy()
    has_mocap = model.nmocap > 0
    if has_mocap:
      mpos = env.sim.data.mocap_pos.detach().cpu().numpy()
      mquat = env.sim.data.mocap_quat.detach().cpu().numpy()

    def _set(i: int) -> None:
      data.qpos[:] = qpos[i]
      data.qvel[:] = qvel[i]
      if has_mocap:
        data.mocap_pos[:] = mpos[i]
        data.mocap_quat[:] = mquat[i]
      mujoco.mj_forward(model, data)

    # Focus env: full scene (incl. its frame + ground). Others: moving parts only.
    _set(focus)
    renderer.update_scene(data, camera=cam)
    for i in range(args.num_envs):
      if i == focus:
        continue
      _set(i)
      mujoco.mjv_addGeoms(model, data, opt, pert, dyn, renderer.scene)
    frames.append(renderer.render())

    if (f + 1) % 30 == 0 or f == n_frames - 1:
      print(f"rendered {f + 1}/{n_frames} frames", flush=True)

  renderer.close()
  mediapy.write_video(args.out, frames, fps=args.fps)
  print(f"wrote {args.out} ({len(frames)} frames, {args.width}x{args.height})")


if __name__ == "__main__":
  main()
