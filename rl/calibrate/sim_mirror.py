"""Render the sim cameras at a given qpos, so the real cameras can be fitted against them live."""
from __future__ import annotations

import os
import pathlib
import platform

# Real camera track -> the sim camera it is calibrated against. The arm camera is the wrist camera
# under another name.
REAL_TO_SIM_CAM = {
    "arm_camera": "wrist_camera",
    "overhead_camera": "overhead_camera",
}

_MOLTENVK_ICD = pathlib.Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json")


class SimMirror:
    """One sim env, rendered on demand. Renders are cached on the pose, so holding one is free."""

    def __init__(self, size: int, show_objects: bool = False):
        if platform.system() == "Darwin" and not os.environ.get("VK_ICD_FILENAMES"):
            if not _MOLTENVK_ICD.exists():
                raise SystemExit("sim rendering on macOS needs MoltenVK: brew install molten-vk")
            os.environ["VK_ICD_FILENAMES"] = str(_MOLTENVK_ICD)

        import gymnasium as gym
        import torch
        import mani_skill.envs  # noqa: F401
        import soframe_rl_maniskill.envs  # noqa: F401
        from mani_skill.utils.structs.pose import Pose

        self.size = int(size)
        self.env = gym.make("SOFramePickPlaceBin-v1", num_envs=1, obs_mode="rgb",
                            sim_backend="cpu", domain_randomization=False,
                            domain_randomization_config={"visual_fidelity": "raster"},
                            sensor_configs=dict(width=self.size, height=self.size))
        self.env.reset(seed=0)
        self.u = self.env.unwrapped
        if not show_objects:
            # Parked far away rather than hidden, so lighting and shadows stay honest.
            self.u.item.set_pose(Pose.create_from_pq(torch.tensor([[5.0, 5.0, 0.5]])))
            self.u.bin.set_pose(Pose.create_from_pq(torch.tensor([[6.0, 6.0, 0.5]])))
        self._idx = {n: i for i, n in enumerate(self.u.agent.joint_names)}
        self._key = None
        self._cache: dict = {}

    def render(self, sim_qpos: dict) -> dict:
        """{sim camera: HxWx3 uint8 RGB} at `sim_qpos`, keyed by joint NAME (no .pos)."""
        key = tuple(round(sim_qpos[n], 5) for n in sorted(sim_qpos))
        if key == self._key:
            return self._cache
        qpos = self.u.agent.robot.get_qpos().clone()
        for name, v in sim_qpos.items():
            if name in self._idx:
                qpos[0, self._idx[name]] = float(v)
        self.u.agent.robot.set_qpos(qpos)
        # A SAPIEN-mounted camera renders from the transforms cached at the PREVIOUS capture, so
        # the first get_obs() after a set_qpos returns the pre-set pose. The discarded capture
        # flushes the mount; update_render() twice does not.
        self.u.scene.update_render()
        self.u.get_obs()
        self.u.scene.update_render()
        obs = self.u.get_obs()
        self._cache = {c: obs["sensor_data"][c]["rgb"][0].cpu().numpy()
                       for c in set(REAL_TO_SIM_CAM.values())}
        self._key = key
        return self._cache

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
