"""Visualize the SOFramePickPlaceBin-v1 ManiSkill3 task.

Adapted from Squint's `examples/visualize_sim.py` (https://github.com/aalmuzairee/squint) for
this repo's single task. Shows the wrist-camera observation (what the policy sees) side by
side with a third-person render, cycling the gripper open/closed with random arm motion.
Useful for sanity-checking the scene, camera calibration, and domain randomization before
spending GPU time on training. Run from `rl/maniskill/`:

    uv run python examples/visualize_sim.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import logging
logging.disable(level=logging.WARN)

import numpy as np
import cv2
import torch
import gymnasium as gym

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.visualization.misc import tile_images

import utils

# Add tasks: registers the SO-101-on-frame agent and SOFramePickPlaceBin-v1 task.
import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'task': 'SOFramePickPlaceBin-v1',

    # Environment settings
    'num_envs': 16,
    'seed': 1,
    'obs_mode': 'rgb+segmentation',  # for the wrist camera view
    'render_mode': 'rgb_array',
    'image_size': 128,
    'color_jitter': False,
    'downsample_size': 128,
    'control_mode': None,
    'domain_randomization': True,

    # Visualization settings
    'window_size': 512,
    'steps': 60,
    'reset_interval': 20,
}


# =============================================================================
# Environment Factory
# =============================================================================

def make_env(task: str, config: dict = CONFIG):
    """Create a ManiSkill environment with the given configuration."""

    sensor_size = {'width': config['image_size'], 'height': config['image_size']}

    env_kwargs = dict(
        obs_mode=config['obs_mode'],
        render_mode=config['render_mode'],
        sensor_configs=sensor_size,
        human_render_camera_configs=sensor_size,
        num_envs=config['num_envs'],
        domain_randomization=config['domain_randomization'],
        reconfiguration_freq=None,
    )

    if config['control_mode'] is not None:
        env_kwargs['control_mode'] = config['control_mode']

    env = gym.make(task, **env_kwargs)

    if "rgb" in config['obs_mode']:
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        if config['downsample_size'] is not None:
            env = utils.DownsampleObsWrapper(env, target_size=config['downsample_size'])
        if config['color_jitter']:
            env = utils.ColorJitterWrapper(env)

    env.reset(seed=config['seed'])
    return env


# =============================================================================
# Visualization
# =============================================================================

def visualize(config: dict = CONFIG):
    """Visualize the task with a scripted open/close gripper action."""

    task = config['task']
    window_size = config['window_size']
    steps = config['steps']
    reset_interval = config['reset_interval']

    print(f"Instantiating: {task}")
    env = make_env(task, config)

    obs, info = env.reset()
    action_shape = env.action_space.shape
    num_envs = config['num_envs']
    video_nrows = int(np.sqrt(num_envs))

    print(f"Running: {task}")

    for step in range(steps):
        # Open gripper for the first third of steps, close after (last action dim == gripper).
        action = np.zeros(action_shape)
        if step < steps // 3:
            action[..., -1] = 1
        else:
            action[..., -1] = -1

        obs, reward, terminated, truncated, info = env.step(action)
        done = (terminated | truncated).any()

        # Third-person render view (N, H, W, 3).
        render_rgb = env.render()

        # Observation RGB (wrist camera view).
        if isinstance(obs, dict) and 'rgb' in obs:
            obs_rgb = obs['rgb']  # (N, H, W, C) where C may be 3 or 3*num_views

            if obs_rgb.shape[-1] != 3 and obs_rgb.shape[-1] % 3 == 0:
                obs_rgb = obs_rgb[..., :3]  # Take first camera view

            render_h, render_w = render_rgb.shape[1], render_rgb.shape[2]
            if obs_rgb.shape[1] != render_h or obs_rgb.shape[2] != render_w:
                obs_rgb = torch.nn.functional.interpolate(
                    obs_rgb.permute(0, 3, 1, 2).float(),  # (N, 3, H, W)
                    size=(render_h, render_w),
                    mode='nearest',
                ).permute(0, 2, 3, 1).to(torch.uint8)  # (N, H, W, 3)

            paired = torch.cat([obs_rgb, render_rgb], dim=2)
            rgb = tile_images(paired, nrows=video_nrows).cpu().numpy().astype(np.uint8)
            rgb = cv2.resize(rgb, dsize=(window_size * 2, window_size))
        else:
            rgb = tile_images(render_rgb, nrows=video_nrows).cpu().numpy().astype(np.uint8)
            rgb = cv2.resize(rgb, dsize=(window_size, window_size))

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        print(f"Step: {step}/{steps}, done={done}", end="\r")
        cv2.imshow("SOFramePickPlaceBin-v1: wrist cam | third-person", rgb)
        cv2.waitKey(30)

        if (step % reset_interval == 0) or done:
            env.reset()

    env.close()
    cv2.destroyAllWindows()
    print(f"Finished: {task}                    ")


if __name__ == '__main__':
    visualize()
