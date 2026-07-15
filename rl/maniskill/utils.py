"""Training helper wrappers, vendored from Squint's `utils.py`
(https://github.com/aalmuzairee/squint). Robot/task-agnostic: resolution downsampling
and color jitter on RGB observations (upstream's buffer-memory helper was dropped)."""

import gymnasium as gym
import torch
import torch.nn.functional as F

import torchvision

# ---------------------------  Wrappers --------------------------------------#

class DownsampleObsWrapper(gym.ObservationWrapper):
    """Downsamples RGB observations from render_size to target_size using area interpolation.

    Expects input in (B, H, W, C) format.
    """
    def __init__(self, env, target_size):
        super().__init__(env)
        self.target_size = target_size
        # Update observation space
        old_rgb_space = self.observation_space['rgb']
        C = old_rgb_space.shape[-1]
        self.observation_space['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(target_size, target_size, C), dtype=old_rgb_space.dtype
        )

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) or (H, W, C)
        if rgb.shape[-2] == self.target_size:
            return obs  # Already at target size

        # Handle batched and unbatched cases
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W) for interpolate
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb.float(), size=(self.target_size, self.target_size), mode='area').to(torch.uint8)
        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs



class ColorJitterWrapper(gym.ObservationWrapper):
    """Applies random color jitter to RGB observations for sim2real robustness.

    Expects input in (B, H, W, C) format. C may be a multiple of 3 (multiple cameras
    concatenated along the channel axis by `FlattenRGBDObservationWrapper`, e.g. 6 for a
    wrist + overhead camera pair): `torchvision.transforms.ColorJitter`'s hue/saturation
    ops hard-require exactly 3 channels, so each camera's 3 channels are jittered as its
    own independent image rather than jittering the stack as one 6+-channel blob.
    """
    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) or (H, W, C) uint8, C a multiple of 3

        # Handle batched and unbatched cases
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W) for ColorJitter
        rgb = rgb.permute(0, 3, 1, 2)
        num_channels = rgb.shape[1]
        assert num_channels % 3 == 0, f"expected a multiple of 3 channels (one or more RGB cameras), got {num_channels}"
        rgb = torch.cat(
            [self.jitter(rgb[:, c:c + 3].float() / 255.0) for c in range(0, num_channels, 3)],
            dim=1,
        )
        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        # Back to uint8
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs
