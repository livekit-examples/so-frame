"""Squint's low-resolution CNN encoder, vendored from https://github.com/aalmuzairee/squint with
the conv stack untouched, so existing squint checkpoints load."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..actor import weight_init


class SquintEncoder(nn.Module):
    """Conv encoder over a heavily downsampled multi-camera RGB stack.

    Input is (B, res, res, 3*num_cams) uint8, cameras concatenated on the channel axis. ``res``
    is the squinted resolution and must be 16, 32 or 64: the conv stack is sized per resolution.
    """

    KIND = "squint"
    RGB_PROJ_DIM = 50   # squint's narrow bottleneck into Projection

    def __init__(self, num_cams, res=32, device=None):
        super().__init__()
        self.num_cams = num_cams
        self.num_channels = 3 * num_cams
        self.res = res
        self.repr_dim = 1024

        if res == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 8, stride=4, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif res == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif res == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        else:
            raise ValueError(f"No CNN encoder supported for image size: {res}")

        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def preprocess(self, rgb):
        """(B, H, W, 3*num_cams) uint8 at any H=W -> the same stack squinted to ``res``.

        Area interpolation, a no-op when the input is already at ``res``. Training and deploy must
        both go through this one definition so they cannot disagree on resize or camera order.
        """
        if rgb.shape[-2] == self.res:
            return rgb
        x = rgb.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, size=(self.res, self.res), mode="area")
        return x.to(torch.uint8).permute(0, 2, 3, 1)

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        obs = obs.contiguous(memory_format=torch.channels_last)
        obs = obs / 255.0 - 0.5
        return self.conv(obs)
