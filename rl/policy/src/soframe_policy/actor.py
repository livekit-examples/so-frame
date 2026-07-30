"""The SAC actor and its observation projection, shared by training and deploy.

Vendored from Squint (https://github.com/aalmuzairee/squint); parameter names and shapes are
unchanged, so existing checkpoints load. ``Projection``'s RGB width is a constructor argument
(50 for the squint CNN, 256 for the DINOv2 heads), which is not a parameter name.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Projection(nn.Module):
    """Fuse encoder features with proprio: RGB -> LayerNorm -> Tanh, state -> LayerNorm -> ReLU,
    concatenated. ``rgb_dim`` comes from the encoder (``Encoder.RGB_PROJ_DIM``)."""

    def __init__(self, n_obs, n_state, device=None, rgb_dim=50, state_dim=256):
        super().__init__()
        self.repr_dim = rgb_dim + state_dim
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_obs, rgb_dim, device=device), nn.LayerNorm(rgb_dim, device=device), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, state_dim, device=device), nn.LayerNorm(state_dim, device=device), nn.ReLU(),
        )

    def forward(self, rgb, state):
        return torch.cat([self.rgb_proj(rgb), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    """Squashed-Gaussian SAC actor over (encoder features, proprio).

    Action bounds are plain arrays, not a gym env, so deploy needs no env object.
    ``action_scale``/``action_bias`` are registered buffers.
    """

    def __init__(self, n_obs, n_state, n_act, action_low, action_high,
                 device=None, rgb_dim=50, hidden_dim=256):
        super().__init__()
        activ = nn.ReLU

        self.proj = Projection(n_obs, n_state, device=device, rgb_dim=rgb_dim)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def forward(self, rgb, state, get_log_std=False):
        x = self.proj(rgb, state)
        x = self.fc(x)
        mean = self.fc_mean(x)
        if get_log_std:
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
            return mean, log_std
        return mean

    def get_eval_action(self, rgb, state):
        mean = self.forward(rgb, state)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, rgb, state):
        mean, log_std = self.forward(rgb, state, get_log_std=True)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
