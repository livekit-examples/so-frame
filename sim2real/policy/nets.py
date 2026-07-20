"""Policy networks vendored from rl/maniskill/train_squint.py, verbatim, so the
deploy host loads the checkpoint with NO mani_skill / sim dependency.

These MUST stay byte-identical in architecture to the training definitions, or the
checkpoint's state_dict won't load. Only weight_init + CNNEncoder + Projection +
Actor are needed for inference (the Critic is training-only). If the training nets
change, re-copy them here.
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


class CNNEncoder(nn.Module):
    def __init__(self, n_obs, device=None):
        super().__init__()
        assert len(n_obs) == 3 and n_obs[0] == n_obs[1]
        self.num_channels = n_obs[2]
        self.image_size = n_obs[0]
        self.repr_dim = 1024

        if self.image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 8, stride=4, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif self.image_size == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif self.image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        else:
            raise ValueError(f"No CNN encoder supported for image size: {self.image_size}")

        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        obs = obs.contiguous(memory_format=torch.channels_last)
        obs = obs / 255.0 - 0.5
        return self.conv(obs)


class Projection(nn.Module):
    def __init__(self, n_obs, n_state, device=None):
        super().__init__()
        self.repr_dim = 50 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_obs, 50, device=device), nn.LayerNorm(50, device=device), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )

    def forward(self, rgb, state):
        return torch.cat([self.rgb_proj(rgb), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_state, n_act, device=None):
        super().__init__()
        hidden_dim = 256
        activ = nn.ReLU

        self.proj = Projection(n_obs, n_state, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        action_space = env.unwrapped.single_action_space
        self.register_buffer("action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device))
        self.register_buffer("action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device))

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
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action
