"""DINOv2-v2 policy networks, vendored from rl/maniskill/train_dino_v2.py, so the deploy
host loads a v39-style checkpoint with NO mani_skill / sim dependency.

Architecture (must stay byte-identical to training or the state_dict won't load):
  frozen DINOv2 ViT-S/14 (registers) backbone  ->  patch tokens  ->  DinoHead
  (attention pooling)  ->  512-d repr  ->  Actor (with the WIDENED 256-d rgb projection,
  not squint's 50-d one).

The frozen backbone is loaded from torch.hub at deploy (needs internet / ~/.cache the
first time) and is NOT in the checkpoint; only DinoHead ('encoder') + Actor are.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import weight_init  # reuse the identical init

# ── Frozen DINOv2 backbone + tokenizer (mirrors utils.DinoTokenWrapper) ──────
_DINO_BACKBONE = {}
_PATCH = 14
_EMBED = 384
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def load_dino_backbone(device):
    key = str(device)
    if key not in _DINO_BACKBONE:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _DINO_BACKBONE[key] = m
    return _DINO_BACKBONE[key]


@torch.no_grad()
def image_to_tokens(rgb_bhwc, backbone, num_cams, dino_res, device):
    """(B, H, W, 3*num_cams) uint8 -> (B, num_cams*(dino_res/14)^2, EMBED) tokens.

    Matches DinoTokenWrapper: each camera's 3 channels are resized to dino_res, ImageNet-
    normalized, and run through DINOv2 forward_features; the per-camera token grids are
    concatenated camera-major.
    """
    x = rgb_bhwc.to(device).permute(0, 3, 1, 2).float() / 255.0  # (B, C, H, W)
    mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)
    toks = []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if x.is_cuda else torch.autocast("cpu", dtype=torch.bfloat16)
    with amp:
        for c in range(num_cams):
            cam = x[:, 3 * c: 3 * c + 3]
            if cam.shape[-1] != dino_res:
                cam = F.interpolate(cam, size=(dino_res, dino_res), mode="bilinear", align_corners=False)
            cam = (cam - mean) / std
            toks.append(backbone.forward_features(cam)["x_norm_patchtokens"])
    return torch.cat(toks, dim=1).float()  # (B, n_tok, EMBED)


# ── Trainable head + actor (widened projection) ─────────────────────────────
class DinoHead(nn.Module):
    """Attention-pooling head over cached DINOv2 tokens (see train_dino_v2.DinoHead)."""

    def __init__(self, n_tok, num_cams, embed=384, device=None, n_queries=8, n_heads=6, repr_dim=512):
        super().__init__()
        assert n_tok % num_cams == 0
        self.EMBED = embed
        self.n_tok = n_tok
        self.num_cams = num_cams
        self.tokens_per_cam = n_tok // num_cams
        self.repr_dim = repr_dim

        self.cam_embed = nn.Parameter(torch.zeros(num_cams, embed, device=device))
        self.queries = nn.Parameter(torch.zeros(n_queries, embed, device=device))
        self.ln_kv = nn.LayerNorm(embed, device=device)
        self.ln_q = nn.LayerNorm(embed, device=device)
        self.attn = nn.MultiheadAttention(embed, n_heads, batch_first=True, device=device)
        self.out = nn.Sequential(
            nn.Linear(n_queries * embed, repr_dim, device=device),
            nn.LayerNorm(repr_dim, device=device), nn.ReLU(),
        )
        self.apply(weight_init)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.zeros_(self.cam_embed)

    def forward(self, tokens):
        tokens = tokens.float()
        b, tpc = tokens.shape[0], self.tokens_per_cam
        parts = [tokens[:, c * tpc:(c + 1) * tpc] + self.cam_embed[c] for c in range(self.num_cams)]
        kv = self.ln_kv(torch.cat(parts, dim=1))
        q = self.ln_q(self.queries).unsqueeze(0).expand(b, -1, -1)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        return self.out(pooled.reshape(b, -1))


class Projection(nn.Module):
    """v2 projection: rgb -> 256 (not squint's 50), matching train_dino_v2.Projection."""

    def __init__(self, n_obs, n_state, device=None):
        super().__init__()
        self.repr_dim = 256 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_obs, 256, device=device), nn.LayerNorm(256, device=device), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )

    def forward(self, rgb, state):
        return torch.cat([self.rgb_proj(rgb), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    """Identical to nets.Actor but bound to the v2 (256-d) Projection above."""

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

    def forward(self, rgb, state):
        x = self.proj(rgb, state)
        x = self.fc(x)
        return self.fc_mean(x)

    def get_eval_action(self, rgb, state):
        mean = self.forward(rgb, state)
        return torch.tanh(mean) * self.action_scale + self.action_bias
