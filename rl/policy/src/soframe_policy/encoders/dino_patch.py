"""Dense patch-token DINOv2 encoder ("Patch Policy" head) over a frozen ViT-S/14.

Adapted from 'Patch Policy: Efficient Embodied Control via Dense Visual Representations'
(Zhou, Cui, Langford, Tan, LeCun, Pinto, 2026): keep the dense frozen patch tokens instead of
global-pooling them, run a small self-attention transformer over both cameras jointly, and
read out a learned [READOUT] token.

The frozen backbone is NOT part of this module (and so not part of the checkpoint) -- it is
cached per device at module level and downloaded from torch.hub on first use. Only the head
trains, which is what lets training cache tokens in the replay buffer and run the ViT once per
env step instead of once per gradient update.

``tokenize`` is the sim2real-critical function: training caches tokens through it in an obs
wrapper, deploy calls it per tick on the rectified camera frames. One definition, so sim and
real cannot silently disagree on resize mode, normalization, or camera ordering.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..actor import weight_init

PATCH = 14                      # DINOv2 ViT-S/14 patch size
EMBED = 384                     # ViT-S token width
_MEAN = (0.485, 0.456, 0.406)   # ImageNet normalization, as DINOv2 was trained
_STD = (0.229, 0.224, 0.225)

_BACKBONE: dict[str, nn.Module] = {}


def load_backbone(device):
    """Frozen DINOv2 ViT-S/14 (with registers), loaded once per device and shared."""
    key = str(device)
    if key not in _BACKBONE:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _BACKBONE[key] = m
    return _BACKBONE[key]


def grid_size(res):
    """Patch-grid side length for a square input resolution."""
    return res // PATCH


def token_count(num_cams, res):
    return num_cams * grid_size(res) ** 2


@torch.no_grad()
def tokenize(rgb, res, num_cams=None, device=None):
    """(B, H, W, 3*num_cams) uint8 -> (B, num_cams*(res/PATCH)^2, EMBED) patch tokens, bf16.

    Each camera is bilinearly resized to ``res``, ImageNet-normalized, and pushed through
    ``forward_features``; the per-camera token grids are concatenated camera-major, so token
    ordering matches the channel ordering of the input stack. Runs under autocast to bf16,
    which is also how the tokens are stored in the replay buffer.
    """
    if rgb.dim() == 3:
        rgb = rgb.unsqueeze(0)
    device = device or rgb.device
    if num_cams is None:
        num_cams = rgb.shape[-1] // 3
    assert rgb.shape[-1] == 3 * num_cams, \
        f"expected {3 * num_cams} channels for {num_cams} cameras, got {rgb.shape[-1]}"

    backbone = load_backbone(device)
    x = rgb.to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)

    amp_device = "cuda" if x.is_cuda else "cpu"
    toks = []
    with torch.autocast(amp_device, dtype=torch.bfloat16):
        for c in range(num_cams):
            cam = x[:, 3 * c: 3 * c + 3]
            if cam.shape[-1] != res:
                cam = F.interpolate(cam, size=(res, res), mode="bilinear", align_corners=False)
            cam = (cam - mean) / std
            toks.append(backbone.forward_features(cam)["x_norm_patchtokens"])
    return torch.cat(toks, dim=1).to(torch.bfloat16)


class DinoPatchEncoder(nn.Module):
    """Self-attention head over frozen DINOv2 patch tokens, reading out a learned token.

    Parameter names match the ``PatchHead`` this replaces (``readout``, ``cam_embed``,
    ``pos_embed``, ``transformer.*``, ``ln.*``, ``out.*``), so v4-era checkpoints load.
    """

    KIND = "dino_patch"
    RGB_PROJ_DIM = 256   # a 50-d bottleneck discards too much of a transformer readout

    def __init__(self, num_cams, res=112, device=None, depth=4, n_heads=6, repr_dim=512,
                 embed=EMBED):
        super().__init__()
        self.num_cams = num_cams
        self.res = res
        self.EMBED = embed
        self.tokens_per_cam = grid_size(res) ** 2
        self.n_tok = num_cams * self.tokens_per_cam
        self.repr_dim = repr_dim

        self.readout = nn.Parameter(torch.zeros(1, 1, embed, device=device))
        self.cam_embed = nn.Parameter(torch.zeros(num_cams, embed, device=device))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tok, embed, device=device))
        layer = nn.TransformerEncoderLayer(
            d_model=embed, nhead=n_heads, dim_feedforward=embed * 4, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True, device=device,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(embed, device=device)
        self.out = nn.Sequential(
            nn.Linear(embed, repr_dim, device=device),
            nn.LayerNorm(repr_dim, device=device), nn.ReLU(),
        )
        # Init the readout MLP with the repo's scheme; leave the transformer to its own init.
        self.out.apply(weight_init)
        nn.init.normal_(self.readout, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.cam_embed)

    def preprocess(self, rgb):
        """Raw camera stack -> frozen patch tokens. Deploy calls this per tick; training does
        it once per env step in an obs wrapper (which calls ``tokenize`` directly)."""
        return tokenize(rgb, self.res, num_cams=self.num_cams, device=self.pos_embed.device)

    def forward(self, tokens):
        # tokens arrive bf16 from the wrapper / preprocess; cast to the fp32 head weights.
        tokens = tokens.float()
        b, tpc = tokens.shape[0], self.tokens_per_cam
        parts = [tokens[:, c * tpc:(c + 1) * tpc] + self.cam_embed[c] for c in range(self.num_cams)]
        x = torch.cat(parts, dim=1) + self.pos_embed
        r = self.readout.expand(b, -1, -1)
        x = torch.cat([r, x], dim=1)
        x = self.transformer(x)
        return self.out(self.ln(x[:, 0]))   # readout token -> repr
