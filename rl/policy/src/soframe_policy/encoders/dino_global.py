"""Global-vector DINOv2 encoder over a frozen ViT-S/14: the collapsed-pooling control for
``dino_patch``.

    dino_patch    2 cams -> 288 tokens x 384  -> 4-layer self-attention -> readout token
    dino_global   2 cams ->   2 vectors x 384 -> MLP

Everything else matches ``dino_patch``: same frozen ViT-S/14 with registers, same resolution,
ImageNet normalization, repr_dim, RGB_PROJ_DIM and cached-feature training path, so the spatial
representation is the only difference. Collapsing also shrinks a cached observation from 216 KB
to 1.5 KB.

``embed_global`` is sim2real-critical, mirroring ``dino_patch.tokenize``: training caches features
through it in an obs wrapper, deploy calls it per tick. One definition, so sim and real cannot
disagree on resize mode, normalization or camera ordering.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..actor import weight_init
from .dino_patch import EMBED, _MEAN, _STD, load_backbone

POOLS = ("cls", "mean", "cls_mean")


def vector_count(num_cams, pool="cls"):
    """Vectors emitted per observation. ``cls_mean`` concatenates two per camera."""
    return num_cams * (2 if pool == "cls_mean" else 1)


@torch.no_grad()
def embed_global(rgb, res, num_cams=None, device=None, pool="cls"):
    """(B, H, W, 3*num_cams) uint8 -> (B, vector_count, EMBED) global vectors, bf16.

    Each camera is bilinearly resized to ``res``, ImageNet-normalized and pushed through
    ``forward_features``; the per-camera vectors are stacked camera-major, so ordering matches
    the input stack's channel order, as the patch encoder's token grids do.

    ``pool`` is what "global" means: ``cls`` (default, what the backbone was trained to make
    globally informative), ``mean`` over patch tokens, or ``cls_mean`` for both concatenated.
    """
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
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
    vecs = []
    with torch.autocast(amp_device, dtype=torch.bfloat16):
        for c in range(num_cams):
            cam = x[:, 3 * c: 3 * c + 3]
            if cam.shape[-1] != res:
                cam = F.interpolate(cam, size=(res, res), mode="bilinear", align_corners=False)
            cam = (cam - mean) / std
            feats = backbone.forward_features(cam)
            if pool in ("cls", "cls_mean"):
                vecs.append(feats["x_norm_clstoken"].unsqueeze(1))
            if pool in ("mean", "cls_mean"):
                vecs.append(feats["x_norm_patchtokens"].mean(dim=1, keepdim=True))
    return torch.cat(vecs, dim=1).to(torch.bfloat16)


class DinoGlobalEncoder(nn.Module):
    """MLP head over frozen DINOv2 global vectors, one (or two) per camera.

    Input is (B, n_vec, EMBED) bf16; output is (B, repr_dim) fp32. Two hidden layers at the patch
    head's own width, so capacity is not a confound against it.
    """

    KIND = "dino_global"
    RGB_PROJ_DIM = 256   # matched to dino_patch so the actor's bottleneck is not a confound

    def __init__(self, num_cams, res=168, device=None, repr_dim=512, hidden=512, depth=2,
                 pool="cls", embed=EMBED):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        self.num_cams = num_cams
        self.res = res
        self.pool = pool
        self.EMBED = embed
        self.n_vec = vector_count(num_cams, pool)
        self.repr_dim = repr_dim

        # Per-camera learned offset, matching dino_patch's cam_embed, so the head can separate the
        # cameras before they are mixed rather than relying on flattening order alone.
        self.cam_embed = nn.Parameter(torch.zeros(self.n_vec, embed, device=device))

        layers = [nn.LayerNorm(self.n_vec * embed, device=device)]
        width = self.n_vec * embed
        for _ in range(depth):
            layers += [nn.Linear(width, hidden, device=device),
                       nn.LayerNorm(hidden, device=device), nn.ReLU()]
            width = hidden
        layers += [nn.Linear(width, repr_dim, device=device),
                   nn.LayerNorm(repr_dim, device=device), nn.ReLU()]
        self.out = nn.Sequential(*layers)

        self.out.apply(weight_init)
        nn.init.zeros_(self.cam_embed)

    def preprocess(self, rgb):
        """Raw camera stack -> frozen global vectors. Deploy calls this per tick; training calls
        ``embed_global`` once per env step in an obs wrapper."""
        return embed_global(rgb, self.res, num_cams=self.num_cams,
                            device=self.cam_embed.device, pool=self.pool)

    def forward(self, vectors):
        # vectors arrive bf16 from the wrapper / preprocess; cast to the fp32 head weights.
        vectors = vectors.float()
        x = vectors + self.cam_embed
        return self.out(x.flatten(1))
