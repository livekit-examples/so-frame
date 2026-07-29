"""Global-vector DINOv2 encoder over a frozen ViT-S/14: the ablation of the patch head.

This is the control experiment for `dino_patch`. Patch Policy's claim is that keeping the DENSE
patch tokens and reading them out with self-attention beats collapsing them, which is what
almost every prior frozen-backbone controller did. `dino_patch` implements the dense side. This
implements the collapsed side, so the claim can be tested on our task instead of assumed:

    dino_patch    2 cams -> 288 tokens x 384  -> 4-layer self-attention -> readout token
    dino_global   2 cams ->   2 vectors x 384 -> MLP

Everything else is deliberately identical -- same frozen ViT-S/14 with registers, same
resolution, same ImageNet normalization, same repr_dim and RGB_PROJ_DIM into the actor, same
cached-feature training path. So a difference in success rate is attributable to the spatial
representation and not to capacity elsewhere in the stack.

Two consequences of collapsing, both of which matter more than they first appear:

* The cached observation goes from 216 KB to 1.5 KB, a factor of 140. Replay retention stops
  being a constraint at all: 2 episodes at 512 envs is 300 MB instead of 42 GB.
* Spatial information survives only insofar as the CLS token encodes it. For a task whose whole
  difficulty is *where* the cube is relative to the jaw, that is the thing being ablated. If the
  patch head's advantage is real, this is where it should show up.

``embed_global`` is the sim2real-critical function, mirroring ``dino_patch.tokenize``: training
caches features through it in an obs wrapper, deploy calls it per tick on rectified frames. One
definition, so sim and real cannot disagree on resize mode, normalization or camera ordering.
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
    the channel ordering of the input stack exactly as the patch encoder's token grids do.

    ``pool`` selects what "global" means: the CLS token (default, the canonical single vector),
    the mean over patch tokens, or both concatenated. CLS is the default because it is what the
    backbone was trained to make globally informative; mean-pooling is the stronger classical
    baseline and is offered so the ablation cannot be dismissed as a straw man.
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

    The head is an MLP rather than a single Linear on purpose. A linear probe would lose to the
    patch head's 4-layer transformer on parameter count alone, and the question is whether DENSE
    versus COLLAPSED matters, not whether 4 layers beat 0. Two hidden layers at the transformer's
    own width is the closest honest match: the representation differs, the capacity to exploit it
    does not.
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

        # Per-camera embedding, matching dino_patch's cam_embed: concatenation order already
        # encodes which camera is which, but a learned offset lets the head separate them before
        # they are mixed, rather than relying on the flattening order alone.
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
        """Raw camera stack -> frozen global vectors. Deploy calls this per tick; training does
        it once per env step in an obs wrapper (which calls ``embed_global`` directly)."""
        return embed_global(rgb, self.res, num_cams=self.num_cams,
                            device=self.cam_embed.device, pool=self.pool)

    def forward(self, vectors):
        # vectors arrive bf16 from the wrapper / preprocess; cast to the fp32 head weights.
        vectors = vectors.float()
        x = vectors + self.cam_embed
        return self.out(x.flatten(1))
