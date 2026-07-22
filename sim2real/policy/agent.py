"""Load the trained Squint RL policy for inference, without a live sim.

CNNEncoder + Actor are VENDORED into policy/nets.py (plain torch, copied verbatim
from rl/maniskill/train_squint.py) so the deploy host loads the checkpoint with no
mani_skill / sim dependency. If the training nets change, re-copy them into nets.py.

The policy replans every tick, so there's no chunk queue / settle gate here.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from nets import Actor, CNNEncoder

# Deploy obs geometry, fixed by the v31 recipe: two 3-channel cameras -> 6
# channels, squinted to 32x32; proprio = qpos(7) + controller target(7) = 14.
IMAGE_SIZE = 32
N_CHANNELS = 6
N_STATE = 14
N_ACT = 7

class SquintPolicy:
    """Inference wrapper around the trained encoder + actor.

    act(rgb6, state14): 128x128x6 uint8 image (wrist|overhead, rectified) + 14-dim
    sim-unit proprio -> 7-dim normalized action in [-1, 1]. Squint (128 -> 32, area)
    matches DeployAgent.downsample_rgb.
    """

    def __init__(self, checkpoint: str | pathlib.Path, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Actor needs an env only for action-space bounds; deployed space is
        # [-1, 1]^7, so a tiny stub suffices.
        env_stub = SimpleNamespace(unwrapped=SimpleNamespace(
            single_action_space=_box(-1.0, 1.0, (N_ACT,))
        ))
        self.encoder = CNNEncoder((IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS), self.device)
        self.actor = Actor(env_stub, n_obs=self.encoder.repr_dim, n_state=N_STATE,
                           n_act=N_ACT, device=self.device)
        ckpt = torch.load(str(checkpoint), map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self.encoder.eval()
        self.actor.eval()
        step = ckpt.get("global_step", "?")
        print(f"[policy] loaded checkpoint {checkpoint} (step {step}) on {self.device}")

    def _squint(self, rgb6: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(rgb6)).to(self.device)
        if t.shape[-2] != IMAGE_SIZE:
            x = t.permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="area")
            t = x.squeeze(0).permute(1, 2, 0).to(torch.uint8)
        return t.unsqueeze(0)  # (1, 32, 32, 6)

    @torch.no_grad()
    def act(self, rgb6: np.ndarray, state14: np.ndarray) -> np.ndarray:
        rgb = self._squint(rgb6)
        feats = self.encoder(rgb)
        state = torch.as_tensor(state14, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor.get_eval_action(feats, state)
        return action.squeeze(0).detach().cpu().numpy()


def _box(low: float, high: float, shape):
    """Minimal gym.spaces.Box stand-in with the attrs Actor.__init__ reads."""
    return SimpleNamespace(
        low=np.full(shape, low, dtype=np.float32),
        high=np.full(shape, high, dtype=np.float32),
        shape=tuple(shape),
    )


# DINOv2-v2 deploy geometry (train_dino_v2): each camera -> DINOv2 at 112px (multiple of
# the 14px patch) -> an 8x8=64 token grid; 2 cameras -> 128 tokens of 384 dims.
DINO_RES = 112
DINO_NUM_CAMS = N_CHANNELS // 3
DINO_N_TOK = DINO_NUM_CAMS * (DINO_RES // 14) ** 2
DINO_EMBED = 384


class DinoPolicy:
    """Inference wrapper for the DINOv2-v2 policy (train_dino_v2): frozen DINOv2 backbone
    + attention-pool head + actor, all vendored in nets_dino so deploy is sim-free.

    Same interface as SquintPolicy -- act(rgb6, state14): HxWx6 uint8 (wrist|overhead,
    rectified) + 14-dim sim-unit proprio -> 7-dim normalized action in [-1, 1]. Each
    camera is resized to 112, run through the frozen ViT to patch tokens, attention-pooled,
    and fed to the actor. The frozen backbone downloads from torch.hub on first use.
    """

    def __init__(self, checkpoint: str | pathlib.Path, device: str | None = None):
        import nets_dino
        self._nd = nets_dino
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone = nets_dino.load_dino_backbone(self.device)
        env_stub = SimpleNamespace(unwrapped=SimpleNamespace(
            single_action_space=_box(-1.0, 1.0, (N_ACT,))
        ))
        self.head = nets_dino.DinoHead(n_tok=DINO_N_TOK, num_cams=DINO_NUM_CAMS,
                                       embed=DINO_EMBED, device=self.device)
        self.actor = nets_dino.Actor(env_stub, n_obs=self.head.repr_dim, n_state=N_STATE,
                                     n_act=N_ACT, device=self.device)
        ckpt = torch.load(str(checkpoint), map_location=self.device)
        self.head.load_state_dict(ckpt["encoder"])   # 'encoder' == the DinoHead in training
        self.actor.load_state_dict(ckpt["actor"])
        self.head.eval()
        self.actor.eval()
        step = ckpt.get("global_step", "?")
        print(f"[policy] loaded DINOv2-v2 checkpoint {checkpoint} (step {step}) on {self.device}")

    @torch.no_grad()
    def act(self, rgb6: np.ndarray, state14: np.ndarray) -> np.ndarray:
        rgb = torch.from_numpy(np.ascontiguousarray(rgb6)).to(self.device)
        if rgb.dim() == 3:
            rgb = rgb.unsqueeze(0)  # (1, H, W, 6)
        tokens = self._nd.image_to_tokens(rgb, self.backbone, DINO_NUM_CAMS, DINO_RES, self.device)
        feats = self.head(tokens)
        state = torch.as_tensor(state14, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor.get_eval_action(feats, state)
        return action.squeeze(0).detach().cpu().numpy()
