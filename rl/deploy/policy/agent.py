"""Load a trained policy for inference on the real robot.

One class for every architecture: the checkpoint records its own encoder kind, input
resolution, camera count, action bounds and proprio layout. See soframe_policy/checkpoint.py.

On a Mac set POLICY_DEVICE=mps; the default device pick is cuda-or-cpu, so DINOv2 lands on
the CPU otherwise.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from soframe_policy import checkpoint


class Policy:
    """Inference wrapper around a trained encoder + actor.

    ``act(rgb, proprio)``: rectified camera stack (H, W, 3*num_cams) uint8 plus a
    ``{field: value}`` proprio mapping -> normalized action in [-1, 1]^n_act.
    """

    def __init__(self, ckpt_path: str | pathlib.Path, device: str | None = None):
        self.encoder, self.actor, self.meta = checkpoint.load(ckpt_path, device=device)
        self.device = next(self.actor.parameters()).device
        self.proprio = self.meta["proprio"]
        print(f"[policy] {self.meta['kind']} @ res {self.meta['res']}, "
              f"{self.meta['num_cams']} cameras, step {self.meta['global_step']}, "
              f"on {self.device}")
        print(f"[policy] proprio {self.proprio.describe()}")

    @property
    def num_cams(self) -> int:
        return self.meta["num_cams"]

    @torch.no_grad()
    def act(self, rgb: np.ndarray, proprio: dict) -> np.ndarray:
        expected = 3 * self.num_cams
        if rgb.shape[-1] != expected:
            raise ValueError(
                f"this checkpoint was trained on {self.num_cams} cameras "
                f"({expected} channels); got {rgb.shape[-1]}"
            )
        # assemble() raises on a missing, unknown or wrong-width field, so a proprio mismatch
        # fails here instead of feeding the policy a misaligned vector.
        state = torch.as_tensor(
            self.proprio.assemble(proprio), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        frame = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).unsqueeze(0)
        # preprocess() is the same code the training obs pipeline used.
        features = self.encoder(self.encoder.preprocess(frame))
        action = self.actor.get_eval_action(features, state)
        return action.squeeze(0).cpu().numpy()
