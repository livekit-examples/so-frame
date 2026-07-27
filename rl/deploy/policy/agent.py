"""Load a trained policy for inference on the real robot.

One class for every architecture. The checkpoint records its own encoder kind, input
resolution, camera count, action bounds and proprio layout, so there is nothing to pass in and
nothing to remember -- see soframe_policy/checkpoint.py.

This replaced a SquintPolicy and a DinoPolicy that each hand-copied the training network
definitions, took the architecture and resolution as constructor arguments, and covered only
2 of the 5 architectures that existed.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch

from soframe_policy import checkpoint


class Policy:
    """Inference wrapper around a trained encoder + actor.

    ``act(rgb, proprio)`` takes the rectified camera stack (H, W, 3*num_cams) uint8 and a
    ``{field: value}`` proprio mapping, and returns the normalized action in [-1, 1]^n_act.
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
        # fails here rather than silently feeding the policy a misaligned vector.
        state = torch.as_tensor(
            self.proprio.assemble(proprio), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        frame = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).unsqueeze(0)
        # preprocess() is the SAME code the training obs pipeline used: the area downsample for
        # the CNN, or the DINOv2 tokenizer for the patch head.
        features = self.encoder(self.encoder.preprocess(frame))
        action = self.actor.get_eval_action(features, state)
        return action.squeeze(0).cpu().numpy()
