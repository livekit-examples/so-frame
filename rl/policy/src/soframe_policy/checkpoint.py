"""The deploy checkpoint format: weights plus enough metadata to rebuild the policy.

Encoder kind, camera count, input resolution, action bounds and proprio layout are properties of
the trained artifact, so they are written into it and read back rather than passed by hand: a
wrong value loads and runs on features the policy never trained on, a silent sim2real mismatch on
real hardware.

``save`` is called by training, ``load`` by deploy. Checkpoints written before this format have no
``meta`` block; pass ``kind`` and ``res`` explicitly to load those.
"""
from __future__ import annotations

import pathlib

import torch

from .actor import Actor
from .encoders import ENCODERS
from .proprio import ProprioSpec

FORMAT_VERSION = 1


def save(path, encoder, actor, *, num_cams, res, n_act, action_low, action_high,
         proprio, global_step=None, extra=None, encoder_kwargs=None):
    """Write encoder + actor weights and the metadata needed to rebuild them.

    ``proprio`` is the ``ProprioSpec`` measured from the training env; it defines n_state and
    the field order deploy must reproduce.

    ``encoder_kwargs`` are constructor arguments beyond (num_cams, res) and MUST be recorded:
    dino_global's ``pool="cls"`` and ``pool="mean"`` produce identical state-dict shapes, so a
    checkpoint trained on one would otherwise load silently into the other.
    """
    if not isinstance(proprio, ProprioSpec):
        raise TypeError(f"proprio must be a ProprioSpec, got {type(proprio).__name__}")
    meta = {
        "format_version": FORMAT_VERSION,
        "kind": type(encoder).KIND,
        "num_cams": int(num_cams),
        "res": int(res),
        "n_state": proprio.n_state,
        "n_act": int(n_act),
        "proprio": proprio.to_meta(),
        "action_low": torch.as_tensor(action_low, dtype=torch.float32).cpu(),
        "action_high": torch.as_tensor(action_high, dtype=torch.float32).cpu(),
        "encoder_kwargs": dict(encoder_kwargs or {}),
    }
    payload = {
        "meta": meta,
        "encoder": encoder.state_dict(),
        "actor": actor.state_dict(),
        "global_step": global_step,
    }
    if extra:
        payload.update(extra)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))
    return path


def build(kind, *, num_cams, res, n_state, n_act, action_low, action_high, device=None,
          encoder_kwargs=None):
    """Construct a matched (encoder, actor) pair: the actor's RGB projection width follows the
    encoder that feeds it."""
    if kind not in ENCODERS:
        raise ValueError(f"unknown encoder kind {kind!r}; known: {sorted(ENCODERS)}")
    encoder_cls = ENCODERS[kind]
    encoder = encoder_cls(num_cams, res=res, device=device, **(encoder_kwargs or {}))
    actor = Actor(encoder.repr_dim, n_state, n_act, action_low, action_high,
                  device=device, rgb_dim=encoder_cls.RGB_PROJ_DIM)
    return encoder, actor


def load(path, device=None, *, kind=None, res=None, num_cams=2, n_act=7, proprio=None,
         encoder_kwargs=None):
    """Rebuild (encoder, actor, meta) from a checkpoint, in eval mode.

    ``meta["proprio"]`` comes back as a ``ProprioSpec``; deploy assembles its state vector
    through it rather than hardcoding a width.

    Metadata in the checkpoint wins. The keyword arguments are only a fallback for pre-``meta``
    checkpoints, and are required rather than defaulted there, since a wrong value is silent.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(str(path), map_location="cpu")
    meta = dict(ckpt.get("meta") or {})

    if not meta:
        missing = [n for n, v in (("kind", kind), ("res", res), ("proprio", proprio)) if v is None]
        if missing:
            raise ValueError(
                f"{path} predates the deploy checkpoint format (no 'meta' block) and "
                f"{', '.join(missing)} was not supplied. Pass the values the checkpoint was "
                "trained with -- guessing them risks running the robot on wrong-resolution "
                "features or a misaligned proprio vector."
            )
        meta = {"kind": kind, "res": res, "num_cams": num_cams,
                "n_state": proprio.n_state, "n_act": n_act, "proprio": proprio.to_meta(),
                "action_low": torch.full((n_act,), -1.0),
                "action_high": torch.full((n_act,), 1.0)}

    spec = ProprioSpec.from_meta(meta["proprio"])
    encoder_kwargs = meta.get("encoder_kwargs") or (encoder_kwargs or {})
    # Build on CPU, then move: the constructors' nn.init.orthogonal_ calls torch.linalg.qr, which
    # has no MPS kernel before torch 2.13, and load_state_dict overwrites the init anyway.
    encoder, actor = build(
        meta["kind"], num_cams=meta["num_cams"], res=meta["res"], n_state=spec.n_state,
        n_act=meta["n_act"], action_low=meta["action_low"], action_high=meta["action_high"],
        device="cpu", encoder_kwargs=encoder_kwargs,
    )
    encoder.load_state_dict(ckpt["encoder"])
    actor.load_state_dict(ckpt["actor"])
    encoder = encoder.to(device).eval()
    actor = actor.to(device).eval()
    meta["proprio"] = spec
    meta["global_step"] = ckpt.get("global_step")
    return encoder, actor, meta
