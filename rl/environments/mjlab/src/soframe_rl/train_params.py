"""Loads training parameters from ``rl/environments/mjlab/train.toml``.

Kept separate so both the RL runner config (``config/rl_cfg.py``) and the env
config (``config/env_cfg.py``, for ``num_envs``) read a single source of truth.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# train_params.py -> soframe_rl -> src -> rl/environments/mjlab/train.toml
TRAIN_TOML = Path(__file__).resolve().parents[2] / "train.toml"
assert TRAIN_TOML.exists(), f"training config not found at {TRAIN_TOML}"

with open(TRAIN_TOML, "rb") as _f:
  PARAMS: dict = tomllib.load(_f)
