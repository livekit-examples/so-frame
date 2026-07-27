"""Train a vision policy on SOFramePickPlaceBin-v1 with Squint SAC.

One script for every architecture; pick the vision encoder with --encoder.

    uv run python train.py                              # squint CNN (default)
    uv run python train.py --encoder dino_patch         # frozen DINOv2 + patch head
    uv run python train.py --help                       # every flag, with docs

Defaults that follow from the encoder (input resolution, replay size, updates per step) are
filled in by Args.resolve(); see src/soframe_rl_maniskill/sac/args.py. Task, robot and reward
constants live in src/soframe_rl_maniskill/config.py.

This replaced train_squint.py / train_dino.py / train_dino_v2.py / train_dino_v3.py /
train_dino_v4.py, which were ~95% the same SAC loop.
"""

import os

# Must precede the torch / tensordict imports that read them.
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["EXCLUDE_TD_FROM_PYTREE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message="Using lock_\\(\\) in a compiled graph")

import torch
import tyro

# Registers the SO-101-on-frame agent and the SOFramePickPlaceBin-v1 task.
import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401

from soframe_rl_maniskill import sac

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


if __name__ == "__main__":
    sac.train(tyro.cli(sac.Args))
