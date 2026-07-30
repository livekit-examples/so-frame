"""Squint SAC: the training algorithm, written once for every vision encoder.

The training-only half: critic, replay buffer, update steps, logging, env construction. What ends
up in a checkpoint (encoders, actor) lives in the shared ``soframe_policy`` package, so the deployed
policy is built from the same definitions that trained it.

Entry point: ``train.py`` at the project root, or ``sac.train(sac.Args())``.
"""

from .args import ENCODER_DEFAULTS, Args
from .critic import Critic
from .logger import Logger, evaluate
from .loop import train

__all__ = ["Args", "ENCODER_DEFAULTS", "Critic", "Logger", "evaluate", "train"]
