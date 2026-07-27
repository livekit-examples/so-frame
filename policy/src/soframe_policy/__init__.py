"""Policy networks shared by SO-101-on-frame training and deployment.

This package is the single definition of everything that ends up in a checkpoint: the vision
encoders, the SAC actor, and how a raw camera stack becomes encoder input. Training
(``rl/maniskill``) and deploy (``sim2real``) both take it as a path dependency, so there is no
"MUST stay byte-identical" hand-copy to drift.

Pure torch. Nothing here imports mani_skill, gymnasium or livekit -- the critic, replay buffer
and training loop stay on the training side, since they never run on the robot.
"""
from .actor import Actor, Projection, weight_init
from .encoders import ENCODERS, DinoPatchEncoder, SquintEncoder
from .proprio import ProprioSpec
from . import checkpoint, proprio, rig

__all__ = [
    "Actor", "Projection", "weight_init",
    "ENCODERS", "SquintEncoder", "DinoPatchEncoder",
    "ProprioSpec", "checkpoint", "proprio", "rig",
]
