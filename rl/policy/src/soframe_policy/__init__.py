"""Policy networks shared by SO-101-on-frame training and deployment.

Single definition of everything that ends up in a checkpoint: the vision encoders, the SAC actor,
and how a raw camera stack becomes encoder input. Training (``rl/environments/maniskill``) and
deploy (``rl/deploy``) both take it as a path dependency.

Pure torch: must not import mani_skill, gymnasium or livekit, since both the GPU box and the
robot host install this package. The critic, replay buffer and training loop stay on the
training side.
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
