"""Vision encoders, keyed by ``KIND`` for ``--encoder`` on the training side and for
reconstructing the right architecture from a checkpoint on the deploy side."""
from .cnn import SquintEncoder
from .dino_global import DinoGlobalEncoder
from .dino_patch import DinoPatchEncoder

ENCODERS = {cls.KIND: cls for cls in (SquintEncoder, DinoPatchEncoder, DinoGlobalEncoder)}

__all__ = ["ENCODERS", "SquintEncoder", "DinoPatchEncoder", "DinoGlobalEncoder"]
