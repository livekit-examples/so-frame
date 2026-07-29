"""Vision encoders, keyed by ``KIND`` for ``--encoder`` and for rebuilding the right
architecture from a checkpoint."""
from .cnn import SquintEncoder
from .dino_global import DinoGlobalEncoder
from .dino_patch import DinoPatchEncoder

ENCODERS = {cls.KIND: cls for cls in (SquintEncoder, DinoPatchEncoder, DinoGlobalEncoder)}

__all__ = ["ENCODERS", "SquintEncoder", "DinoPatchEncoder", "DinoGlobalEncoder"]
