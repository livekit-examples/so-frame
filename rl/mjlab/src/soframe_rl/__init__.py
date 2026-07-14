"""RL task: SO-101-on-frame picks up a cube and places it in a bin.

Importing this package registers the task(s) with mjlab's task registry, so
``import soframe_rl`` must run before mjlab's train/play scripts look the task up.
The ``soframe-train`` / ``soframe-play`` console scripts (and ``train.py`` /
``play.py``) do this for you.
"""

from soframe_rl import config as config  # noqa: F401  (side effect: registers tasks)

__all__ = ["config"]
