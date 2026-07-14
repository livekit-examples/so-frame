"""Thin wrapper around mjlab's train CLI that first registers our task.

Usage (from anywhere, after `uv pip install -e rl`):

    soframe-train Mjlab-Pick-Place-Bin-SO101 --env.scene.num-envs 4096

or equivalently:

    python -m soframe_rl.train Mjlab-Pick-Place-Bin-SO101 ...
"""

import soframe_rl  # noqa: F401  (registers the task before mjlab reads the registry)
from mjlab.scripts.train import main as _main


def main() -> None:
  _main()


if __name__ == "__main__":
  main()
