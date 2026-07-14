"""Thin wrapper around mjlab's play CLI that first registers our task.

Usage (after training):

    soframe-play Mjlab-Pick-Place-Bin-SO101 --checkpoint-file <path>

or:

    python -m soframe_rl.play Mjlab-Pick-Place-Bin-SO101 ...
"""

import soframe_rl  # noqa: F401  (registers the task before mjlab reads the registry)
from mjlab.scripts.play import main as _main


def main() -> None:
  _main()


if __name__ == "__main__":
  main()
