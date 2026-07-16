#!/usr/bin/env python3
"""Executable entry point for direct reconstruction pretraining."""

from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hear_distill.training.pretrain import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
