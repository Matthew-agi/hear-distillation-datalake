#!/usr/bin/env python3
"""Organized entry point for the legacy-compatible distillation trainer."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_hear_vit_s_canon2d import main  # noqa: E402


if __name__ == "__main__":
    main()
