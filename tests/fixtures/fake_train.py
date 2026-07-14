#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--out")
_args, _unknown = parser.parse_known_args()
print("step=1 loss=0.1 batch=1", flush=True)
time.sleep(0.05)
