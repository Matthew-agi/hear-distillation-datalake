#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--num-streams", type=int, default=1)
parser.add_argument("--stream-index", type=int, default=0)
args, _unknown = parser.parse_known_args()

out = args.out / f"stream-{args.stream_index:03d}" if args.num_streams > 1 else args.out
out.mkdir(parents=True, exist_ok=True)
existing = sorted(out.glob("shard-*.tar"))
index = len(existing)
path = out / f"shard-{index:06d}.tar"
with tarfile.open(path, "w") as shard:
    for name, payload in (
        (f"fake-{index}.wav", b"pcm"),
        (f"fake-{index}.json", json.dumps({"clip_id": f"fake-{index}"}).encode()),
    ):
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        shard.addfile(info, io.BytesIO(payload))
size = path.stat().st_size
print(f"DONE written=1 written_bytes={size} seen=1 skipped=0 elapsed=0.1s rate=10.0 clips/s byte_rate=0.1 MiB/s", flush=True)
