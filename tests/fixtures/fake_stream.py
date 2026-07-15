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
parser.add_argument("--fake-clips", type=int, default=1)
args, _unknown = parser.parse_known_args()

out = args.out / f"stream-{args.stream_index:03d}" if args.num_streams > 1 else args.out
out.mkdir(parents=True, exist_ok=True)
existing = sorted(out.glob("shard-*.tar"))
counter_path = out / "fake_next_index.txt"
index = int(counter_path.read_text()) if counter_path.exists() else len(existing)
counter_path.write_text(str(index + 1))
path = out / f"shard-{index:06d}.tar"
with tarfile.open(path, "w") as shard:
    for offset in range(args.fake_clips):
        clip_id = f"fake-{index * args.fake_clips + offset}"
        for name, payload in (
            (f"{clip_id}.wav", b"pcm"),
            (f"{clip_id}.json", json.dumps({"clip_id": clip_id}).encode()),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            shard.addfile(info, io.BytesIO(payload))
size = path.stat().st_size
print(
    f"DONE written={args.fake_clips} written_bytes={size} seen={args.fake_clips} "
    "skipped=0 elapsed=0.1s rate=10.0 clips/s byte_rate=0.1 MiB/s",
    flush=True,
)
