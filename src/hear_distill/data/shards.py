"""Iterable access to the LAION-Audio tar shards produced by the data lake."""

from __future__ import annotations

import json
import os
import random
import tarfile
import time
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from hear_distill.audio import decode_wav_bytes


def discover_shards(
    data_dir: Path,
    shards_glob: str = "shard-*.tar",
    streams_glob: str = "stream-*",
) -> list[Path]:
    """Find either flat shards or the orchestrator's per-stream shards."""
    stream_dirs = sorted(path for path in data_dir.glob(streams_glob) if path.is_dir())
    if stream_dirs:
        return [shard for directory in stream_dirs for shard in sorted(directory.glob(shards_glob))]
    return sorted(data_dir.glob(shards_glob))


def iter_tar_pairs(tar_path: Path) -> Iterator[tuple[bytes, dict]]:
    """Yield complete WAV/JSON pairs and tolerate an actively written final shard."""
    try:
        archive = tarfile.open(tar_path, mode="r")
    except (FileNotFoundError, tarfile.TarError, OSError):
        return
    try:
        with archive:
            pending: dict[str, dict[str, bytes]] = {}
            for member in archive:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                stem, extension = os.path.splitext(name)
                if extension not in (".wav", ".json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    data = extracted.read()
                except Exception:
                    return
                entry = pending.setdefault(stem, {})
                entry[extension] = data
                if ".wav" not in entry or ".json" not in entry:
                    continue
                try:
                    metadata = json.loads(entry[".json"].decode("utf-8", "replace"))
                except Exception:
                    metadata = {}
                yield entry[".wav"], metadata
                pending.pop(stem, None)
    except (tarfile.TarError, OSError):
        return


class AudioShardDataset(IterableDataset):
    """Fixed-length clips with live discovery of newly curated shards."""

    def __init__(
        self,
        shards: list[Path],
        *,
        clip_samples: int = 32_000,
        sample_rate: int = 16_000,
        shuffle_shards: bool = True,
        seed: int = 1337,
        repeat: bool = True,
        live_data_dir: Path | None = None,
        shards_glob: str = "shard-*.tar",
        streams_glob: str = "stream-*",
        refresh_interval_sec: float = 30.0,
    ) -> None:
        super().__init__()
        self.shards = list(shards)
        self.clip_samples = int(clip_samples)
        self.sample_rate = int(sample_rate)
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.repeat = bool(repeat)
        self.live_data_dir = live_data_dir
        self.shards_glob = shards_glob
        self.streams_glob = streams_glob
        self.refresh_interval_sec = max(1.0, float(refresh_interval_sec))
        self._last_refresh = 0.0

    def _current_shards(self) -> list[Path]:
        if self.live_data_dir is None:
            return self.shards
        now = time.time()
        if now - self._last_refresh >= self.refresh_interval_sec:
            fresh = discover_shards(
                self.live_data_dir, self.shards_glob, self.streams_glob
            )
            if fresh:
                self.shards = fresh
            self._last_refresh = now
        return self.shards

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        worker_count = worker.num_workers if worker is not None else 1
        rng = random.Random(self.seed + worker_id)

        while True:
            shards = self._current_shards()[worker_id::worker_count]
            if not shards:
                if not self.repeat:
                    return
                time.sleep(min(2.0, self.refresh_interval_sec))
                continue
            order = list(shards)
            if self.shuffle_shards:
                rng.shuffle(order)
            for shard in order:
                for wav_bytes, _metadata in iter_tar_pairs(shard):
                    audio = decode_wav_bytes(wav_bytes, self.sample_rate)
                    if audio is None:
                        continue
                    if audio.numel() < self.clip_samples:
                        audio = torch.nn.functional.pad(
                            audio, (0, self.clip_samples - audio.numel())
                        )
                    elif audio.numel() > self.clip_samples:
                        start = rng.randint(0, audio.numel() - self.clip_samples)
                        audio = audio[start : start + self.clip_samples]
                    yield audio
            if not self.repeat:
                return


__all__ = ["AudioShardDataset", "discover_shards", "iter_tar_pairs"]
