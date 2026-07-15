"""Iterable access to the LAION-Audio tar shards produced by the data lake."""

from __future__ import annotations

import json
import os
import random
import shutil
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


def discard_claimed_shards(claim_dir: Path) -> int:
    """Discard shards claimed by an earlier process so they can never be replayed."""
    if not claim_dir.exists():
        claim_dir.mkdir(parents=True, exist_ok=True)
        return 0
    discarded = sum(1 for path in claim_dir.rglob("*.tar") if path.is_file())
    shutil.rmtree(claim_dir)
    claim_dir.mkdir(parents=True, exist_ok=True)
    return discarded


def _claim_shard(shard: Path, claim_dir: Path, worker_id: int) -> Path | None:
    """Atomically remove one shard from the shared train inventory."""
    claim_dir.mkdir(parents=True, exist_ok=True)
    claimed = claim_dir / (
        f"{shard.parent.name}-{shard.stem}-w{worker_id}-p{os.getpid()}-{time.time_ns()}.tar"
    )
    try:
        shard.replace(claimed)
    except (FileNotFoundError, OSError):
        return None
    return claimed


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
        consume_shards: bool = False,
        claim_dir: Path | None = None,
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
        self.consume_shards = bool(consume_shards)
        self.claim_dir = claim_dir
        self._last_refresh = 0.0
        if self.consume_shards and self.claim_dir is None:
            raise ValueError("claim_dir is required when consume_shards is enabled.")

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
            if self.consume_shards and self.live_data_dir is not None:
                shards = discover_shards(
                    self.live_data_dir, self.shards_glob, self.streams_glob
                )
            else:
                shards = self._current_shards()
            if not self.consume_shards:
                shards = shards[worker_id::worker_count]
            if not shards:
                if not self.repeat and self.live_data_dir is None:
                    return
                time.sleep(min(2.0, self.refresh_interval_sec))
                continue
            order = list(shards)
            if self.shuffle_shards:
                rng.shuffle(order)
            for shard in order:
                claimed = None
                if self.consume_shards:
                    assert self.claim_dir is not None
                    claimed = _claim_shard(shard, self.claim_dir, worker_id)
                    if claimed is None:
                        continue
                    shard = claimed
                try:
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
                finally:
                    if claimed is not None:
                        claimed.unlink(missing_ok=True)
            if not self.repeat:
                if self.consume_shards and self.live_data_dir is not None:
                    time.sleep(min(1.0, self.refresh_interval_sec))
                    continue
                return


__all__ = [
    "AudioShardDataset",
    "discard_claimed_shards",
    "discover_shards",
    "iter_tar_pairs",
]
