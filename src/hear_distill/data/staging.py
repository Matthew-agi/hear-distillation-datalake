"""Durable, at-most-once staging for source audio downloaded to local NVMe."""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Iterator


def _write_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(time.time())
    archive.addfile(info, io.BytesIO(payload))


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


class RawSourceShardWriter:
    """Prepare a raw MP3 tar, then expose it with one atomic rename."""

    def __init__(self, stage_dir: Path, shard_index: int) -> None:
        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.final_path = self.stage_dir / f"source-{int(shard_index):08d}.tar"
        self.tmp_path = self.final_path.with_suffix(".tar.tmp")
        self.tmp_path.unlink(missing_ok=True)
        self.archive = tarfile.open(self.tmp_path, mode="w")
        self.count = 0
        self._closed = False

    def add(self, source_index: int, example: dict[str, Any]) -> bool:
        audio = example.get("audio.mp3") or {}
        mp3_bytes = audio.get("bytes")
        if not isinstance(mp3_bytes, (bytes, bytearray)) or not mp3_bytes:
            return False
        stem = f"{int(source_index):016d}"
        metadata = {
            "source_index": int(source_index),
            "__key__": str(example.get("__key__", "")),
            "__url__": str(example.get("__url__", "")),
            "metadata.json": example.get("metadata.json", {}),
            "audio_path": str(audio.get("path", "")),
        }
        _write_member(self.archive, f"{stem}.mp3", bytes(mp3_bytes))
        _write_member(self.archive, f"{stem}.json", _json_bytes(metadata))
        self.count += 1
        return True

    def prepare(self) -> None:
        if not self._closed:
            self.archive.close()
            self._closed = True

    def publish(self) -> Path | None:
        self.prepare()
        if self.count <= 0:
            self.tmp_path.unlink(missing_ok=True)
            return None
        os.replace(self.tmp_path, self.final_path)
        return self.final_path

    def abort(self) -> None:
        try:
            self.prepare()
        finally:
            self.tmp_path.unlink(missing_ok=True)


def load_download_state(stage_dir: Path) -> dict[str, Any]:
    path = Path(stage_dir) / "download_state.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_download_state(stage_dir: Path, payload: dict[str, Any]) -> None:
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / "download_state.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def clean_raw_stage(stage_dir: Path) -> int:
    """Remove incomplete and previously claimed source shards without replaying them."""
    stage_dir = Path(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for tmp_path in stage_dir.glob("source-*.tar.tmp"):
        tmp_path.unlink(missing_ok=True)
    claim_dir = stage_dir / "inflight"
    discarded = 0
    if claim_dir.exists():
        discarded = sum(1 for path in claim_dir.glob("*.tar") if path.is_file())
        shutil.rmtree(claim_dir)
    claim_dir.mkdir(parents=True, exist_ok=True)
    return discarded


def ready_raw_shards(stage_dir: Path) -> list[Path]:
    return sorted(Path(stage_dir).glob("source-*.tar"))


def claim_raw_shard(stage_dir: Path) -> Path | None:
    stage_dir = Path(stage_dir)
    claim_dir = stage_dir / "inflight"
    claim_dir.mkdir(parents=True, exist_ok=True)
    for source_path in ready_raw_shards(stage_dir):
        claimed = claim_dir / (
            f"{source_path.stem}-p{os.getpid()}-{time.time_ns()}.tar"
        )
        try:
            source_path.replace(claimed)
        except (FileNotFoundError, OSError):
            continue
        return claimed
    return None


def iter_raw_source_shard(path: Path) -> Iterator[dict[str, Any]]:
    """Reconstruct streaming-dataset examples from one staged source tar."""
    try:
        archive = tarfile.open(path, mode="r")
    except (FileNotFoundError, tarfile.TarError, OSError):
        return
    with archive:
        pending: dict[str, dict[str, bytes]] = {}
        for member in archive:
            if not member.isfile():
                continue
            member_path = Path(member.name)
            if member_path.suffix not in {".mp3", ".json"}:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            entry = pending.setdefault(member_path.stem, {})
            entry[member_path.suffix] = extracted.read()
            if ".mp3" not in entry or ".json" not in entry:
                continue
            try:
                metadata = json.loads(entry[".json"].decode("utf-8", "replace"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                metadata = {}
            yield {
                "__key__": str(metadata.get("__key__", "")),
                "__url__": str(metadata.get("__url__", "")),
                "metadata.json": metadata.get("metadata.json", {}),
                "audio.mp3": {
                    "bytes": entry[".mp3"],
                    "path": str(metadata.get("audio_path", "")),
                },
                "_source_index": int(metadata.get("source_index", -1)),
            }
            pending.pop(member_path.stem, None)


__all__ = [
    "RawSourceShardWriter",
    "claim_raw_shard",
    "clean_raw_stage",
    "iter_raw_source_shard",
    "load_download_state",
    "ready_raw_shards",
    "save_download_state",
]
