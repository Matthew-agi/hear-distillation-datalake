from __future__ import annotations

import io
import json
import tarfile
import wave
from pathlib import Path

from hear_distill.data import AudioShardDataset, discard_claimed_shards


def _write_audio_shard(path: Path) -> None:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 32_000)
    with tarfile.open(path, "w") as archive:
        for name, payload in {
            "clip.wav": wav_buffer.getvalue(),
            "clip.json": json.dumps({"source": "test"}).encode(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_consumable_dataset_claims_each_shard_once(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    claim_dir = tmp_path / "inflight"
    train_dir.mkdir()
    shard = train_dir / "shard-000000.tar"
    _write_audio_shard(shard)
    dataset = AudioShardDataset(
        [shard],
        shuffle_shards=False,
        repeat=False,
        consume_shards=True,
        claim_dir=claim_dir,
    )

    assert len(list(dataset)) == 1
    assert not shard.exists()
    assert list(dataset) == []
    assert list(claim_dir.glob("*.tar")) == []


def test_stale_claims_are_discarded_instead_of_replayed(tmp_path: Path) -> None:
    claim_dir = tmp_path / "inflight"
    claim_dir.mkdir()
    (claim_dir / "stale.tar").write_bytes(b"not replayable")

    assert discard_claimed_shards(claim_dir) == 1
    assert list(claim_dir.iterdir()) == []
