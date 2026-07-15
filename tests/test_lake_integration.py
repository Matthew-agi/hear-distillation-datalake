from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


def test_orchestrator_bootstraps_curates_and_starts_training(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "lake"
    command = [
        sys.executable,
        str(root / "datalake" / "run_lake.py"),
        "--repo-root",
        str(root),
        "--data-dir",
        str(data_dir),
        "--train-out",
        str(tmp_path / "checkpoints"),
        "--train-script",
        str(root / "tests" / "fixtures" / "fake_train.py"),
        "--stream-script",
        str(root / "tests" / "fixtures" / "fake_stream.py"),
        "--python",
        sys.executable,
        "--num-streams",
        "1",
        "--min-streams",
        "1",
        "--chunk-clips-per-stream",
        "1",
        "--reserve-low-clips",
        "1",
        "--reserve-high-clips",
        "2",
        "--shard-size",
        "1",
        "--curation-shard-size",
        "1",
        "--train-batch-size",
        "1",
        "--train-num-workers",
        "1",
        "--decay-steps",
        "0",
        "--decay-retain-ratio",
        "0.5",
        "--stream-extra-args",
        "--fake-clips 4",
        "--poll-sec",
        "0.01",
        "--status-every-sec",
        "0.05",
        "--train-extra-args",
        "--max-steps 1 --val-fraction 0",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=15)
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "waiting for first curated train shard" in output
    assert "[train] step=1" in output
    assert "stable training exited rc=0" in output
    train_shards = list((data_dir / "train").glob("shard-*.tar"))
    decay_shards = list((data_dir / "decay").glob("shard-*.tar"))
    assert train_shards
    assert decay_shards

    def wav_stems(shards: list[Path]) -> set[str]:
        stems: set[str] = set()
        for shard_path in shards:
            with tarfile.open(shard_path) as archive:
                stems.update(Path(member.name).stem for member in archive if member.name.endswith(".wav"))
        return stems

    train_stems = wav_stems(train_shards)
    decay_stems = wav_stems(decay_shards)
    assert train_stems
    assert decay_stems
    assert train_stems.isdisjoint(decay_stems)
