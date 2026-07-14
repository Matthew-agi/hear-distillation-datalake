from __future__ import annotations

import subprocess
import sys
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
    assert list((data_dir / "train").glob("shard-*.tar"))
