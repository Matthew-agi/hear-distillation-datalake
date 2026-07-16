from pathlib import Path

from hear_distill.cli import main


def test_reconstruction_dry_run_selects_new_trainer_and_model(
    tmp_path: Path, capsys
) -> None:
    result = main(
        [
            "run",
            "--objective",
            "reconstruct",
            "--model-size",
            "large",
            "--data-dir",
            str(tmp_path / "lake"),
            "--max-steps",
            "1000",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert result == 0
    assert "objective=reconstruct model=large" in output
    assert "scripts/train/pretrain_reconstruction.py" in output
    assert "--python" in output
    assert "--model-size large" in output
    assert "--auto-warmup" in output
    assert "--auto-warmup-max-batch-size" in output
    assert "--lr-schedule none" in output
    assert "cosine" not in output
    assert "--optimizer-mode teacher-superbatch" not in output
