from __future__ import annotations

import io
import json
import tarfile
import wave
from pathlib import Path

import torch

from evaluate_distilled_hear import _load_student_and_proj
from hear_distill.training.pretrain import main


def _write_audio_shard(path: Path) -> None:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * 32_000)
    members = {
        "clip.wav": wav_buffer.getvalue(),
        "clip.json": json.dumps({"source": "test"}).encode(),
    }
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _args(data_dir: Path, out_dir: Path, max_steps: int) -> list[str]:
    return [
        "--data-dir",
        str(data_dir),
        "--out",
        str(out_dir),
        "--model-size",
        "tiny",
        "--decoder-dim",
        "96",
        "--decoder-depth",
        "1",
        "--decoder-heads",
        "4",
        "--max-steps",
        str(max_steps),
        "--batch-size",
        "1",
        "--num-workers",
        "0",
        "--device",
        "cpu",
        "--lr-warmup-steps",
        "0",
        "--no-amp",
        "--no-compile-model",
        "--save-every",
        "0",
        "--log-every",
        "1",
        "--no-consume-shards",
        "--repeat",
    ]


def test_direct_pretraining_checkpoint_and_resume(tmp_path: Path) -> None:
    data_dir = tmp_path / "train"
    data_dir.mkdir()
    _write_audio_shard(data_dir / "shard-000000.tar")
    out_dir = tmp_path / "checkpoints"

    assert main(_args(data_dir, out_dir, 1)) == 0
    checkpoint_path = out_dir / "ckpt_final.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["objective"] == "masked-reconstruction"
    assert checkpoint["step"] == 1
    assert checkpoint["model_config"]["grid_size"] == [12, 8]
    assert "encoder" in checkpoint and "decoder" in checkpoint
    assert checkpoint["adaptive_warmup_state"]["current_batch_size"] == 1
    assert checkpoint["memory_plan"]["batch_cap"] == 1

    resume_args = _args(data_dir, out_dir, 2) + ["--resume-from", str(checkpoint_path)]
    assert main(resume_args) == 0
    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert resumed["step"] == 2

    encoder, head, checkpoint_args, embedding_dim = _load_student_and_proj(
        checkpoint_path,
        torch.device("cpu"),
        allow_unsafe=False,
        embedding_head="proj",
    )
    assert checkpoint_args["model_size"] == "tiny"
    assert embedding_dim == 192
    assert isinstance(head, torch.nn.Identity)
    assert encoder.embed_dim == 192
