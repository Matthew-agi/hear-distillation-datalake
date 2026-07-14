import io
import json
import sys
import tarfile
import time
import wave

import stream_laion_audio_clips as streaming
from stream_laion_audio_clips import ClipResult, _segment_starts, _wav_bytes_from_pcm16_mono


def test_segment_starts_adds_only_eligible_overlap_tail() -> None:
    assert _segment_starts(32_000, clip_samples=32_000, overlap_threshold_samples=16_000) == [0]
    assert _segment_starts(80_000, clip_samples=32_000, overlap_threshold_samples=16_000) == [0, 32_000, 48_000]


def test_pcm_wrapper_writes_valid_wav() -> None:
    pcm = b"\x00\x00" * 32_000
    payload = _wav_bytes_from_pcm16_mono(pcm, sample_rate=16_000)
    with wave.open(io.BytesIO(payload), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.getnframes() == 32_000


def test_parallel_decode_pipeline_preserves_source_order(tmp_path, monkeypatch) -> None:
    examples = [{"__key__": f"item-{index}"} for index in range(6)]

    def fake_stream(**_kwargs):
        return iter(examples)

    def fake_decode(example, **_kwargs):
        index = int(example["__key__"].split("-")[-1])
        time.sleep((6 - index) * 0.002)
        clip_id = example["__key__"]
        return [ClipResult(clip_id=clip_id, wav_bytes=b"pcm", meta={"clip_id": clip_id})]

    monkeypatch.setattr(streaming, "_iter_laion_stream", fake_stream)
    monkeypatch.setattr(streaming, "_make_clips_from_example", fake_decode)
    monkeypatch.setattr(streaming, "_which", lambda _command: "/usr/bin/true")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream_laion_audio_clips.py",
            "--out",
            str(tmp_path),
            "--num-clips",
            "6",
            "--shard-size",
            "6",
            "--decode-workers",
            "3",
            "--decode-prefetch",
            "6",
        ],
    )

    streaming.main()

    with tarfile.open(tmp_path / "shard-000000.tar") as shard:
        wav_names = [member.name for member in shard if member.name.endswith(".wav")]
    assert wav_names == [f"item-{index}.wav" for index in range(6)]
    state = json.loads((tmp_path / "resume_state.json").read_text())
    assert state["seen"] == 6
