#!/usr/bin/env python3
"""Microbenchmark the optimized preprocessing and PCM16 decode paths."""

from __future__ import annotations

import argparse
import io
import statistics
import time
import wave

import numpy as np
import torch

from hear_distill.audio import AudioPreprocessor, decode_wav_bytes


def _measure(fn, repeats: int) -> tuple[float, float]:
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        values.append(time.perf_counter() - started)
    return statistics.median(values), min(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    audio = torch.rand((args.batch_size, 32_000), device=device) * 2.0 - 1.0
    preprocessor = AudioPreprocessor().to(device).eval()
    for _ in range(2):
        preprocessor(audio)
    if device.type == "cuda":
        torch.cuda.synchronize()

    def preprocess() -> None:
        preprocessor(audio)
        if device.type == "cuda":
            torch.cuda.synchronize()

    pcm = (np.random.default_rng(7).standard_normal(32_000).clip(-1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())
    wav_bytes = buffer.getvalue()

    pre_median, pre_best = _measure(preprocess, args.repeats)
    decode_median, decode_best = _measure(
        lambda: decode_wav_bytes(wav_bytes, 16_000),
        args.repeats * 10,
    )
    print(
        f"preprocess batch={args.batch_size} device={device.type} "
        f"median_ms={pre_median*1000:.2f} best_ms={pre_best*1000:.2f} "
        f"clips_per_s={args.batch_size/pre_median:.1f}"
    )
    print(
        f"pcm16_decode median_ms={decode_median*1000:.3f} "
        f"best_ms={decode_best*1000:.3f} clips_per_s={1.0/decode_median:.1f}"
    )


if __name__ == "__main__":
    main()
