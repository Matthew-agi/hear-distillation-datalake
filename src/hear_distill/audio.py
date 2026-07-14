# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained, accelerated HeAR audio preprocessing.

This is a compatible PyTorch implementation of Google Health's HeAR
``preprocess_audio`` routine. Constants are constructed once and the PCEN EMA
is evaluated in a vectorized form instead of rebuilding matrices and launching
one matrix multiplication per time step on every batch.
"""

from __future__ import annotations

import math
import io
import struct
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def decode_wav_bytes(wav_bytes: bytes, target_sr: int) -> torch.Tensor | None:
    """Decode WAV bytes, with a zero-copy-parser fast path for pipeline PCM16."""
    fmt: tuple[int, int, int, int] | None = None
    data_range: tuple[int, int] | None = None
    if len(wav_bytes) >= 12 and wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE":
        offset = 12
        while offset + 8 <= len(wav_bytes):
            chunk_id = wav_bytes[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
            payload_start = offset + 8
            payload_end = min(len(wav_bytes), payload_start + chunk_size)
            if chunk_id == b"fmt " and payload_end - payload_start >= 16:
                audio_format, channels, sample_rate, _byte_rate, _align, bits = struct.unpack_from(
                    "<HHIIHH", wav_bytes, payload_start
                )
                fmt = (audio_format, channels, sample_rate, bits)
            elif chunk_id == b"data":
                data_range = (payload_start, payload_end)
            offset = payload_start + chunk_size + (chunk_size & 1)

    if fmt is not None and data_range is not None:
        audio_format, channels, sample_rate, bits = fmt
        if audio_format == 1 and bits == 16 and channels > 0 and sample_rate == target_sr:
            import numpy as np

            start, end = data_range
            pcm = np.frombuffer(wav_bytes, dtype="<i2", offset=start, count=(end - start) // 2)
            if channels > 1:
                pcm = pcm[: (pcm.size // channels) * channels].reshape(-1, channels).mean(axis=1)
            audio = pcm.astype("float32", copy=True)
            audio *= 1.0 / 32768.0
            return torch.from_numpy(audio)

    try:
        import soundfile as sf

        with io.BytesIO(wav_bytes) as bio:
            audio, sample_rate = sf.read(bio, dtype="float32", always_2d=False)
    except Exception:
        return None
    if audio is None:
        return None
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != target_sr:
        from scipy import signal

        new_len = int(round(audio.shape[0] * (target_sr / sample_rate)))
        audio = signal.resample(audio, new_len).astype("float32", copy=False)
    return torch.from_numpy(audio).float()


def _enclosing_power_of_two(value: int) -> int:
    return int(2 ** math.ceil(math.log2(value))) if value > 0 else 1


def _hertz_to_mel(frequencies_hertz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + frequencies_hertz / 700.0)


def _linear_to_mel_weight_matrix(
    *,
    num_mel_bins: int = 128,
    num_spectrogram_bins: int = 201,
    sample_rate: float = 16_000,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: float = 8_000.0,
) -> torch.Tensor:
    """Build the TensorFlow-compatible HTK mel projection matrix."""
    if num_mel_bins <= 0 or num_spectrogram_bins <= 0 or sample_rate <= 0:
        raise ValueError("Mel dimensions and sample rate must be positive.")
    if not 0.0 <= lower_edge_hertz < upper_edge_hertz <= sample_rate / 2.0:
        raise ValueError("Invalid mel frequency range.")

    dtype = torch.float32
    zero = torch.tensor(0.0, dtype=dtype)
    linear_frequencies = torch.linspace(
        zero,
        torch.tensor(sample_rate / 2.0, dtype=dtype),
        num_spectrogram_bins,
        dtype=dtype,
    )[1:]
    spectrogram_bins_mel = _hertz_to_mel(linear_frequencies).unsqueeze(1)
    band_edges_mel = torch.linspace(
        _hertz_to_mel(torch.tensor(lower_edge_hertz, dtype=dtype)),
        _hertz_to_mel(torch.tensor(upper_edge_hertz, dtype=dtype)),
        num_mel_bins + 2,
        dtype=dtype,
    ).unfold(0, 3, 1)
    lower_edge_mel = band_edges_mel[:, 0].unsqueeze(0)
    center_mel = band_edges_mel[:, 1].unsqueeze(0)
    upper_edge_mel = band_edges_mel[:, 2].unsqueeze(0)
    lower_slopes = (spectrogram_bins_mel - lower_edge_mel) / (
        center_mel - lower_edge_mel
    )
    upper_slopes = (upper_edge_mel - spectrogram_bins_mel) / (
        upper_edge_mel - center_mel
    )
    weights = torch.maximum(zero, torch.minimum(lower_slopes, upper_slopes))
    return F.pad(weights, (0, 0, 1, 0), mode="constant", value=0.0)


def _ema_vectorized(
    inputs: torch.Tensor,
    smooth_coef: float,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Evaluate the HeAR PCEN EMA in numerically stable vectorized chunks.

    A single closed-form scan is fast for the roughly 200 frames in the
    training input, but inverse decay powers overflow on sufficiently long
    full-clip evaluation inputs. Chunking carries only the last EMA state
    between bounded scans and does not add a loop to the normal training path.
    """
    if inputs.ndim != 3:
        raise ValueError(f"Expected [batch, time, channels], got {inputs.shape}.")
    timesteps = int(inputs.shape[1])
    if timesteps <= 1:
        return inputs

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    alpha = float(smooth_coef)
    beta = 1.0 - alpha
    beta_t = torch.tensor(beta, device=inputs.device, dtype=inputs.dtype)
    previous = inputs[:, 0, :]
    outputs = [inputs[:, :1, :]]
    for start in range(1, timesteps, chunk_size):
        chunk = inputs[:, start : start + chunk_size, :]
        offsets = torch.arange(chunk.shape[1], device=inputs.device, dtype=inputs.dtype)
        powers = torch.pow(beta_t, offsets).view(1, -1, 1)
        decayed_inputs = powers * torch.cumsum(chunk / powers, dim=1)
        carried_state = (powers * beta_t) * previous.unsqueeze(1)
        chunk_output = carried_state + alpha * decayed_inputs
        outputs.append(chunk_output)
        previous = chunk_output[:, -1, :]
    return torch.cat(outputs, dim=1)


class AudioPreprocessor(nn.Module):
    """Convert two-second 16 kHz waveforms to HeAR mel-PCEN images."""

    def __init__(self) -> None:
        super().__init__()
        frame_length = 16 * 25
        self.frame_length = frame_length
        self.frame_step = 160
        # HeAR explicitly uses a 400-point FFT rather than the enclosing power
        # of two used by the generic STFT helper default.
        self.fft_length = frame_length
        self.register_buffer(
            "window",
            torch.hann_window(frame_length, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mel_transform",
            _linear_to_mel_weight_matrix(
                num_spectrogram_bins=self.fft_length // 2 + 1
            ),
            persistent=False,
        )

    def _stft(self, signals: torch.Tensor) -> torch.Tensor:
        n_frames = math.ceil(signals.shape[-1] / self.frame_step)
        padded_length = max(0, (n_frames - 1) * self.frame_step + self.frame_length)
        padding_needed = max(0, padded_length - signals.shape[-1])
        if padding_needed:
            signals = F.pad(signals, (0, padding_needed))
        framed = signals.unfold(-1, self.frame_length, self.frame_step)
        framed = framed * self.window.to(dtype=framed.dtype)
        return torch.fft.rfft(framed, n=self.fft_length, dim=-1)

    def _mel_pcen(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.float()
        x = x - torch.amin(x)
        x = x / (torch.amax(x) + 1e-8)
        x = (x * 2.0) - 1.0

        spectrograms = torch.square(torch.abs(self._stft(x)))
        mel_spectrograms = torch.matmul(spectrograms, self.mel_transform)
        smoother = _ema_vectorized(mel_spectrograms, smooth_coef=0.04)
        pcen = (
            mel_spectrograms / torch.pow(1e-8 + smoother, 0.8) + 2.0
        ) ** 0.5 - math.sqrt(2.0)
        return pcen

    def _resize(self, pcen: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            pcen.unsqueeze(1),
            size=(192, 128),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Input audio must have rank 2, got rank {audio.ndim}.")
        if audio.shape[1] < 32_000:
            audio = F.pad(audio, (0, 32_000 - audio.shape[1]))
        elif audio.shape[1] > 32_000:
            raise ValueError(f"Input audio must have 32000 samples, got {audio.shape[1]}.")
        return self._resize(self._mel_pcen(audio))

    def forward_full_clip(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"Input audio must have rank 2, got rank {audio.ndim}.")
        if audio.shape[1] <= 0:
            audio = F.pad(audio, (0, 1))
        return self._resize(self._mel_pcen(audio))


_PREPROCESSORS: Dict[Tuple[str, int | None], AudioPreprocessor] = {}


def preprocess_audio(audio: torch.Tensor) -> torch.Tensor:
    """Drop-in HeAR preprocessing function with per-device constant caching."""
    key = (audio.device.type, audio.device.index)
    module = _PREPROCESSORS.get(key)
    if module is None:
        module = AudioPreprocessor().eval().to(audio.device)
        _PREPROCESSORS[key] = module
    return module(audio)


def preprocess_audio_full_clip(audio: torch.Tensor) -> torch.Tensor:
    """HeAR preprocessing for evaluator inputs longer than two seconds."""
    key = (audio.device.type, audio.device.index)
    module = _PREPROCESSORS.get(key)
    if module is None:
        module = AudioPreprocessor().eval().to(audio.device)
        _PREPROCESSORS[key] = module
    return module.forward_full_clip(audio)


__all__ = [
    "AudioPreprocessor",
    "decode_wav_bytes",
    "preprocess_audio",
    "preprocess_audio_full_clip",
]
