import io
import math
import wave

import numpy as np
import torch

from hear_distill.audio import AudioPreprocessor, _ema_vectorized, decode_wav_bytes


def _ema_reference(inputs: torch.Tensor, smooth_coef: float) -> torch.Tensor:
    state = inputs[:, 0, :]
    output = [state]
    for index in range(1, inputs.shape[1]):
        state = smooth_coef * inputs[:, index, :] + (1.0 - smooth_coef) * state
        output.append(state)
    return torch.stack(output, dim=1)


def test_vectorized_ema_matches_recurrence() -> None:
    generator = torch.Generator().manual_seed(7)
    inputs = torch.rand((3, 201, 128), generator=generator)
    expected = _ema_reference(inputs, 0.04)
    actual = _ema_vectorized(inputs, 0.04)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_long_vectorized_ema_is_finite() -> None:
    generator = torch.Generator().manual_seed(11)
    inputs = torch.rand((1, 3_000, 4), generator=generator)
    expected = _ema_reference(inputs, 0.04)
    actual = _ema_vectorized(inputs, 0.04)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_preprocessor_shape_and_finiteness() -> None:
    audio = torch.linspace(-1.0, 1.0, 32_000).repeat(2, 1)
    preprocessor = AudioPreprocessor()
    output = preprocessor(audio)
    assert preprocessor.fft_length == 400
    assert output.shape == (2, 1, 192, 128)
    assert torch.isfinite(output).all()


def test_pcm16_wav_fast_decode() -> None:
    sample_rate = 16_000
    samples = (np.sin(np.arange(32_000) * (2 * math.pi * 440 / sample_rate)) * 20_000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())

    decoded = decode_wav_bytes(buffer.getvalue(), sample_rate)
    assert decoded is not None
    assert decoded.shape == (32_000,)
    np.testing.assert_allclose(decoded.numpy(), samples.astype("float32") / 32768.0)
