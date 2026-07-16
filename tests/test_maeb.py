from __future__ import annotations

import numpy as np

from hear_distill.evaluation.maeb import canon_config_from_checkpoint, prepare_audio


def test_prepare_audio_center_crops_and_pads() -> None:
    long = np.arange(10, dtype=np.float32)
    assert prepare_audio(long, clip_samples=4).tolist() == [3.0, 4.0, 5.0, 6.0]
    short = np.array([1.0, 2.0], dtype=np.float32)
    assert prepare_audio(short, clip_samples=4).tolist() == [1.0, 2.0, 0.0, 0.0]


def test_prepare_audio_mixes_channels_to_mono() -> None:
    stereo = np.array([[1.0, 3.0], [3.0, 5.0]], dtype=np.float32)
    assert prepare_audio(stereo, clip_samples=2).tolist() == [2.0, 4.0]


def test_checkpoint_canon_abcd_enables_all_placements() -> None:
    config = canon_config_from_checkpoint(
        {
            "canon": True,
            "canon_2d": True,
            "canon_abcd": True,
            "canon_no_pos_enc": True,
        }
    )
    assert config.enabled and config.use_2d and config.disable_positional_encoding
    assert config.a and config.b and config.c and config.d
