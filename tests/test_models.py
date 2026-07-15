from __future__ import annotations

import torch

from hear_distill.models import (
    CanonConfig,
    MaskedSpectrogramModel,
    build_audio_vit,
    estimate_training_memory,
)
from hear_distill.models.canon import CanonBlockWrapper, CanonFC1Wrapper, CanonInputWrapper
from hear_distill.models.reconstruction import DecoderConfig
from hear_distill.models.vit import VIT_SPECS


def test_supported_vit_family_has_size_aware_defaults() -> None:
    assert list(VIT_SPECS) == ["tiny", "small", "base", "large"]
    assert VIT_SPECS["large"].decoder_dim > VIT_SPECS["small"].decoder_dim
    saved_bytes = [
        estimate_training_memory(size, "reconstruct").saved_bytes_per_sample
        for size in VIT_SPECS
    ]
    assert saved_bytes == sorted(saved_bytes)


def test_memory_estimate_uses_exact_decoder_configuration() -> None:
    default = estimate_training_memory("tiny", "reconstruct")
    compact = estimate_training_memory(
        "tiny",
        "reconstruct",
        decoder_config=DecoderConfig(dim=96, depth=1, heads=4),
    )

    assert compact.trainable_parameters < default.trainable_parameters
    assert compact.saved_bytes_per_sample < default.saved_bytes_per_sample


def test_canon_adapts_to_patch_grid_and_mlp_width() -> None:
    encoder = build_audio_vit("tiny", canon=CanonConfig())
    first = encoder.blocks[0]

    assert isinstance(first, CanonBlockWrapper)
    assert first.grid_size == (12, 8)
    assert isinstance(first.block.mlp, CanonInputWrapper)
    assert isinstance(first.block.mlp.module.fc1, CanonFC1Wrapper)
    hidden_canon = first.block.mlp.module.fc1.canon
    assert hidden_canon.conv.in_channels == 768
    assert encoder.embed_dim == 192


def test_masked_reconstruction_preserves_full_canon_grid() -> None:
    torch.manual_seed(7)
    encoder = build_audio_vit("tiny", canon=CanonConfig())
    model = MaskedSpectrogramModel(
        encoder,
        model_size="tiny",
        decoder=DecoderConfig(dim=96, depth=1, heads=4),
        mask_ratio=0.75,
    )
    images = torch.randn(1, 1, 192, 128)
    output = model(images)

    assert output.predictions.shape == (1, 96, 256)
    assert output.target.shape == (1, 96, 256)
    assert output.mask.shape == (1, 96)
    assert output.mask.sum().item() == 72
    assert torch.isfinite(output.loss)


def test_decoder_reuse_requires_exact_metadata() -> None:
    config = DecoderConfig(dim=96, depth=1, heads=4)
    source = MaskedSpectrogramModel(
        build_audio_vit("tiny", canon=CanonConfig()),
        model_size="tiny",
        decoder=config,
    )
    checkpoint = {
        "model_config": source.checkpoint_metadata(),
        "decoder": source.decoder_state_dict(),
    }
    target = MaskedSpectrogramModel(
        build_audio_vit("tiny", canon=CanonConfig()),
        model_size="tiny",
        decoder=config,
    )
    target.load_decoder_checkpoint(checkpoint)

    bad = dict(checkpoint)
    bad["model_config"] = {**checkpoint["model_config"], "model_size": "small"}
    try:
        target.load_decoder_checkpoint(bad)
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Mismatched decoder metadata was accepted.")
