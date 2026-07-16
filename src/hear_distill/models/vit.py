"""Size-aware construction of spectrogram ViTs with optional Canon layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn as nn

from .canon import CanonConfig, add_canon_to_vit


@dataclass(frozen=True)
class ViTSpec:
    size: str
    timm_name: str
    decoder_dim: int
    decoder_depth: int
    decoder_heads: int


VIT_SPECS: Mapping[str, ViTSpec] = {
    "tiny": ViTSpec("tiny", "vit_tiny_patch16_224", 192, 2, 6),
    "small": ViTSpec("small", "vit_small_patch16_224", 256, 2, 8),
    "base": ViTSpec("base", "vit_base_patch16_224", 384, 3, 8),
    "large": ViTSpec("large", "vit_large_patch16_224", 512, 4, 16),
}


def resolve_vit_spec(model: str) -> ViTSpec:
    key = model.strip().lower()
    if key in VIT_SPECS:
        return VIT_SPECS[key]
    for spec in VIT_SPECS.values():
        if key == spec.timm_name.lower():
            return spec
    supported = ", ".join(VIT_SPECS)
    raise ValueError(f"Unsupported ViT '{model}'. Choose one of: {supported}.")


def disable_positional_embeddings(model: nn.Module) -> None:
    position = getattr(model, "pos_embed", None)
    if position is None or not torch.is_tensor(position):
        raise ValueError("The selected ViT has no tensor positional embedding to disable.")
    with torch.no_grad():
        position.zero_()
    position.requires_grad_(False)


def build_audio_vit(
    model: str = "small",
    *,
    image_size: tuple[int, int] = (192, 128),
    in_channels: int = 1,
    pretrained: bool = False,
    canon: CanonConfig | None = None,
) -> nn.Module:
    """Build a ViT whose dimensions and Canon layers follow the chosen family size."""
    try:
        import timm
    except Exception as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError(f"timm is required to build the audio ViT: {exc}") from exc

    spec = resolve_vit_spec(model)
    try:
        encoder = timm.create_model(
            spec.timm_name,
            pretrained=pretrained,
            img_size=image_size,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not build {spec.timm_name} for image_size={image_size}: {exc}"
        ) from exc

    canon_config = canon or CanonConfig(enabled=False)
    add_canon_to_vit(encoder, canon_config)
    if canon_config.enabled and canon_config.disable_positional_encoding:
        disable_positional_embeddings(encoder)

    encoder.hear_model_size = spec.size
    encoder.hear_model_config = {
        "spec": asdict(spec),
        "image_size": list(image_size),
        "in_channels": int(in_channels),
        "pretrained": bool(pretrained),
        "canon": asdict(canon_config),
    }
    return encoder


def pooled_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return the encoder representation used by existing distilled checkpoints."""
    features = model.forward_features(images) if hasattr(model, "forward_features") else model(images)
    if isinstance(features, (list, tuple)):
        features = features[-1]
    if features.ndim == 3:
        features = features[:, 0]
    elif features.ndim == 4:
        features = features.mean(dim=(-2, -1))
    return features


__all__ = [
    "VIT_SPECS",
    "ViTSpec",
    "build_audio_vit",
    "disable_positional_embeddings",
    "pooled_features",
    "resolve_vit_spec",
]
