"""Masked spectrogram reconstruction for direct Canon-ViT pretraining.

Unlike MAE's visible-token encoder, this model keeps the complete patch grid in
the encoder and replaces masked patches with a learned token. Canon2D therefore
always receives the H x W topology it is designed to mix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from .vit import resolve_vit_spec


@dataclass(frozen=True)
class DecoderConfig:
    dim: int
    depth: int
    heads: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class ReconstructionOutput(NamedTuple):
    loss: torch.Tensor
    predictions: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor


def decoder_config_for_model(
    model: str,
    *,
    dim: int | None = None,
    depth: int | None = None,
    heads: int | None = None,
) -> DecoderConfig:
    spec = resolve_vit_spec(model)
    return DecoderConfig(
        dim=int(dim or spec.decoder_dim),
        depth=int(depth or spec.decoder_depth),
        heads=int(heads or spec.decoder_heads),
    )


class MaskedSpectrogramModel(nn.Module):
    """A Canon-compatible masked-patch encoder with a disposable decoder."""

    checkpoint_format = 1

    def __init__(
        self,
        encoder: nn.Module,
        *,
        model_size: str,
        decoder: DecoderConfig | None = None,
        mask_ratio: float = 0.75,
        normalize_patch_targets: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be between 0 and 1.")
        self.encoder = encoder
        self.model_size = resolve_vit_spec(model_size).size
        self.decoder_config = decoder or decoder_config_for_model(model_size)
        if self.decoder_config.dim <= 0 or self.decoder_config.depth <= 0:
            raise ValueError("Decoder width and depth must be positive.")
        if self.decoder_config.heads <= 0 or self.decoder_config.dim % self.decoder_config.heads:
            raise ValueError("Decoder width must be divisible by its positive head count.")
        self.mask_ratio = float(mask_ratio)
        self.normalize_patch_targets = bool(normalize_patch_targets)

        patch_embed = getattr(encoder, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        grid_size = getattr(patch_embed, "grid_size", None)
        encoder_dim = getattr(encoder, "embed_dim", None) or getattr(
            encoder, "num_features", None
        )
        if patch_size is None or grid_size is None or encoder_dim is None:
            raise ValueError("Encoder must expose patch_size, grid_size, and embedding width.")
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.grid_size = tuple(int(v) for v in grid_size)
        self.patch_count = self.grid_size[0] * self.grid_size[1]
        self.in_channels = int(getattr(patch_embed.proj, "in_channels", 1))
        self.patch_dim = self.in_channels * self.patch_size[0] * self.patch_size[1]
        self.num_prefix_tokens = int(getattr(encoder, "num_prefix_tokens", 0))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(encoder_dim)))
        self.decoder_embed = nn.Linear(int(encoder_dim), self.decoder_config.dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_count, self.decoder_config.dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.decoder_config.dim,
            nhead=self.decoder_config.heads,
            dim_feedforward=int(self.decoder_config.dim * self.decoder_config.mlp_ratio),
            dropout=self.decoder_config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_blocks = nn.TransformerEncoder(
            decoder_layer,
            num_layers=self.decoder_config.depth,
            enable_nested_tensor=False,
        )
        self.decoder_norm = nn.LayerNorm(self.decoder_config.dim)
        self.decoder_pred = nn.Linear(self.decoder_config.dim, self.patch_dim)
        self._reset_decoder_parameters()

    def _reset_decoder_parameters(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        for layer in self.decoder_blocks.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            if layer.self_attn.in_proj_bias is not None:
                nn.init.zeros_(layer.self_attn.in_proj_bias)
        for module in self.decoder_blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.zeros_(self.decoder_embed.bias)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)

    def checkpoint_metadata(self) -> dict:
        return {
            "format": self.checkpoint_format,
            "model_size": self.model_size,
            "patch_size": list(self.patch_size),
            "grid_size": list(self.grid_size),
            "patch_dim": self.patch_dim,
            "decoder": asdict(self.decoder_config),
        }

    def decoder_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = (
            "mask_token",
            "decoder_embed.",
            "decoder_pos_embed",
            "decoder_blocks.",
            "decoder_norm.",
            "decoder_pred.",
        )
        return {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith(prefixes)
        }

    def load_decoder_checkpoint(self, checkpoint: dict) -> None:
        metadata = checkpoint.get("model_config")
        if metadata != self.checkpoint_metadata():
            raise ValueError(
                "Decoder checkpoint is incompatible with the selected encoder. "
                "Model size, patch grid, and decoder dimensions must match exactly."
            )
        state = checkpoint.get("decoder")
        if not isinstance(state, dict):
            raise ValueError("Decoder checkpoint has no `decoder` state dictionary.")
        result = self.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing_decoder = [
            key
            for key in result.missing_keys
            if not key.startswith("encoder.")
        ]
        if unexpected or missing_decoder:
            raise ValueError(
                f"Invalid decoder state (missing={missing_decoder}, unexpected={unexpected})."
            )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("Spectrograms must have shape [B, C, H, W].")
        batch, channels, height, width = images.shape
        patch_h, patch_w = self.patch_size
        if channels != self.in_channels or height % patch_h or width % patch_w:
            raise ValueError(
                f"Input {tuple(images.shape)} is incompatible with patch size {self.patch_size}."
            )
        grid_h, grid_w = height // patch_h, width // patch_w
        if (grid_h, grid_w) != self.grid_size:
            raise ValueError(
                f"Input patch grid {(grid_h, grid_w)} does not match {self.grid_size}."
            )
        patches = images.reshape(
            batch, channels, grid_h, patch_h, grid_w, patch_w
        ).permute(0, 2, 4, 1, 3, 5)
        return patches.reshape(batch, grid_h * grid_w, self.patch_dim)

    def random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        masked = max(1, min(self.patch_count - 1, round(self.patch_count * self.mask_ratio)))
        noise = torch.rand(batch_size, self.patch_count, device=device)
        order = noise.argsort(dim=1)
        mask = torch.zeros_like(noise, dtype=torch.bool)
        mask.scatter_(1, order[:, :masked], True)
        return mask

    def _encode(self, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder.patch_embed(images)
        if tokens.ndim != 3 or tokens.shape[1] != self.patch_count:
            raise RuntimeError("The encoder did not produce the configured full patch grid.")
        replacement = self.mask_token.to(dtype=tokens.dtype).expand(tokens.shape[0], -1, -1)
        tokens = torch.where(mask.unsqueeze(-1), replacement, tokens)
        tokens = self.encoder._pos_embed(tokens)
        tokens = self.encoder.patch_drop(tokens)
        tokens = self.encoder.norm_pre(tokens)
        for block in self.encoder.blocks:
            tokens = block(tokens)
        tokens = self.encoder.norm(tokens)
        if self.num_prefix_tokens:
            tokens = tokens[:, self.num_prefix_tokens :]
        return tokens

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ReconstructionOutput:
        if mask is None:
            mask = self.random_mask(images.shape[0], images.device)
        if mask.shape != (images.shape[0], self.patch_count):
            raise ValueError(
                f"Mask must have shape {(images.shape[0], self.patch_count)}, got {tuple(mask.shape)}."
            )
        mask = mask.bool()
        encoded = self._encode(images, mask)
        decoded = self.decoder_embed(encoded) + self.decoder_pos_embed.to(encoded.dtype)
        decoded = self.decoder_blocks(decoded)
        predictions = self.decoder_pred(self.decoder_norm(decoded))

        target = self.patchify(images)
        if self.normalize_patch_targets:
            mean = target.mean(dim=-1, keepdim=True)
            variance = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / torch.sqrt(variance + 1e-6)
        per_patch_loss = (predictions.float() - target.float()).square().mean(dim=-1)
        loss = (per_patch_loss * mask.float()).sum() / mask.sum().clamp_min(1)
        return ReconstructionOutput(loss, predictions, mask, target)


__all__ = [
    "DecoderConfig",
    "MaskedSpectrogramModel",
    "ReconstructionOutput",
    "decoder_config_for_model",
]
