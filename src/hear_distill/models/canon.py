"""Canon token-mixing layers shared by every training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CanonConfig:
    """Placement and topology of Canon layers in a ViT encoder."""

    enabled: bool = True
    use_2d: bool = True
    kernel_size: int = 4
    a: bool = True
    b: bool = True
    b_qkv: bool = False
    c: bool = True
    d: bool = True
    causal: bool = False
    disable_positional_encoding: bool = True


class CanonLayer(nn.Module):
    """Depthwise residual convolution over a token sequence."""

    def __init__(self, dim: int, kernel_size: int = 4, causal: bool = False) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=self.kernel_size,
            groups=dim,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        if self.causal:
            pad_left, pad_right = self.kernel_size - 1, 0
        else:
            pad_left = (self.kernel_size - 1) // 2
            pad_right = self.kernel_size // 2
        y = self.conv(F.pad(y, (pad_left, pad_right))).transpose(1, 2)
        return x + y


class Canon2DLayer(nn.Module):
    """Depthwise residual convolution over the spectrogram patch grid."""

    def __init__(
        self,
        dim: int,
        kernel_h: int,
        kernel_w: int,
        causal_time: bool = False,
        *,
        grid_size: Optional[tuple[int, int]] = None,
        expect_cls: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.kernel_h = int(kernel_h)
        self.kernel_w = int(kernel_w)
        self.causal_time = bool(causal_time)
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=(self.kernel_h, self.kernel_w),
            groups=dim,
            bias=True,
        )
        self.grid_size = tuple(grid_size) if grid_size is not None else None
        self.expect_cls = expect_cls
        self._warned = False
        self._fallback = CanonLayer(dim, kernel_size=self.kernel_h, causal=self.causal_time)

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(message, flush=True)
            self._warned = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError("Canon2D expects input of shape [B, N, C].")
        if self.grid_size is None:
            self._warn_once("Warning: Canon2D missing grid_size; falling back to 1D Canon.")
            return self._fallback(x)

        height, width = (int(v) for v in self.grid_size)
        batch, token_count, channels = x.shape
        patch_count = height * width
        if self.expect_cls is True and token_count != patch_count + 1:
            self._warn_once(
                "Warning: Canon2D expected a CLS token and full patch grid; "
                "falling back to 1D Canon."
            )
            return self._fallback(x)

        if token_count == patch_count + 1:
            prefix, patches = x[:, :1], x[:, 1:]
        elif token_count == patch_count:
            prefix, patches = None, x
        else:
            self._warn_once(
                f"Warning: Canon2D token mismatch (N={token_count}, H*W={patch_count}); "
                "falling back to 1D Canon."
            )
            return self._fallback(x)

        patches = patches.transpose(1, 2).contiguous().view(
            batch, channels, height, width
        )
        if self.causal_time:
            pad_top, pad_bottom = self.kernel_h - 1, 0
        else:
            pad_top = (self.kernel_h - 1) // 2
            pad_bottom = self.kernel_h // 2
        pad_left = (self.kernel_w - 1) // 2
        pad_right = self.kernel_w // 2
        mixed = self.conv(F.pad(patches, (pad_left, pad_right, pad_top, pad_bottom)))
        mixed = mixed.view(batch, channels, patch_count).transpose(1, 2)
        if prefix is not None:
            mixed = torch.cat((torch.zeros_like(prefix), mixed), dim=1)
        return x + mixed


class CanonInputWrapper(nn.Module):
    def __init__(self, module: nn.Module, canon: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.canon = canon

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.module(self.canon(x), *args, **kwargs)


class CanonQKVWrapper(nn.Module):
    def __init__(self, qkv: nn.Module, canon: nn.Module) -> None:
        super().__init__()
        self.qkv = qkv
        self.canon = canon

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        projected = self.qkv(x, *args, **kwargs)
        if projected.ndim != 3:
            raise RuntimeError("Canon-B expects QKV output of shape [B, N, 3*D].")
        return self.canon(projected)


class CanonFC1Wrapper(nn.Module):
    def __init__(self, fc1: nn.Module, canon: nn.Module) -> None:
        super().__init__()
        self.fc1 = fc1
        self.canon = canon

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        projected = self.fc1(x, *args, **kwargs)
        if projected.ndim != 3:
            raise RuntimeError("Canon-D expects MLP FC1 output of shape [B, N, M].")
        return self.canon(projected)


class CanonBlockWrapper(nn.Module):
    """Insert Canon layers without assuming a particular ViT width."""

    def __init__(
        self,
        block: nn.Module,
        dim: int,
        *,
        config: CanonConfig,
        grid_size: Optional[tuple[int, int]],
        expect_cls: Optional[bool],
    ) -> None:
        super().__init__()
        self.block = block
        self.use_2d = bool(config.use_2d)
        self.grid_size = tuple(grid_size) if grid_size is not None else None
        self.canon_b_qkv = bool(config.b_qkv)
        self.expect_cls = expect_cls
        self._insert_canon(dim=int(dim), config=config)

    def _make_canon(self, dim: int, config: CanonConfig) -> nn.Module:
        if self.use_2d:
            return Canon2DLayer(
                dim,
                config.kernel_size,
                config.kernel_size,
                causal_time=config.causal,
                grid_size=self.grid_size,
                expect_cls=self.expect_cls,
            )
        return CanonLayer(dim, kernel_size=config.kernel_size, causal=config.causal)

    def _insert_canon(self, *, dim: int, config: CanonConfig) -> None:
        block = self.block
        if config.b:
            if not hasattr(block, "attn"):
                raise ValueError("Canon-B requested but block has no `.attn`.")
            attention = block.attn
            if config.b_qkv:
                if not hasattr(attention, "qkv"):
                    raise ValueError("Canon-B(QKV) requested but attention has no `.qkv`.")
                qkv = attention.qkv
                qkv_dim = getattr(qkv, "out_features", None)
                if qkv_dim is None:
                    raise ValueError("Could not determine QKV output width for Canon-B.")
                attention.qkv = CanonQKVWrapper(
                    qkv, self._make_canon(int(qkv_dim), config)
                )
            else:
                if not hasattr(attention, "proj"):
                    raise ValueError("Canon-B requested but attention has no `.proj`.")
                attention.proj = nn.Sequential(
                    attention.proj, self._make_canon(dim, config)
                )

        if config.a:
            if not hasattr(block, "attn"):
                raise ValueError("Canon-A requested but block has no `.attn`.")
            block.attn = CanonInputWrapper(block.attn, self._make_canon(dim, config))

        if config.d:
            if not hasattr(block, "mlp") or not hasattr(block.mlp, "fc1"):
                raise ValueError("Canon-D requested but block MLP has no `.fc1`.")
            fc1 = block.mlp.fc1
            hidden_dim = getattr(fc1, "out_features", None)
            if hidden_dim is None:
                raise ValueError("Could not determine MLP hidden width for Canon-D.")
            block.mlp.fc1 = CanonFC1Wrapper(
                fc1, self._make_canon(int(hidden_dim), config)
            )

        if config.c:
            if not hasattr(block, "mlp"):
                raise ValueError("Canon-C requested but block has no `.mlp`.")
            block.mlp = CanonInputWrapper(block.mlp, self._make_canon(dim, config))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def add_canon_to_vit(model: nn.Module, config: CanonConfig) -> nn.Module:
    """Adapt Canon to the actual width, MLP width, and patch grid of a ViT."""
    if not config.enabled:
        return model
    if config.kernel_size <= 0:
        raise ValueError("Canon kernel size must be positive.")
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("The selected model has no `.blocks` ViT stack.")
    dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    if dim is None:
        raise ValueError("Could not determine the ViT embedding width.")

    patch_embed = getattr(model, "patch_embed", None)
    raw_grid = getattr(patch_embed, "grid_size", None)
    grid_size = tuple(int(v) for v in raw_grid) if raw_grid is not None else None
    num_prefix = int(getattr(model, "num_prefix_tokens", 0))
    if config.use_2d and (grid_size is None or num_prefix not in (0, 1)):
        raise ValueError(
            "Canon2D requires a known patch grid and zero or one prefix token."
        )
    expect_cls = bool(num_prefix) if num_prefix in (0, 1) else None

    for index in range(len(blocks)):
        blocks[index] = CanonBlockWrapper(
            blocks[index],
            int(dim),
            config=config,
            grid_size=grid_size,
            expect_cls=expect_cls,
        )
    return model


__all__ = [
    "Canon2DLayer",
    "CanonBlockWrapper",
    "CanonConfig",
    "CanonLayer",
    "add_canon_to_vit",
]
