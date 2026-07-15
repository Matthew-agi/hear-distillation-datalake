"""Architecture-derived training-memory estimates.

The estimate is deliberately based on the selected model's autograd graph, not
on a table of model-name multipliers.  A live CUDA calibration in the trainer
remains authoritative because compiler workspaces and allocator behaviour are
device and PyTorch-version dependent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn

from .canon import CanonConfig
from .reconstruction import DecoderConfig, MaskedSpectrogramModel
from .vit import build_audio_vit, pooled_features, resolve_vit_spec


@dataclass(frozen=True)
class TrainingMemoryEstimate:
    model_size: str
    objective: str
    trainable_parameters: int
    persistent_training_bytes: int
    saved_bytes_per_sample: int

    def maximum_batch_size(
        self,
        memory_gib: float,
        *,
        reserve_fraction: float = 0.10,
        external_bytes: int = 0,
    ) -> int:
        """Return an analytical ceiling; the live device probe may lower it."""

        if memory_gib <= 0:
            return 8
        if not 0.0 <= reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must be in [0, 1).")
        usable = int(memory_gib * (1024**3) * (1.0 - reserve_fraction))
        remaining = usable - self.persistent_training_bytes - max(0, int(external_bytes))
        return max(1, remaining // max(1, self.saved_bytes_per_sample))


class _DistillationStudent(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projection = nn.Linear(int(encoder.num_features), 512)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(pooled_features(self.encoder, images)).square().mean()


def _saved_tensor_bytes(model: nn.Module, batch_size: int) -> int:
    saved = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved
        saved += tensor.numel() * tensor.element_size()
        return tensor

    images = torch.empty(batch_size, 1, 192, 128, device="meta")
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model(images)
        loss = output.loss if hasattr(output, "loss") else output
        loss.backward()
    return saved


@lru_cache(maxsize=16)
def estimate_training_memory(
    model_size: str,
    objective: str,
    *,
    canon_config: CanonConfig | None = None,
    decoder_config: DecoderConfig | None = None,
) -> TrainingMemoryEstimate:
    """Inspect the real meta-device graph for model-dependent memory weight.

    ``persistent_training_bytes`` is exact for FP32 parameters, gradients, and
    AdamW's two FP32 moments.  Saved-tensor bytes are measured by subtracting a
    batch-one graph from a batch-two graph, which removes model-size constants.
    The FP32 graph is intentionally conservative before the live AMP probe.
    """

    size = resolve_vit_spec(model_size).size
    if objective not in {"distill", "reconstruct"}:
        raise ValueError(f"Unknown objective: {objective}.")
    selected_canon = canon_config or CanonConfig()
    with torch.device("meta"):
        encoder = build_audio_vit(size, canon=selected_canon)
        if objective == "reconstruct":
            model: nn.Module = MaskedSpectrogramModel(
                encoder,
                model_size=size,
                decoder=decoder_config,
            )
        else:
            model = _DistillationStudent(encoder)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    saved_one = _saved_tensor_bytes(model, 1)
    saved_two = _saved_tensor_bytes(model, 2)
    saved_per_sample = max(1, saved_two - saved_one)
    # FP32 parameter + gradient + two AdamW moment tensors.
    persistent = trainable * 4 * 4
    return TrainingMemoryEstimate(
        model_size=size,
        objective=objective,
        trainable_parameters=trainable,
        persistent_training_bytes=persistent,
        saved_bytes_per_sample=saved_per_sample,
    )


def rounded_initial_batch(maximum_batch_size: int, *, round_to: int = 8) -> int:
    """Choose the largest power-of-two warmup batch under the measured cap."""

    if maximum_batch_size <= 0 or round_to <= 0:
        raise ValueError("maximum_batch_size and round_to must be positive.")
    return 1 << int(math.floor(math.log2(maximum_batch_size)))


def round_batch_cap(maximum_batch_size: int, *, round_to: int = 8) -> int:
    """Round an analytical ceiling down without ever returning zero."""

    if maximum_batch_size <= 0 or round_to <= 0:
        raise ValueError("maximum_batch_size and round_to must be positive.")
    if maximum_batch_size < round_to:
        return int(maximum_batch_size)
    return max(round_to, int(math.floor(maximum_batch_size / round_to)) * round_to)


__all__ = [
    "TrainingMemoryEstimate",
    "estimate_training_memory",
    "round_batch_cap",
    "rounded_initial_batch",
]
