"""Hardware-aware defaults for the streaming and training pipeline."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .models.memory import estimate_training_memory, round_batch_cap


GIB = 1024**3


@dataclass(frozen=True)
class HostResources:
    cpu_count: int
    disk_total_gib: float
    disk_free_gib: float
    gpu_name: Optional[str]
    gpu_memory_gib: float


@dataclass(frozen=True)
class RuntimePlan:
    disk_fraction: float
    disk_budget_gib: float
    disk_min_free_gib: float
    lake_max_gib: float
    reserve_low_gib: float
    reserve_high_gib: float
    chunk_gib_per_stream: float
    val_max_gib: float
    decay_max_gib: float
    num_streams: int
    min_streams: int
    train_num_workers: int
    train_batch_size: int
    shuffle_buffer: int
    shard_size: int
    device: str
    amp: bool
    teacher_batch_factor: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _gpu_info() -> tuple[Optional[str], float]:
    query = "name,memory.total"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, 0.0
    if proc.returncode != 0 or not proc.stdout.strip():
        return None, 0.0
    first = proc.stdout.strip().splitlines()[0]
    try:
        name, memory_mib = [part.strip() for part in first.rsplit(",", 1)]
        return name, float(memory_mib) / 1024.0
    except (TypeError, ValueError):
        return first.strip() or None, 0.0


def inspect_host(path: Path) -> HostResources:
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    gpu_name, gpu_memory_gib = _gpu_info()
    return HostResources(
        cpu_count=max(1, int(os.cpu_count() or 1)),
        disk_total_gib=usage.total / GIB,
        disk_free_gib=usage.free / GIB,
        gpu_name=gpu_name,
        gpu_memory_gib=gpu_memory_gib,
    )


def training_batch_size(
    memory_gib: float,
    *,
    model_size: str = "small",
    objective: str = "distill",
) -> int:
    """Derive a hardware ceiling from the selected model's autograd graph."""

    estimate = estimate_training_memory(model_size, objective)
    return round_batch_cap(estimate.maximum_batch_size(memory_gib), round_to=8)


def build_runtime_plan(
    resources: HostResources,
    *,
    disk_fraction: float = 0.5,
    model_size: str = "small",
    objective: str = "distill",
) -> RuntimePlan:
    if not 0.05 <= disk_fraction <= 0.9:
        raise ValueError("disk_fraction must be between 0.05 and 0.9.")

    disk_min_free = min(50.0, max(5.0, resources.disk_total_gib * 0.05))
    capacity_budget = resources.disk_total_gib * disk_fraction
    free_budget = max(1.0, resources.disk_free_gib - disk_min_free)
    disk_budget = min(capacity_budget, free_budget)

    cpu_count = resources.cpu_count
    num_streams = max(1, min(8, cpu_count // 4))
    min_streams = 1 if num_streams <= 2 else 2
    train_workers = max(1, min(12, cpu_count - num_streams - 1, cpu_count // 2))

    lake_max = max(1.0, disk_budget * 0.70)
    reserve_high = max(0.5, min(lake_max * 0.20, disk_budget * 0.10))
    reserve_low = max(0.25, reserve_high * 0.5)
    chunk = min(2.0, max(0.25, (disk_budget * 0.04) / num_streams))
    val_max = max(0.25, disk_budget * 0.01)
    decay_max = max(0.5, disk_budget * 0.08)
    # Hugging Face streaming shuffle buffers hold full examples, including MP3
    # bytes. Bound the aggregate across workers instead of allocating 20k per
    # process (which can consume many GiB on long source clips).
    shuffle_buffer = max(500, min(4_000, 8_000 // num_streams))

    device = "cuda" if resources.gpu_name else "cpu"
    amp = device == "cuda"
    teacher_batch_factor = (
        2 if objective == "distill" and resources.gpu_memory_gib >= 39 else 1
    )

    return RuntimePlan(
        disk_fraction=float(disk_fraction),
        disk_budget_gib=round(disk_budget, 3),
        disk_min_free_gib=round(disk_min_free, 3),
        lake_max_gib=round(lake_max, 3),
        reserve_low_gib=round(reserve_low, 3),
        reserve_high_gib=round(reserve_high, 3),
        chunk_gib_per_stream=round(chunk, 3),
        val_max_gib=round(val_max, 3),
        decay_max_gib=round(decay_max, 3),
        num_streams=num_streams,
        min_streams=min_streams,
        train_num_workers=train_workers,
        train_batch_size=training_batch_size(
            resources.gpu_memory_gib,
            model_size=model_size,
            objective=objective,
        ),
        shuffle_buffer=shuffle_buffer,
        shard_size=2_000,
        device=device,
        amp=amp,
        teacher_batch_factor=teacher_batch_factor,
    )


__all__ = [
    "HostResources",
    "RuntimePlan",
    "build_runtime_plan",
    "inspect_host",
    "training_batch_size",
]
