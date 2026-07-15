"""Run a direct-pretraining encoder on MTEB's MAEB audio-only benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from hear_distill.audio import AudioPreprocessor
from hear_distill.models import CanonConfig, build_audio_vit, pooled_features

try:
    from mteb.models.abs_encoder import AbsEncoder
except ImportError:  # Keep core training installs independent of the benchmark extra.
    class AbsEncoder:  # type: ignore[no-redef]
        pass


BENCHMARK_NAME = "MAEB(audio-only)"
SAMPLE_RATE = 16_000
CLIP_SAMPLES = 32_000


def prepare_audio(audio: np.ndarray, clip_samples: int = CLIP_SAMPLES) -> torch.Tensor:
    """Convert a decoded clip to mono float32 and take its deterministic center crop."""
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=0)
    array = array.reshape(-1)
    if array.shape[0] > clip_samples:
        start = (array.shape[0] - clip_samples) // 2
        array = array[start : start + clip_samples]
    elif array.shape[0] < clip_samples:
        array = np.pad(array, (0, clip_samples - array.shape[0]))
    return torch.from_numpy(array.copy())


def canon_config_from_checkpoint(args: dict[str, Any]) -> CanonConfig:
    placements = tuple(bool(args.get(name, False)) for name in ("canon_a", "canon_b", "canon_c", "canon_d"))
    enable_all = bool(args.get("canon", False) and (args.get("canon_abcd", False) or not any(placements)))
    return CanonConfig(
        enabled=bool(args.get("canon", False)),
        use_2d=bool(args.get("canon_2d", False)),
        kernel_size=int(args.get("canon_kernel", 4)),
        a=bool(args.get("canon_a", False) or enable_all),
        b=bool(args.get("canon_b", False) or enable_all),
        b_qkv=bool(args.get("canon_b_qkv", False)),
        c=bool(args.get("canon_c", False) or enable_all),
        d=bool(args.get("canon_d", False) or enable_all),
        causal=bool(args.get("canon_causal", False)),
        disable_positional_encoding=bool(args.get("canon_no_pos_enc", False)),
    )


def load_encoder_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load only the encoder state; decoder and optimizer tensors are never moved to the GPU."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("encoder")
    if not isinstance(state, dict):
        raise ValueError(f"{path} is not a direct-pretraining checkpoint with an `encoder` state.")
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"{path} has invalid checkpoint arguments.")
    encoder = build_audio_vit(
        str(args.get("model_size", "small")),
        canon=canon_config_from_checkpoint(args),
    )
    encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False).eval().to(device)
    return encoder, checkpoint


class CanonAudioMTEBEncoder(AbsEncoder):
    """MTEB encoder protocol adapter for the trained Canon audio ViT."""

    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        amp_dtype: str = "bfloat16",
    ) -> None:
        try:
            from mteb.models import ModelMeta
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install the benchmark dependencies with `pip install -e '.[benchmark]'`.") from exc

        self.device = torch.device(device)
        self.model, raw_checkpoint = load_encoder_checkpoint(checkpoint, self.device)
        self.preprocessor = AudioPreprocessor().eval().to(self.device)
        self.amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(amp_dtype)
        model_args = raw_checkpoint.get("args", {})
        dim = int(getattr(self.model, "num_features", getattr(self.model, "embed_dim", 0)))
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.mteb_model_meta = ModelMeta.create_empty(
            {
                "name": f"local/canon-audio-vit-{model_args.get('model_size', 'unknown')}",
                "revision": f"step-{raw_checkpoint.get('step', 'unknown')}",
                "n_parameters": parameters,
                "embed_dim": dim,
                "similarity_fn_name": "cosine",
                "framework": ["PyTorch", "timm"],
                "modalities": ["audio"],
                "model_type": ["dense"],
            }
        )

    def _embed_batch(self, items: Iterable[dict[str, Any]]) -> torch.Tensor:
        waveforms = torch.stack([prepare_audio(item["array"]) for item in items]).to(self.device)
        amp_enabled = self.device.type == "cuda" and self.amp_dtype is not None
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=amp_enabled,
        ):
            spectrograms = self.preprocessor(waveforms)
            embeddings = pooled_features(self.model, spectrograms)
        return embeddings.float().cpu()

    def encode(
        self,
        inputs: DataLoader,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        show_progress_bar: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        del task_metadata, hf_split, hf_subset, prompt_type, kwargs
        from mteb.models.modality_collators import AudioCollator

        inputs.collate_fn = AudioCollator(target_sampling_rate=SAMPLE_RATE)
        chunks = [
            self._embed_batch(batch["audio"])
            for batch in tqdm(inputs, disable=not show_progress_bar, desc="Embedding audio")
        ]
        if not chunks:
            return np.empty((0, int(self.mteb_model_meta.embed_dim)), dtype=np.float32)
        return torch.cat(chunks).numpy()


def _find_audio_examples(task: Any, limit: int) -> list[dict[str, Any]]:
    """Find representative decoded audio values in any MTEB task storage layout."""
    examples: list[dict[str, Any]] = []
    roots = [getattr(task, name, None) for name in ("dataset", "queries", "corpus")]
    declared_columns = {
        getattr(task, name, None)
        for name in ("input_column_name", "input1_column_name", "input2_column_name")
    }
    declared_columns.discard(None)

    def visit(value: Any) -> None:
        if len(examples) >= limit or value is None:
            return
        if hasattr(value, "get_all_samples"):
            examples.append(value)
            return
        if isinstance(value, Mapping):
            if "array" in value and "sampling_rate" in value:
                examples.append(value)
                return
            for child in value.values():
                visit(child)
                if len(examples) >= limit:
                    return
            return
        if hasattr(value, "column_names") and hasattr(value, "__len__"):
            audio_columns = [
                name
                for name in value.column_names
                if name in declared_columns or name.startswith("audio")
            ]
            for index in range(min(len(value), limit)):
                row = value[index]
                for name in audio_columns:
                    visit(row[name])
                    if len(examples) >= limit:
                        return

    for root in roots:
        visit(root)
    return examples


def smoke_all_tasks(model: CanonAudioMTEBEncoder, tasks: Iterable[Any], samples: int) -> list[dict[str, Any]]:
    """Load and encode real examples from every task before running full metrics."""
    report = []
    for task in tasks:
        name = task.metadata.name
        print(f"[smoke] {name}: loading", flush=True)
        task.load_data()
        examples = _find_audio_examples(task, samples)
        if not examples:
            raise RuntimeError(f"{name} loaded but exposed no decoded audio examples.")
        embeddings = model._embed_batch(examples)
        if embeddings.shape[0] != len(examples) or not torch.isfinite(embeddings).all():
            raise RuntimeError(f"{name} produced invalid embeddings with shape {tuple(embeddings.shape)}.")
        row = {"task": name, "samples": len(examples), "embedding_dim": embeddings.shape[1]}
        report.append(row)
        print(f"[smoke] {name}: ok ({len(examples)} clips, dim={embeddings.shape[1]})", flush=True)
        task.unload_data()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/maeb-audio"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--smoke-samples", type=int, default=2)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0 or args.smoke_samples <= 0 or args.num_workers < 0:
        raise SystemExit("Batch/smoke sizes must be positive and workers must be non-negative.")
    try:
        import mteb
    except ImportError as exc:
        raise SystemExit("Install benchmark dependencies with `pip install -e '.[benchmark]'`.") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    model = CanonAudioMTEBEncoder(args.checkpoint, device=args.device, amp_dtype=args.amp_dtype)
    benchmark = mteb.get_benchmark(BENCHMARK_NAME)
    tasks = list(benchmark.tasks)
    print(f"Loaded {len(tasks)} tasks from {benchmark.name}.", flush=True)

    if not args.skip_smoke:
        report = smoke_all_tasks(model, tasks, args.smoke_samples)
        (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.smoke_only:
        return

    result = mteb.evaluate(
        model,
        tasks,
        encode_kwargs={"batch_size": args.batch_size, "show_progress_bar": True},
        cache=mteb.ResultCache(cache_path=args.output / "cache"),
        overwrite_strategy="only-missing",
        prediction_folder=args.output / "predictions",
        show_progress_bar=True,
        num_proc=args.num_workers,
    )
    (args.output / "model-result.json").write_text(result.model_dump_json(indent=2) + "\n")


if __name__ == "__main__":
    main()
