"""Direct masked-spectrogram pretraining for Canon audio encoders."""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import re
import time
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from adaptive_warmup import (
    AdaptiveWarmup,
    WarmupConfig,
    estimate_critical_learning_rate,
    estimate_gradient_noise,
    set_reference_lr,
)

from hear_distill.audio import AudioPreprocessor
from hear_distill.data import AudioShardDataset, discover_shards
from hear_distill.models import CanonConfig, MaskedSpectrogramModel, build_audio_vit
from hear_distill.models.memory import (
    estimate_training_memory,
    round_batch_cap,
    rounded_initial_batch,
)
from hear_distill.models.reconstruction import decoder_config_for_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly pretrain a Canon-ViT audio encoder by masked reconstruction."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/laion_audio_lake/train"))
    parser.add_argument("--shards-glob", default="shard-*.tar")
    parser.add_argument("--streams-glob", default="stream-*")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/canon_audio_pretrain"))
    parser.add_argument("--model-size", choices=("tiny", "small", "base", "large"), default="small")
    parser.add_argument("--encoder-pretrained", action="store_true")
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--decoder-dim", type=int, default=0)
    parser.add_argument("--decoder-depth", type=int, default=0)
    parser.add_argument("--decoder-heads", type=int, default=0)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument(
        "--normalize-patch-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--canon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--canon-2d", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--canon-kernel", type=int, default=4)
    parser.add_argument("--canon-a", action="store_true")
    parser.add_argument("--canon-b", action="store_true")
    parser.add_argument("--canon-b-qkv", action="store_true")
    parser.add_argument("--canon-c", action="store_true")
    parser.add_argument("--canon-d", action="store_true")
    parser.add_argument("--canon-abcd", action="store_true")
    parser.add_argument("--canon-causal", action="store_true")
    parser.add_argument(
        "--canon-no-pos-enc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Manual batch size, or an optional adaptive ceiling (0 lets the probe decide).",
    )
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument(
        "--auto-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt LR and batch size within the measured device-memory ceiling.",
    )
    parser.add_argument("--auto-warmup-steps", type=int, default=1000)
    parser.add_argument("--auto-warmup-metric-every", type=int, default=5)
    parser.add_argument("--auto-warmup-init-lr", type=float, default=0.0)
    parser.add_argument("--auto-warmup-max-lr", type=float, default=0.0)
    parser.add_argument("--auto-warmup-max-batch-size", type=int, default=0)
    parser.add_argument(
        "--auto-warmup-batch-multiplier",
        type=float,
        default=2.0,
        help="Post-warmup WSD batch multiple applied to the selected critical batch.",
    )
    parser.add_argument("--auto-warmup-batch-round-to", type=int, default=8)
    parser.add_argument("--auto-warmup-memory-reserve", type=float, default=0.10)
    parser.add_argument("--auto-warmup-oom-buffer-frac", type=float, default=0.10)
    parser.add_argument("--gns-param-sample", type=int, default=200_000)
    parser.add_argument(
        "--lr-schedule-start-step",
        type=int,
        default=0,
        help="Step where phase-local decay begins; managed by the lake decay phase.",
    )
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--lr-schedule", choices=("none", "cosine", "linear"), default="cosine")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--compile-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-checkpoints", type=int, default=5)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--repeat", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-shard-refresh", action="store_true")
    parser.add_argument("--shard-refresh-sec", type=float, default=30.0)

    # Accepted for compatibility with the shared data-lake orchestrator. Direct
    # reconstruction currently keeps validation external to the hot train loop.
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-target-clips", type=int, default=0)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--val-live-refresh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-shard-refresh-sec", type=float, default=30.0)
    parser.add_argument("--gns-every", type=int, default=0)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="canon-audio-pretrain")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-tags")
    return parser


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")
    return device


def _safe_load(path: Path) -> dict:
    import pathlib

    try:
        from torch.serialization import safe_globals
    except ImportError:
        safe_globals = None
    if safe_globals is not None:
        with safe_globals([pathlib.PosixPath]):
            try:
                return torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                return torch.load(path, map_location="cpu")
    return torch.load(path, map_location="cpu")


def _latest_checkpoint(out_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    pattern = re.compile(r"ckpt_(\d+)\.pt$")
    for path in out_dir.glob("ckpt_*.pt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return max(candidates)[1]
    final = out_dir / "ckpt_final.pt"
    return final if final.exists() else None


def _prune_checkpoints(out_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    pattern = re.compile(r"ckpt_(\d+)\.pt$")
    candidates = []
    for path in out_dir.glob("ckpt_*.pt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for _step, path in sorted(candidates)[:-keep]:
        path.unlink(missing_ok=True)


def _canon_config(args: argparse.Namespace) -> CanonConfig:
    placements = (args.canon_a, args.canon_b, args.canon_c, args.canon_d)
    enable_all = bool(args.canon and (args.canon_abcd or not any(placements)))
    return CanonConfig(
        enabled=bool(args.canon),
        use_2d=bool(args.canon_2d),
        kernel_size=int(args.canon_kernel),
        a=bool(args.canon_a or enable_all),
        b=bool(args.canon_b or enable_all),
        b_qkv=bool(args.canon_b_qkv),
        c=bool(args.canon_c or enable_all),
        d=bool(args.canon_d or enable_all),
        causal=bool(args.canon_causal),
        disable_positional_encoding=bool(args.canon_no_pos_enc),
    )


def _lr_at_step(args: argparse.Namespace, step: int) -> float:
    if (
        args.lr_schedule_start_step == 0
        and args.lr_warmup_steps > 0
        and step <= args.lr_warmup_steps
    ):
        return args.lr * step / args.lr_warmup_steps
    if args.lr_schedule == "none":
        return args.lr
    schedule_start = (
        args.lr_schedule_start_step
        if args.lr_schedule_start_step > 0
        else args.lr_warmup_steps
    )
    decay_steps = max(1, args.max_steps - schedule_start)
    progress = min(1.0, max(0.0, (step - schedule_start) / decay_steps))
    if args.lr_schedule == "linear":
        factor = 1.0 - (1.0 - args.lr_min_ratio) * progress
    else:
        factor = args.lr_min_ratio + (1.0 - args.lr_min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    return args.lr * factor


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "max_steps": args.max_steps,
        "grad_accum": args.grad_accum,
        "canon_kernel": args.canon_kernel,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.batch_size < 0:
        raise SystemExit("--batch-size cannot be negative.")
    if not args.auto_warmup and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive when adaptive warmup is disabled.")
    if args.num_workers < 0 or args.lr_warmup_steps < 0 or args.lr_schedule_start_step < 0:
        raise SystemExit("Worker and warmup counts cannot be negative.")
    if args.auto_warmup and args.lr_warmup_steps > 0:
        raise SystemExit("--auto-warmup cannot be combined with --lr-warmup-steps.")
    if (
        not args.auto_warmup
        and args.lr_schedule_start_step == 0
        and args.lr_warmup_steps >= args.max_steps
    ):
        raise SystemExit("--lr-warmup-steps must be smaller than --max-steps.")
    if args.auto_warmup_steps <= 0 or args.auto_warmup_metric_every <= 0:
        raise SystemExit("Adaptive warmup step counts must be positive.")
    if (
        args.auto_warmup_batch_round_to <= 0
        or args.auto_warmup_batch_multiplier <= 0
        or args.gns_param_sample <= 0
    ):
        raise SystemExit("Adaptive batch rounding and GNS sample size must be positive.")
    if args.auto_warmup_max_lr < 0 or args.auto_warmup_max_batch_size < 0:
        raise SystemExit("Adaptive warmup caps cannot be negative.")
    if not 0.0 <= args.auto_warmup_memory_reserve < 1.0:
        raise SystemExit("--auto-warmup-memory-reserve must be in [0, 1).")
    if not 0.0 <= args.auto_warmup_oom_buffer_frac < 1.0:
        raise SystemExit("--auto-warmup-oom-buffer-frac must be in [0, 1).")
    if args.resume_from is not None and args.resume_latest:
        raise SystemExit("Use only one of --resume-from and --resume-latest.")


def _make_loader(
    dataset: AudioShardDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def _live_cuda_batch_cap(
    model: MaskedSpectrogramModel,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    reserve_fraction: float,
    analytical_cap: int,
) -> tuple[int, dict[str, int]]:
    """Calibrate the architecture estimate using real eager CUDA allocations."""

    device = next(model.parameters()).device
    if device.type != "cuda":
        return analytical_cap, {}
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    current_allocated = torch.cuda.memory_allocated(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    external_bytes = max(0, int(total_bytes - free_bytes - current_allocated))

    def peak_for(batch_size: int) -> int:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        images = torch.zeros(batch_size, 1, 192, 128, device=device)
        autocast = (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if amp_enabled
            else contextlib.nullcontext()
        )
        with autocast:
            loss = model(images).loss
        loss.backward()
        torch.cuda.synchronize(device)
        return int(torch.cuda.max_memory_allocated(device))

    try:
        peak_one = peak_for(1)
        peak_two = peak_for(2)
    except torch.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return 1, {"total_bytes": int(total_bytes), "external_bytes": external_bytes}
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    per_sample = max(1, peak_two - peak_one)
    fixed_without_moments = max(current_allocated, peak_one - per_sample)
    adam_moments = trainable * 8
    fixed = fixed_without_moments + adam_moments
    target = int(total_bytes * (1.0 - reserve_fraction)) - external_bytes
    measured_cap = max(1, (target - fixed) // per_sample)
    cap = min(int(analytical_cap), int(measured_cap))
    return cap, {
        "total_bytes": int(total_bytes),
        "external_bytes": external_bytes,
        "fixed_bytes": int(fixed),
        "bytes_per_sample": int(per_sample),
        "measured_cap": int(measured_cap),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    shards = discover_shards(args.data_dir, args.shards_glob, args.streams_glob)
    if not shards:
        raise SystemExit(f"No audio shards found under {args.data_dir}.")
    dataset = AudioShardDataset(
        shards,
        clip_samples=32_000,
        sample_rate=16_000,
        shuffle_shards=args.shuffle_shards,
        seed=args.seed,
        repeat=args.repeat,
        live_data_dir=args.data_dir if args.live_shard_refresh else None,
        shards_glob=args.shards_glob,
        streams_glob=args.streams_glob,
        refresh_interval_sec=args.shard_refresh_sec,
    )
    encoder = build_audio_vit(
        args.model_size,
        pretrained=args.encoder_pretrained,
        canon=_canon_config(args),
    )
    decoder_config = decoder_config_for_model(
        args.model_size,
        dim=args.decoder_dim or None,
        depth=args.decoder_depth or None,
        heads=args.decoder_heads or None,
    )
    model = MaskedSpectrogramModel(
        encoder,
        model_size=args.model_size,
        decoder=decoder_config,
        mask_ratio=args.mask_ratio,
        normalize_patch_targets=args.normalize_patch_targets,
    ).to(device)

    if args.encoder_checkpoint is not None:
        checkpoint = _safe_load(args.encoder_checkpoint)
        state = checkpoint.get("encoder") or checkpoint.get("student")
        if not isinstance(state, dict):
            raise SystemExit("Encoder checkpoint has neither `encoder` nor `student` weights.")
        model.encoder.load_state_dict(state, strict=True)
    if args.decoder_checkpoint is not None:
        model.load_decoder_checkpoint(_safe_load(args.decoder_checkpoint))

    fused = bool(args.fused_adamw and device.type == "cuda")
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=fused
        )
    except (TypeError, RuntimeError):
        fused = False
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    amp_enabled = bool(device.type == "cuda" if args.amp is None else args.amp)
    use_bfloat16 = bool(
        amp_enabled
        and device.type == "cuda"
        and (
            args.amp_dtype == "bfloat16"
            or (args.amp_dtype == "auto" and torch.cuda.is_bf16_supported())
        )
    )
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and device.type == "cuda" and not use_bfloat16
    )
    preprocessor = AudioPreprocessor().eval().to(device)

    memory_estimate = estimate_training_memory(
        args.model_size,
        "reconstruct",
        canon_config=_canon_config(args),
        decoder_config=decoder_config,
    )
    device_memory_gib = (
        torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    analytical_cap = round_batch_cap(
        memory_estimate.maximum_batch_size(
            device_memory_gib,
            reserve_fraction=args.auto_warmup_memory_reserve,
        )
        if device.type == "cuda"
        else max(1, args.batch_size or args.auto_warmup_batch_round_to),
        round_to=args.auto_warmup_batch_round_to,
    )
    live_cap, live_memory = _live_cuda_batch_cap(
        model,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        reserve_fraction=args.auto_warmup_memory_reserve,
        analytical_cap=analytical_cap,
    )
    batch_cap = live_cap
    if args.batch_size > 0:
        batch_cap = min(batch_cap, args.batch_size)
    if args.auto_warmup_max_batch_size > 0:
        batch_cap = min(batch_cap, args.auto_warmup_max_batch_size)
    batch_cap = max(1, int(batch_cap))

    warmup: AdaptiveWarmup | None = None
    if args.auto_warmup:
        current_batch_size = rounded_initial_batch(
            batch_cap,
            round_to=args.auto_warmup_batch_round_to,
        )
        initial_lr = args.auto_warmup_init_lr or max(args.lr * 0.01, 1e-8)
        warmup = AdaptiveWarmup(
            initial_lr=initial_lr,
            initial_batch_size=current_batch_size,
            config=WarmupConfig(
                warmup_steps=args.auto_warmup_steps,
                measurement_interval=args.auto_warmup_metric_every,
                batch_multiplier=args.auto_warmup_batch_multiplier,
                batch_round_to=args.auto_warmup_batch_round_to,
                max_lr=args.auto_warmup_max_lr or args.lr,
                max_batch_size=batch_cap,
                oom_buffer_fraction=args.auto_warmup_oom_buffer_frac,
            ),
        )
        set_reference_lr(optimizer, initial_lr)
    else:
        current_batch_size = args.batch_size

    loader = _make_loader(
        dataset,
        batch_size=current_batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    step = 0
    resume_path = args.resume_from or (_latest_checkpoint(args.out) if args.resume_latest else None)
    if resume_path is not None:
        checkpoint = _safe_load(resume_path)
        if checkpoint.get("model_config") != model.checkpoint_metadata():
            raise SystemExit("Resume checkpoint architecture does not match this run.")
        model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        model.load_decoder_checkpoint(checkpoint)
        optimizer.load_state_dict(checkpoint["optim"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        if warmup is not None:
            warmup_state = checkpoint.get("adaptive_warmup_state")
            if not isinstance(warmup_state, dict):
                raise SystemExit(
                    "Resume checkpoint has no adaptive-warmup state; use "
                    "--no-auto-warmup or restart from step 0."
                )
            warmup.load_state_dict(warmup_state)
            current_batch_size = warmup.current_batch_size
            set_reference_lr(optimizer, warmup.current_lr)
            loader = _make_loader(
                dataset,
                batch_size=current_batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
        step = int(checkpoint.get("step", 0))
        print(f"Resumed training from {resume_path} at step={step}.", flush=True)

    runner = model
    compile_active = bool(args.compile_model and device.type == "cuda" and hasattr(torch, "compile"))
    if compile_active:
        try:
            runner = torch.compile(
                model,
                mode=args.compile_mode,
                dynamic=bool(warmup is not None),
            )
        except Exception as exc:
            compile_active = False
            print(f"Warning: torch.compile unavailable ({exc}); using eager mode.", flush=True)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            config=vars(args),
        )
        wandb_run.config.update(
            {
                "memory/analytical_batch_cap": analytical_cap,
                "memory/effective_batch_cap": batch_cap,
                "memory/saved_bytes_per_sample": memory_estimate.saved_bytes_per_sample,
                "memory/trainable_parameters": memory_estimate.trainable_parameters,
                **{f"memory/live_{key}": value for key, value in live_memory.items()},
            },
            allow_val_change=True,
        )

    def save_checkpoint(tag: str) -> None:
        payload = {
            "objective": "masked-reconstruction",
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder_state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "step": step,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "model_config": model.checkpoint_metadata(),
            "adaptive_warmup_state": warmup.state_dict() if warmup is not None else None,
            "memory_plan": {
                "analytical_cap": analytical_cap,
                "batch_cap": batch_cap,
                "live": live_memory,
            },
        }
        torch.save(payload, args.out / f"ckpt_{tag}.pt")
        if tag != "final":
            _prune_checkpoints(args.out, args.max_checkpoints)

    print(
        "Direct pretraining: "
        f"model={args.model_size} patches={model.patch_count} decoder={decoder_config} "
        f"canon={_canon_config(args)}",
        flush=True,
    )
    print(
        f"Runtime: device={device} amp={amp_enabled} fused_adamw={fused} "
        f"compile={compile_active} shards={len(shards)}",
        flush=True,
    )
    print(
        "Adaptive memory plan: "
        f"params={memory_estimate.trainable_parameters:,} "
        f"saved_bytes/sample={memory_estimate.saved_bytes_per_sample:,} "
        f"analytical_cap={analytical_cap} live_cap={live_cap} "
        f"effective_cap={batch_cap} initial_batch={current_batch_size} "
        f"adaptive={int(warmup is not None)}",
        flush=True,
    )

    data_iterator = iter(loader)

    def next_audio_batch() -> torch.Tensor:
        nonlocal data_iterator
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)
        return batch.to(device, non_blocking=True)

    def preprocess_audio(audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return preprocessor(audio)

    def autocast_context():
        return (
            torch.amp.autocast("cuda", dtype=amp_dtype)
            if amp_enabled and device.type == "cuda"
            else contextlib.nullcontext()
        )

    def probe_example() -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = preprocess_audio(next_audio_batch())
        return spectrogram, model.random_mask(spectrogram.shape[0], spectrogram.device)

    def probe_loss(batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        spectrogram, mask = batch
        with autocast_context():
            return model(spectrogram, mask).loss

    def adaptive_decay_lr(train_step: int) -> float:
        assert warmup is not None
        if train_step <= args.auto_warmup_steps:
            return warmup.current_lr
        progress = min(
            1.0,
            max(
                0.0,
                (train_step - args.auto_warmup_steps)
                / max(1, args.max_steps - args.auto_warmup_steps),
            ),
        )
        if args.lr_schedule == "none":
            factor = 1.0
        elif args.lr_schedule == "linear":
            factor = 1.0 - (1.0 - args.lr_min_ratio) * progress
        else:
            factor = args.lr_min_ratio + (1.0 - args.lr_min_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return warmup.current_lr * factor

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    latest_adaptive_metrics: dict[str, float] = {}
    last_log = time.perf_counter()
    while step < args.max_steps:
        step += 1
        step_loss = 0.0
        lr = adaptive_decay_lr(step) if warmup is not None else _lr_at_step(args, step)
        set_reference_lr(optimizer, lr)
        try:
            for _micro_step in range(args.grad_accum):
                spectrogram = preprocess_audio(next_audio_batch())
                with autocast_context():
                    output = runner(spectrogram)
                    loss = output.loss / args.grad_accum
                scaler.scale(loss).backward()
                step_loss += float(output.loss.detach())
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if warmup is None or current_batch_size <= 1:
                raise
            recommendation = warmup.report_oom(
                step - 1,
                failed_batch_size=current_batch_size,
            )
            current_batch_size = recommendation.batch_size
            loader = _make_loader(
                dataset,
                batch_size=current_batch_size,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            data_iterator = iter(loader)
            step -= 1
            print(
                f"OOM: reduced adaptive batch cap to {recommendation.batch_size_cap} "
                f"and retrying with batch={current_batch_size}.",
                flush=True,
            )
            continue

        if warmup is not None and warmup.should_measure(step - 1):
            scaler.unscale_(optimizer)
            try:
                probe_a = probe_example()
                probe_b = probe_example()
                gradient_noise = estimate_gradient_noise(
                    model.parameters(),
                    probe_loss,
                    probe_a,
                    probe_b,
                    batch_size=current_batch_size,
                    max_elements=args.gns_param_sample,
                )
                critical_lr = estimate_critical_learning_rate(
                    lambda: probe_loss(probe_a),
                    optimizer=optimizer,
                    model=model,
                    current_lr=lr,
                    previous_estimate=warmup.last_critical_lr,
                    max_lr=args.auto_warmup_max_lr or args.lr,
                )
                recommendation = warmup.observe(
                    step - 1,
                    critical_lr=critical_lr,
                    critical_batch_size=gradient_noise,
                )
                lr = set_reference_lr(optimizer, recommendation.learning_rate)
                latest_adaptive_metrics = {
                    "adaptive/critical_lr": critical_lr.critical_lr,
                    "adaptive/critical_sharpness": critical_lr.critical_sharpness,
                    "adaptive/critical_batch_size": gradient_noise.critical_batch_size,
                    "adaptive/noise_to_signal_ratio": gradient_noise.noise_to_signal_ratio,
                    "adaptive/batch_goal": float(recommendation.batch_size_goal or 0),
                }
                print(
                    f"adaptive@step={step} lr={lr:.3e} "
                    f"critical_lr={critical_lr.critical_lr:.3e} "
                    f"sharpness={critical_lr.critical_sharpness:.3e} "
                    f"critical_batch={gradient_noise.critical_batch_size:.1f} "
                    f"batch={current_batch_size} goal={recommendation.batch_size_goal} "
                    f"cap={recommendation.batch_size_cap}",
                    flush=True,
                )
            except torch.OutOfMemoryError:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                print(
                    "Warning: adaptive metric probe exceeded memory; "
                    "skipping this measurement while preserving the real step.",
                    flush=True,
                )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        running_loss += step_loss / args.grad_accum

        if warmup is not None and step == args.auto_warmup_steps:
            handoff = warmup.complete_warmup(step)
            if handoff.batch_size != current_batch_size:
                current_batch_size = handoff.batch_size
                loader = _make_loader(
                    dataset,
                    batch_size=current_batch_size,
                    num_workers=args.num_workers,
                    pin_memory=device.type == "cuda",
                )
                data_iterator = iter(loader)
            print(
                f"adaptive handoff step={step} batch={current_batch_size} "
                f"selected_critical_batch={handoff.selected_critical_batch_size} "
                f"multiplier={args.auto_warmup_batch_multiplier:.3f}",
                flush=True,
            )

        if step % args.log_every == 0 or step == 1:
            elapsed = max(1e-6, time.perf_counter() - last_log)
            interval = 1 if step == 1 else args.log_every
            average = running_loss / interval
            steps_per_second = interval / elapsed
            samples_per_second = steps_per_second * current_batch_size * args.grad_accum
            print(
                f"step={step} loss={average:.6f} lr={lr:.3e} "
                f"batch={current_batch_size} cap={batch_cap} steps/s={steps_per_second:.2f} "
                f"samples/s={samples_per_second:.1f}",
                flush=True,
            )
            if wandb_run is not None:
                metrics = {
                    "train/step": step,
                    "train/reconstruction_loss": average,
                    "train/lr": lr,
                    "train/batch_size": current_batch_size,
                    "train/batch_cap": batch_cap,
                    "performance/steps_per_second": steps_per_second,
                    "performance/samples_per_second": samples_per_second,
                    **latest_adaptive_metrics,
                }
                if device.type == "cuda":
                    metrics.update(
                        {
                            "cuda/allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                            "cuda/reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                            "cuda/peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                            / (1024**3),
                        }
                    )
                wandb_run.log(metrics, step=step)
            running_loss = 0.0
            last_log = time.perf_counter()
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(str(step))

    save_checkpoint("final")
    if wandb_run is not None:
        wandb_run.finish()
    return 0


__all__ = ["main"]
