"""Small, hardware-aware command line entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from .autotune import build_runtime_plan, inspect_host
from .models.memory import rounded_initial_batch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _has_option(tokens: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in tokens)


def _append_default(tokens: list[str], passthrough: Sequence[str], option: str, value: object) -> None:
    if not _has_option(passthrough, option):
        tokens.extend((option, str(value)))


def _hf_token_available() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _redact_command(tokens: Iterable[str]) -> str:
    result: list[str] = []
    hide_next = False
    for token in tokens:
        if hide_next:
            result.append("<redacted>")
            hide_next = False
            continue
        if token == "--hf-token":
            result.append(token)
            hide_next = True
            continue
        if token.startswith("--hf-token="):
            result.append("--hf-token=<redacted>")
            continue
        result.append(token)
    return shlex.join(result)


def _train_defaults(
    *,
    objective: str,
    model_size: str,
    device: str,
    amp: bool,
    teacher_batch_factor: int,
    batch_cap: int,
    max_steps: int,
) -> list[str]:
    tokens = [
        "--device",
        device,
        "--max-steps",
        str(max_steps),
        "--model-size",
        model_size,
        "--shuffle-shards",
        "--canon",
        "--canon-2d",
        "--canon-abcd",
        "--canon-no-pos-enc",
        "--lr",
        "3e-4",
        "--lr-schedule",
        "cosine",
        "--auto-warmup",
        "--auto-warmup-max-batch-size",
        str(batch_cap),
        "--gns-every",
        "0",
    ]
    if objective != "reconstruct":
        tokens.append("--repeat")
    warmup_steps = min(1000, max(1, max_steps // 10))
    tokens.extend(("--auto-warmup-steps", str(warmup_steps)))
    if objective == "distill":
        tokens.extend(
            ("--auto-warmup-probe-batch-size", str(rounded_initial_batch(batch_cap)))
        )
    if amp:
        tokens.append("--amp")
    if objective == "distill" and teacher_batch_factor > 1:
        tokens.extend(
            (
                "--optimizer-mode",
                "teacher-superbatch",
                "--teacher-batch-factor",
                str(teacher_batch_factor),
            )
        )
    return tokens


def _run(args: argparse.Namespace, passthrough: Sequence[str]) -> int:
    repo_root = _repo_root()
    data_dir = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()
    resources = inspect_host(data_dir)
    plan = build_runtime_plan(
        resources,
        disk_fraction=args.disk_fraction,
        model_size=args.model_size,
        objective=args.objective,
    )
    os.environ.setdefault("HF_HOME", str(data_dir / "cache" / "huggingface"))

    if (
        args.objective == "distill"
        and not args.dry_run
        and not args.skip_auth_check
        and not _hf_token_available()
    ):
        raise SystemExit(
            "HeAR model access requires a Hugging Face login and acceptance of the model terms.\n"
            "Accept https://huggingface.co/google/hear-pytorch, run `.venv/bin/hf auth login`, then retry.\n"
            "The token itself never needs to be placed in this repository."
        )

    command: list[str] = [sys.executable, str(repo_root / "datalake" / "run_lake.py")]
    train_script = (
        Path("scripts/train/distill.py")
        if args.objective == "distill"
        else Path("scripts/train/pretrain_reconstruction.py")
    )
    train_out = args.train_out or (
        Path("checkpoints/hear_vit_s_lake")
        if args.objective == "distill"
        else Path("checkpoints/canon_audio_pretrain")
    )
    _append_default(command, passthrough, "--repo-root", repo_root)
    _append_default(command, passthrough, "--python", sys.executable)
    _append_default(command, passthrough, "--data-dir", args.data_dir)
    _append_default(command, passthrough, "--train-out", train_out)
    _append_default(command, passthrough, "--train-script", train_script)
    _append_default(command, passthrough, "--num-streams", plan.num_streams)
    _append_default(command, passthrough, "--min-streams", plan.min_streams)
    _append_default(command, passthrough, "--lake-max-gb", plan.lake_max_gib)
    _append_default(command, passthrough, "--disk-min-free-gb", plan.disk_min_free_gib)
    _append_default(command, passthrough, "--chunk-gb-per-stream", plan.chunk_gib_per_stream)
    _append_default(command, passthrough, "--reserve-low-gb", plan.reserve_low_gib)
    _append_default(command, passthrough, "--reserve-high-gb", plan.reserve_high_gib)
    _append_default(command, passthrough, "--val-max-gb", plan.val_max_gib)
    _append_default(command, passthrough, "--decay-max-gb", plan.decay_max_gib)
    _append_default(command, passthrough, "--shuffle-buffer", plan.shuffle_buffer)
    _append_default(command, passthrough, "--shard-size", plan.shard_size)
    _append_default(command, passthrough, "--curation-shard-size", plan.shard_size)
    _append_default(command, passthrough, "--train-batch-size", plan.train_batch_size)
    _append_default(command, passthrough, "--train-num-workers", plan.train_num_workers)
    if args.objective == "reconstruct" and not _has_option(passthrough, "--fresh-data"):
        command.append("--fresh-data")
    if not _has_option(passthrough, "--train-extra-args"):
        train_tokens = _train_defaults(
            objective=args.objective,
            model_size=args.model_size,
            device=plan.device,
            amp=plan.amp,
            teacher_batch_factor=plan.teacher_batch_factor,
            batch_cap=plan.train_batch_size,
            max_steps=args.max_steps,
        )
        command.extend(("--train-extra-args", shlex.join(train_tokens)))
    command.extend(passthrough)

    print(
        "[auto] "
        f"objective={args.objective} model={args.model_size} "
        f"cpu={resources.cpu_count} gpu={resources.gpu_name or 'none'} "
        f"gpu_mem={resources.gpu_memory_gib:.1f}GiB "
        f"disk={resources.disk_total_gib:.1f}GiB free={resources.disk_free_gib:.1f}GiB "
        f"budget={plan.disk_budget_gib:.1f}GiB ({plan.disk_fraction:.0%})",
        flush=True,
    )
    print(
        "[auto] "
        f"streams={plan.min_streams}-{plan.num_streams} loader_workers={plan.train_num_workers} "
        f"batch_cap={plan.train_batch_size} adaptive_batch=1 amp={int(plan.amp)} "
        f"lake={plan.lake_max_gib:.1f}GiB "
        f"hf_cache={os.environ['HF_HOME']}",
        flush=True,
    )
    if args.dry_run:
        print(_redact_command(command))
        return 0
    os.execv(sys.executable, command)
    return 0


def _doctor(args: argparse.Namespace) -> int:
    root = _repo_root()
    resources = inspect_host(root / "data")
    checks = {
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "adaptive_warmup": importlib.util.find_spec("adaptive_warmup") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "datasets": importlib.util.find_spec("datasets") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "huggingface_token": _hf_token_available(),
        "nvidia_gpu": bool(resources.gpu_name),
    }
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repository: {root}")
    print(
        f"Host: {resources.cpu_count} CPUs, disk {resources.disk_free_gib:.1f}/"
        f"{resources.disk_total_gib:.1f} GiB free"
    )
    print(
        f"GPU: {resources.gpu_name or 'not detected'}"
        + (f" ({resources.gpu_memory_gib:.1f} GiB)" if resources.gpu_name else "")
    )
    for name, ok in checks.items():
        print(f"{'OK' if ok else 'MISSING':7s} {name}")
    if not checks["huggingface_token"]:
        print("Action: accept the HeAR model terms, then run `.venv/bin/hf auth login`.")
    required = (
        checks["ffmpeg"]
        and checks["adaptive_warmup"]
        and checks["torch"]
        and checks["datasets"]
        and checks["transformers"]
    )
    if args.strict and (not required or not checks["huggingface_token"]):
        return 1
    return 0


def _defaults(args: argparse.Namespace) -> int:
    path = args.path.resolve()
    resources = inspect_host(path)
    plan = build_runtime_plan(
        resources,
        disk_fraction=args.disk_fraction,
        model_size=args.model_size,
        objective=args.objective,
    )
    payload = {
        "objective": args.objective,
        "model_size": args.model_size,
        "host": asdict(resources),
        "plan": asdict(plan),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hear-distill")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Auto-tune and start streaming + training.")
    run.add_argument(
        "--objective",
        choices=("distill", "reconstruct"),
        default="distill",
        help="Teacher distillation or direct masked-spectrogram pretraining.",
    )
    run.add_argument("--model-size", choices=("tiny", "small", "base", "large"), default="small")
    run.add_argument("--disk-fraction", type=float, default=0.5)
    run.add_argument("--max-steps", type=int, default=200_000)
    run.add_argument("--data-dir", type=Path, default=Path("data/laion_audio_lake"))
    run.add_argument("--train-out", type=Path, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--skip-auth-check", action="store_true")

    doctor = subparsers.add_parser("doctor", help="Check the local runtime without exposing credentials.")
    doctor.add_argument("--strict", action="store_true")

    defaults = subparsers.add_parser("defaults", help="Print the resolved hardware-aware plan.")
    defaults.add_argument("--path", type=Path, default=Path.cwd())
    defaults.add_argument("--disk-fraction", type=float, default=0.5)
    defaults.add_argument("--objective", choices=("distill", "reconstruct"), default="distill")
    defaults.add_argument("--model-size", choices=("tiny", "small", "base", "large"), default="small")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.command == "run":
        return _run(args, passthrough)
    if passthrough:
        parser.error(f"unrecognized arguments: {' '.join(passthrough)}")
    if args.command == "doctor":
        return _doctor(args)
    if args.command == "defaults":
        return _defaults(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
