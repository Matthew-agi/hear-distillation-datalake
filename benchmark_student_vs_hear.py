#!/usr/bin/env python3
from __future__ import annotations

"""Benchmark distilled student (no projection head) vs full HeAR.

This script compares embedding throughput on random audio clips using:
- Distilled student backbone features (optionally Canon/Canon2D from checkpoint args)
- Full HeAR model pooler output from Hugging Face
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _die(msg: str) -> "None":
  raise SystemExit(msg)


def _sync_device(device_type: str) -> None:
  if device_type == "cuda":
    import torch

    torch.cuda.synchronize()
  elif device_type == "mps":
    try:
      import torch

      torch.mps.synchronize()
    except (ImportError, AttributeError):
      pass


def _pick_device(requested: str) -> "tuple[str, object]":
  import torch

  if requested == "cpu":
    return "cpu", torch.device("cpu")
  if requested == "cuda":
    if not torch.cuda.is_available():
      print("Warning: CUDA requested but unavailable; falling back to CPU.", flush=True)
      return "cpu", torch.device("cpu")
    return "cuda", torch.device("cuda")
  if requested == "mps":
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
      print("Warning: MPS requested but unavailable; falling back to CPU.", flush=True)
      return "cpu", torch.device("cpu")
    return "mps", torch.device("mps")

  if torch.cuda.is_available():
    return "cuda", torch.device("cuda")
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return "mps", torch.device("mps")
  return "cpu", torch.device("cpu")


def _cpu_flags() -> set[str]:
  flags: set[str] = set()
  cpuinfo = Path("/proc/cpuinfo")
  if cpuinfo.exists():
    for line in cpuinfo.read_text().splitlines():
      if line.startswith("flags"):
        _, value = line.split(":", 1)
        flags.update(value.strip().split())
        break
  return flags


def _physical_cpu_ids() -> list[int]:
  cpu_root = Path("/sys/devices/system/cpu")
  if not cpu_root.exists():
    return []

  chosen: dict[tuple[int, int], int] = {}
  for cpu_dir in cpu_root.glob("cpu[0-9]*"):
    try:
      cpu_id = int(cpu_dir.name.removeprefix("cpu"))
    except ValueError:
      continue

    topo = cpu_dir / "topology"
    core_id_path = topo / "core_id"
    package_id_path = topo / "physical_package_id"
    if not core_id_path.exists() or not package_id_path.exists():
      continue

    try:
      core_id = int(core_id_path.read_text().strip())
      package_id = int(package_id_path.read_text().strip())
    except ValueError:
      continue

    key = (package_id, core_id)
    chosen[key] = min(cpu_id, chosen.get(key, cpu_id))

  return sorted(chosen.values())


def _set_cpu_affinity(cpu_ids: list[int]) -> None:
  if not cpu_ids:
    return
  if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, set(cpu_ids))


def _import_preprocess_audio(repo_root: Path):
  import importlib

  src_root = repo_root / "src"
  if src_root.exists() and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
  try:
    from hear_distill.audio import preprocess_audio

    return preprocess_audio
  except Exception:
    pass

  candidate_roots: List[Path] = [repo_root, repo_root.parent]
  checked: List[Path] = []
  last_exc: Optional[Exception] = None

  for root in candidate_roots:
    if root in checked:
      continue
    checked.append(root)
    if str(root) not in sys.path:
      sys.path.insert(0, str(root))
    try:
      audio_utils = importlib.import_module("hear.python.data_processing.audio_utils")
    except Exception as exc:  # noqa: BLE001
      last_exc = exc
      continue
    if not hasattr(audio_utils, "preprocess_audio"):
      _die("`audio_utils` import succeeded but has no `preprocess_audio` attribute.")
    return audio_utils.preprocess_audio

  roots_txt = ", ".join(str(p / "hear") for p in checked)
  _die(
    "Failed to import `hear.python.data_processing.audio_utils`.\n"
    "Expected a cloned `hear` repo at one of: "
    f"{roots_txt}\n"
    "Install dependencies (torch/scipy/numpy) and retry.\n"
    f"Original error: {last_exc}"
  )


def _safe_load_checkpoint(path: Path, *, unsafe_load: bool) -> dict:
  import torch

  if not path.exists():
    _die(f"Checkpoint not found: {path}")

  ckpt = None
  try:
    try:
      from torch.serialization import safe_globals  # type: ignore[attr-defined]
    except Exception:
      safe_globals = None

    if safe_globals is not None:
      import pathlib

      with safe_globals([pathlib.PosixPath]):
        try:
          ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
          ckpt = torch.load(path, map_location="cpu")
    else:
      ckpt = torch.load(path, map_location="cpu")
  except Exception as exc:  # noqa: BLE001
    if not unsafe_load:
      _die(
        "Safe checkpoint load failed. If you trust this checkpoint, retry with:\n"
        "  --unsafe-load\n"
        f"Original error: {exc}"
      )
    try:
      ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
      ckpt = torch.load(path, map_location="cpu")

  if not isinstance(ckpt, dict):
    _die("Unexpected checkpoint format (expected dict).")
  if "student" not in ckpt:
    _die("Checkpoint missing required key: 'student'.")
  return ckpt


def _resolve_canon_flags_from_dict(d: dict) -> Tuple[bool, bool, bool, bool, bool]:
  canon_a = bool(d.get("canon_a", False))
  canon_b = bool(d.get("canon_b", False))
  canon_c = bool(d.get("canon_c", False))
  canon_d = bool(d.get("canon_d", False))
  legacy_pre = bool(d.get("canon_pre", False))
  legacy_post = bool(d.get("canon_post", False))

  if d.get("canon_abcd", False):
    canon_a = canon_b = canon_c = canon_d = True

  if legacy_pre:
    canon_a = True
  if legacy_post:
    canon_c = True

  if d.get("canon", False) and not (canon_a or canon_b or canon_c or canon_d or legacy_pre or legacy_post):
    canon_a = canon_b = canon_c = canon_d = True

  use_canon = bool(
    d.get("canon", False)
    or d.get("canon_abcd", False)
    or canon_a
    or canon_b
    or canon_c
    or canon_d
    or legacy_pre
    or legacy_post
  )
  return use_canon, canon_a, canon_b, canon_c, canon_d


def _stats(times: List[float], *, num_clips: int) -> Dict[str, float]:
  avg = float(statistics.mean(times))
  stdev = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
  ms_per_clip = (avg / float(num_clips)) * 1000.0
  clips_per_s = (float(num_clips) / avg) if avg > 0 else float("inf")
  return {
    "avg_s": avg,
    "stdev_s": stdev,
    "ms_per_clip": float(ms_per_clip),
    "clips_per_s": float(clips_per_s),
  }


def _benchmark_model(
  *,
  name: str,
  audio,
  batch_size: int,
  warmup: int,
  repeats: int,
  device_type: str,
  preprocess_audio,
  run_forward,
):
  preprocess_times: List[float] = []
  model_times: List[float] = []
  total_times: List[float] = []
  last_embeddings = None

  total_passes = int(warmup + repeats)
  for p in range(total_passes):
    preprocess_s = 0.0
    model_s = 0.0
    embeddings = []

    for start in range(0, int(audio.shape[0]), int(batch_size)):
      end = min(int(audio.shape[0]), start + int(batch_size))
      batch = audio[start:end]

      _sync_device(device_type)
      t0 = time.perf_counter()
      spec = preprocess_audio(batch)
      _sync_device(device_type)
      t1 = time.perf_counter()

      emb = run_forward(spec)
      _sync_device(device_type)
      t2 = time.perf_counter()

      preprocess_s += t1 - t0
      model_s += t2 - t1
      embeddings.append(emb)

    import torch

    last_embeddings = torch.cat(embeddings, dim=0)
    total_s = preprocess_s + model_s

    if p >= warmup:
      preprocess_times.append(preprocess_s)
      model_times.append(model_s)
      total_times.append(total_s)

  assert last_embeddings is not None

  pre_stats = _stats(preprocess_times, num_clips=int(audio.shape[0]))
  model_stats = _stats(model_times, num_clips=int(audio.shape[0]))
  total_stats = _stats(total_times, num_clips=int(audio.shape[0]))

  return {
    "name": name,
    "embedding_shape": list(last_embeddings.shape),
    "embedding_dtype": str(last_embeddings.dtype),
    "preprocess": pre_stats,
    "model": model_stats,
    "total": total_stats,
  }


def _print_stats(label: str, payload: Dict[str, float]) -> None:
  print(
    f"{label}: avg={payload['avg_s']:.4f}s (±{payload['stdev_s']:.4f}s), "
    f"{payload['ms_per_clip']:.2f} ms/clip, {payload['clips_per_s']:.2f} clips/s"
  )


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(
    description="Benchmark distilled student (no projection head) vs full HeAR on random clips."
  )
  ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/hear_vit_s_lake/ckpt_final.pt"))
  ap.add_argument("--hear-model-id", type=str, default="google/hear-pytorch")
  ap.add_argument("--num-clips", type=int, default=64)
  ap.add_argument("--batch-size", type=int, default=64)
  ap.add_argument("--warmup", type=int, default=1)
  ap.add_argument("--repeats", type=int, default=5)
  ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
  ap.add_argument("--clip-seconds", type=float, default=None, help="Override clip seconds (default: from checkpoint or 2.0).")
  ap.add_argument("--sample-rate", type=int, default=None, help="Override sample rate (default: from checkpoint or 16000).")
  ap.add_argument("--pin-physical-cores", action="store_true", help="Pin process to one thread per physical core (CPU-only, Linux).")
  ap.add_argument("--threads", type=int, default=None, help="Optional torch CPU intra-op threads.")
  ap.add_argument("--interop-threads", type=int, default=None, help="Optional torch CPU inter-op threads.")
  ap.add_argument("--output-hidden-states", action="store_true", help="Request hidden states from full HeAR (usually slower).")
  ap.add_argument("--unsafe-load", action="store_true", help="Allow unsafe torch.load fallback for trusted checkpoints.")
  ap.add_argument("--save-json", type=Path, default=None, help="Optional path to save benchmark summary JSON.")
  return ap.parse_args()


def main() -> None:
  os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

  if sys.version_info < (3, 10):
    _die(f"Python 3.10+ required (found {sys.version.split()[0]}).")

  args = _parse_args()
  if args.num_clips <= 0:
    _die("--num-clips must be > 0.")
  if args.batch_size <= 0:
    _die("--batch-size must be > 0.")
  if args.warmup < 0:
    _die("--warmup must be >= 0.")
  if args.repeats <= 0:
    _die("--repeats must be > 0.")

  if args.pin_physical_cores:
    cpu_ids = _physical_cpu_ids()
    if not cpu_ids:
      _die("--pin-physical-cores requested, but could not determine physical cores on this system.")
    _set_cpu_affinity(cpu_ids)

  import torch
  from transformers import AutoModel

  if args.threads is not None:
    if args.threads <= 0:
      _die("--threads must be > 0.")
    torch.set_num_threads(args.threads)
  if args.interop_threads is not None:
    if args.interop_threads <= 0:
      _die("--interop-threads must be > 0.")
    torch.set_num_interop_threads(args.interop_threads)

  device_type, device = _pick_device(args.device)

  repo_root = Path(__file__).resolve().parent
  preprocess_audio = _import_preprocess_audio(repo_root)

  print(f"Python: {sys.version.split()[0]}")
  print(f"Platform: {platform.platform()}")
  print(f"Torch: {torch.__version__}")
  try:
    import transformers

    print(f"Transformers: {transformers.__version__}")
  except Exception:
    pass
  print(f"Device: {device_type}")
  if device_type == "cpu":
    print(f"Torch threads: {torch.get_num_threads()}")
    print(f"CPU AVX2: {'avx2' in _cpu_flags()}")
    if args.pin_physical_cores and hasattr(os, "sched_getaffinity"):
      affinity = sorted(os.sched_getaffinity(0))
      print(f"CPU affinity (count={len(affinity)}): {affinity}")

  ckpt = _safe_load_checkpoint(args.ckpt, unsafe_load=bool(args.unsafe_load))
  ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
  if not isinstance(ckpt_args, dict):
    ckpt_args = {}

  clip_seconds = float(args.clip_seconds) if args.clip_seconds is not None else float(ckpt_args.get("clip_seconds", 2.0))
  sample_rate = int(args.sample_rate) if args.sample_rate is not None else int(ckpt_args.get("sample_rate", 16000))
  clip_samples = int(round(float(clip_seconds) * float(sample_rate)))
  if clip_samples <= 0:
    _die("--clip-seconds and --sample-rate produced clip length <= 0.")

  from distill_hear_vit_s_canon2d import _build_student, _student_features

  use_canon, canon_a, canon_b, canon_c, canon_d = _resolve_canon_flags_from_dict(ckpt_args)
  student = _build_student(
    model_size=str(ckpt_args.get("model_size", "small")),
    use_canon=bool(use_canon),
    canon_2d=bool(ckpt_args.get("canon_2d", False)),
    canon_no_pos_enc=bool(ckpt_args.get("canon_no_pos_enc", False)),
    canon_kernel=int(ckpt_args.get("canon_kernel", 4)),
    canon_a=bool(canon_a),
    canon_b=bool(canon_b),
    canon_b_qkv=bool(ckpt_args.get("canon_b_qkv", False)),
    canon_c=bool(canon_c),
    canon_d=bool(canon_d),
    canon_causal=bool(ckpt_args.get("canon_causal", False)),
  ).to(device)
  student_state = ckpt.get("student") or ckpt.get("encoder")
  if not isinstance(student_state, dict):
    _die("Checkpoint has neither `student` nor `encoder` weights.")
  student.load_state_dict(student_state, strict=True)
  student.eval()

  print(
    "Student config: "
    f"canon={use_canon} canon2d={bool(ckpt_args.get('canon_2d', False))} "
    f"A={canon_a} B={canon_b} C={canon_c} D={canon_d} "
    f"kernel={int(ckpt_args.get('canon_kernel', 4))}"
  )

  print(f"Loading full HeAR model: {args.hear_model_id}")
  t0 = time.perf_counter()
  try:
    hear_model = AutoModel.from_pretrained(args.hear_model_id)
  except Exception as exc:  # noqa: BLE001
    _die(
      f"Failed to load model '{args.hear_model_id}'.\n"
      "If the repo is gated, run `huggingface-cli login` or set `HF_TOKEN`.\n"
      f"Original error: {exc}"
    )
  hear_load_s = time.perf_counter() - t0
  hear_model.eval().to(device)
  print(f"HeAR loaded in {hear_load_s:.2f}s")

  audio = torch.rand((int(args.num_clips), int(clip_samples)), dtype=torch.float32, device=device)
  print(
    f"Benchmark config: clips={args.num_clips} batch_size={args.batch_size} "
    f"clip_seconds={clip_seconds:.3f} sample_rate={sample_rate} warmup={args.warmup} repeats={args.repeats}"
  )

  with torch.inference_mode():
    student_result = _benchmark_model(
      name="student",
      audio=audio,
      batch_size=int(args.batch_size),
      warmup=int(args.warmup),
      repeats=int(args.repeats),
      device_type=device_type,
      preprocess_audio=preprocess_audio,
      run_forward=lambda spec: _student_features(student, spec),
    )

    hear_result = _benchmark_model(
      name="hear",
      audio=audio,
      batch_size=int(args.batch_size),
      warmup=int(args.warmup),
      repeats=int(args.repeats),
      device_type=device_type,
      preprocess_audio=preprocess_audio,
      run_forward=lambda spec: hear_model.forward(
        spec,
        return_dict=True,
        output_hidden_states=bool(args.output_hidden_states),
      ).pooler_output,
    )

  print("\n== Student (No Projection Head) ==")
  print(f"Embeddings: shape={tuple(student_result['embedding_shape'])}, dtype={student_result['embedding_dtype']}")
  _print_stats("Preprocess", student_result["preprocess"])
  _print_stats("Model", student_result["model"])
  _print_stats("Total", student_result["total"])

  print("\n== Full HeAR ==")
  print(f"Embeddings: shape={tuple(hear_result['embedding_shape'])}, dtype={hear_result['embedding_dtype']}")
  _print_stats("Preprocess", hear_result["preprocess"])
  _print_stats("Model", hear_result["model"])
  _print_stats("Total", hear_result["total"])

  student_model_cps = float(student_result["model"]["clips_per_s"])
  hear_model_cps = float(hear_result["model"]["clips_per_s"])
  student_total_cps = float(student_result["total"]["clips_per_s"])
  hear_total_cps = float(hear_result["total"]["clips_per_s"])

  ratio_model = (student_model_cps / hear_model_cps) if hear_model_cps > 0 else float("inf")
  ratio_total = (student_total_cps / hear_total_cps) if hear_total_cps > 0 else float("inf")

  print("\n== Relative Speedup (Student / HeAR) ==")
  print(f"Model-only speedup: {ratio_model:.3f}x")
  print(f"End-to-end speedup: {ratio_total:.3f}x")

  summary = {
    "config": {
      "ckpt": str(args.ckpt),
      "hear_model_id": str(args.hear_model_id),
      "num_clips": int(args.num_clips),
      "batch_size": int(args.batch_size),
      "clip_seconds": float(clip_seconds),
      "sample_rate": int(sample_rate),
      "warmup": int(args.warmup),
      "repeats": int(args.repeats),
      "device": str(device_type),
      "student_uses_projection_head": False,
    },
    "student": student_result,
    "hear": hear_result,
    "relative": {
      "model_speedup_student_over_hear": float(ratio_model),
      "total_speedup_student_over_hear": float(ratio_total),
    },
  }

  if args.save_json is not None:
    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary JSON to: {args.save_json}")


if __name__ == "__main__":
  main()
