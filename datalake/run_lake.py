#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


TRAIN_STEP_RE = re.compile(r"\bstep=(\d+)\b")
TRAIN_RESUMED_RE = re.compile(r"^Resumed training from\s+(.+)\s+at step=(\d+)\.")
STREAM_RESUME_RE = re.compile(r"^Resume:\s+found\s+(\d+)\s+clips\s+across\s+existing\s+shards\.")
STREAM_PROGRESS_RE = re.compile(
  r"^PROGRESS\s+written=(\d+)/(\d+)\s+seen=(\d+)\s+skipped=(\d+)\s+errors=(\d+)\s+rate=([0-9.]+)\s+clips/s\s+elapsed=([0-9.]+)s"
)
STREAM_DONE_RE = re.compile(
  r"^DONE\s+written=(\d+)\s+seen=(\d+)\s+skipped=(\d+)\s+elapsed=([0-9.]+)s\s+rate=([0-9.]+)\s+clips/s"
)
UNSAFE_RESUME_RE = re.compile(r"^Refusing unsafe resume:")
CKPT_STEP_FILENAME_RE = re.compile(r"^ckpt_(\d+)\.pt$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _die(msg: str) -> "None":
  raise SystemExit(msg)


def _fmt_int(x: int) -> str:
  return f"{x:,}"


def _ema(prev: float, new: float, alpha: float) -> float:
  if prev <= 0:
    return new
  return (alpha * new) + ((1.0 - alpha) * prev)


def _disk_free_gb(path: Path) -> float:
  usage = shutil.disk_usage(path)
  return usage.free / (1024.0 ** 3)


def _truthy(value: Optional[str]) -> bool:
  if value is None:
    return False
  return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_env_file(path: Path) -> Dict[str, str]:
  if not path.exists():
    return {}
  out: Dict[str, str] = {}
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except Exception:
    return {}
  for raw in lines:
    line = raw.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("export "):
      line = line[len("export ") :].strip()
    if "=" not in line:
      continue
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not ENV_KEY_RE.match(key):
      continue
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
      value = value[1:-1]
    out[key] = value
  return out


def _has_visible_shards(data_dir: Path) -> bool:
  return any(data_dir.glob("stream-*/shard-*.tar")) or any(data_dir.glob("shard-*.tar"))


def _count_wavs_in_tar(path: Path) -> int:
  count = 0
  try:
    with tarfile.open(path, "r") as tf:
      for m in tf:
        if m.isfile() and m.name.endswith(".wav"):
          count += 1
  except Exception:
    return 0
  return count


def _parse_stream_shard(path: Path) -> Tuple[int, int]:
  stream_idx = -1
  parent_name = path.parent.name
  if parent_name.startswith("stream-"):
    try:
      stream_idx = int(parent_name.split("-")[-1])
    except Exception:
      stream_idx = -1
  shard_idx = -1
  stem = path.stem  # shard-000123
  if stem.startswith("shard-"):
    try:
      shard_idx = int(stem.split("-")[-1])
    except Exception:
      shard_idx = -1
  return stream_idx, shard_idx


def _discover_final_shards(data_dir: Path) -> List[Path]:
  shards = sorted(data_dir.glob("stream-*/shard-*.tar"))
  shards.extend(sorted(data_dir.glob("shard-*.tar")))
  dedup: Dict[str, Path] = {}
  for p in shards:
    dedup[str(p.resolve())] = p
  out = list(dedup.values())
  out.sort(key=lambda p: (_parse_stream_shard(p)[0], _parse_stream_shard(p)[1], str(p)))
  return out


def _load_val_shard_protection(path: Path) -> Set[str]:
  if not path.exists():
    return set()
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return set()
  if not isinstance(payload, dict):
    return set()
  raw = payload.get("val_shards")
  if not isinstance(raw, list):
    return set()
  out: Set[str] = set()
  for item in raw:
    if not isinstance(item, str):
      continue
    try:
      out.add(str(Path(item).resolve()))
    except Exception:
      continue
  return out


def _build_shard_inventory(
  data_dir: Path,
  clip_count_cache: Dict[str, Tuple[int, int, int]],
) -> List[Tuple[Path, str, int, int, int, int, int]]:
  """
  Returns list of shard records:
    (path, key, clips, bytes, mtime_ns, stream_idx, shard_idx)
  """
  records: List[Tuple[Path, str, int, int, int, int, int]] = []
  for p in _discover_final_shards(data_dir):
    try:
      st = p.stat()
    except FileNotFoundError:
      continue
    except OSError:
      continue
    key = str(p.resolve())
    cache = clip_count_cache.get(key)
    if cache is not None and cache[0] == int(st.st_size) and cache[1] == int(st.st_mtime_ns):
      clips = int(cache[2])
    else:
      clips = int(_count_wavs_in_tar(p))
      clip_count_cache[key] = (int(st.st_size), int(st.st_mtime_ns), clips)
    sidx, shidx = _parse_stream_shard(p)
    records.append((p, key, clips, int(st.st_size), int(st.st_mtime_ns), sidx, shidx))
  records.sort(key=lambda r: (r[4], r[5], r[6], r[1]))
  return records


def _prune_old_shards(
  records: List[Tuple[Path, str, int, int, int, int, int]],
  *,
  protected_keys: Set[str],
  reserve_clips: int,
  reserve_floor_clips: int,
  target_non_val_bytes: int,
  keep_recent_shards_per_stream: int,
  max_delete_shards: int,
) -> Tuple[int, int, int]:
  if not records or max_delete_shards <= 0:
    return 0, 0, 0

  keep_keys: Set[str] = set()
  if keep_recent_shards_per_stream > 0:
    by_stream: Dict[int, List[Tuple[Path, str, int, int, int, int, int]]] = {}
    for rec in records:
      by_stream.setdefault(rec[5], []).append(rec)
    for items in by_stream.values():
      for rec in items[-keep_recent_shards_per_stream:]:
        keep_keys.add(rec[1])

  current_non_val_bytes = sum(rec[3] for rec in records if rec[1] not in protected_keys)
  max_delete_clips = max(0, int(reserve_clips) - int(reserve_floor_clips))

  deleted_bytes = 0
  deleted_clips = 0
  deleted_shards = 0
  for rec in records:
    if deleted_shards >= max_delete_shards:
      break
    path, key, clips, nbytes, _mtime_ns, _sidx, _shidx = rec
    if key in protected_keys or key in keep_keys:
      continue
    need_bytes_relief = current_non_val_bytes > target_non_val_bytes
    need_clip_relief = deleted_clips < max_delete_clips
    if not need_bytes_relief and not need_clip_relief:
      break
    if (not need_bytes_relief) and (deleted_clips + clips > max_delete_clips):
      continue
    try:
      path.unlink()
    except FileNotFoundError:
      continue
    except OSError:
      continue
    deleted_bytes += nbytes
    deleted_clips += clips
    deleted_shards += 1
    current_non_val_bytes = max(0, current_non_val_bytes - nbytes)
  return deleted_shards, deleted_clips, deleted_bytes


def _stream_dirs(data_dir: Path) -> List[Tuple[int, Path]]:
  dirs: List[Tuple[int, Path]] = []
  for d in sorted([p for p in data_dir.glob("stream-*") if p.is_dir()]):
    try:
      idx = int(d.name.split("-")[-1])
    except Exception:
      continue
    dirs.append((idx, d))
  if dirs:
    return dirs
  return [(0, data_dir)]


def _initial_counts(data_dir: Path, *, exact: bool) -> Dict[int, int]:
  out: Dict[int, int] = {}
  for idx, d in _stream_dirs(data_dir):
    if not d.exists():
      out[idx] = 0
      continue
    if not exact:
      manifest = d / "manifest.json"
      if manifest.exists():
        try:
          import json

          payload = json.loads(manifest.read_text(encoding="utf-8"))
          out[idx] = int(payload.get("num_clips", 0))
          continue
        except Exception:
          pass
    total = 0
    for tar_path in sorted(d.glob("shard-*.tar")):
      total += _count_wavs_in_tar(tar_path)
    out[idx] = total
  return out


def _find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
  best: Optional[Tuple[int, Path]] = None
  for p in out_dir.glob("ckpt_*.pt"):
    m = CKPT_STEP_FILENAME_RE.match(p.name)
    if m is None:
      continue
    cur = (int(m.group(1)), p)
    if best is None or cur[0] > best[0]:
      best = cur
  if best is not None:
    return best[1]
  final_p = out_dir / "ckpt_final.pt"
  if final_p.exists():
    return final_p
  return None


def _parse_resume_request(train_extra_tokens: List[str]) -> Tuple[Optional[str], bool]:
  resume_from_raw: Optional[str] = None
  resume_latest = False
  i = 0
  while i < len(train_extra_tokens):
    tok = train_extra_tokens[i]
    if tok == "--resume-latest":
      resume_latest = True
      i += 1
      continue
    if tok.startswith("--resume-from="):
      resume_from_raw = tok.split("=", 1)[1]
      i += 1
      continue
    if tok == "--resume-from":
      if i + 1 >= len(train_extra_tokens):
        _die("Missing value for --resume-from in --train-extra-args.")
      resume_from_raw = train_extra_tokens[i + 1]
      i += 2
      continue
    i += 1
  if resume_from_raw is not None and resume_latest:
    _die("Use only one of --resume-from or --resume-latest in --train-extra-args.")
  return resume_from_raw, resume_latest


def _resolve_resume_checkpoint(
  *,
  repo_root: Path,
  train_out: Path,
  train_extra_tokens: List[str],
) -> Optional[Path]:
  resume_from_raw, resume_latest = _parse_resume_request(train_extra_tokens)
  if resume_from_raw is not None:
    p = Path(resume_from_raw)
    p = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
    if not p.exists():
      _die(f"--resume-from path in --train-extra-args does not exist: {p}")
    return p
  if resume_latest:
    p = _find_latest_checkpoint(train_out)
    if p is None:
      _die(f"--resume-latest requested, but no checkpoint found in {train_out}.")
    return p.resolve()
  return None


def _checkpoint_step_hint(path: Path) -> int:
  m = CKPT_STEP_FILENAME_RE.match(path.name)
  if m is not None:
    return int(m.group(1))
  try:
    import torch

    try:
      ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
      ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
      return int(ckpt.get("step", 0))
  except Exception:
    return 0
  return 0


def _start_process(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
  merged_env = os.environ.copy()
  if env:
    merged_env.update(env)
  return subprocess.Popen(
    cmd,
    cwd=str(cwd),
    env=merged_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
  )


def _reader(name: str, proc: subprocess.Popen, q: queue.Queue) -> None:
  assert proc.stdout is not None
  for line in iter(proc.stdout.readline, ""):
    q.put((name, line.rstrip("\n")))
  try:
    proc.stdout.close()
  except Exception:
    pass


def _worker_indices_by_need(
  stream_written: Dict[int, int],
  running_workers: Set[int],
  max_workers: int,
) -> List[int]:
  candidates = [i for i in range(max_workers) if i not in running_workers]
  return sorted(candidates, key=lambda i: (stream_written.get(i, 0), i))


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Run streaming + training as a bounded adaptive data lake pipeline.")
  ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
  ap.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional env file to load before launching workers.")
  ap.add_argument("--data-dir", type=Path, default=Path("data/laion_audio_lake"))
  ap.add_argument("--train-script", type=Path, default=Path("distill_hear_vit_s_canon2d.py"))
  ap.add_argument("--stream-script", type=Path, default=Path("stream_laion_audio_clips.py"))
  ap.add_argument("--python", type=str, default="python3")

  ap.add_argument("--num-streams", type=int, default=3, help="Maximum stream workers.")
  ap.add_argument("--min-streams", type=int, default=1, help="Minimum stream workers when streaming is needed.")
  ap.add_argument("--auto-tune-streams", action=argparse.BooleanOptionalAction, default=True, help="Adapt active stream workers to training throughput.")
  ap.add_argument("--stream-headroom", type=float, default=1.10, help="Target write/consume ratio when auto tuning.")
  ap.add_argument("--rate-ema-alpha", type=float, default=0.25, help="EMA alpha for throughput estimates.")
  ap.add_argument("--chunk-clips-per-stream", type=int, default=20000, help="Per-worker clip increment per chunk.")
  ap.add_argument("--reserve-low-clips", type=int, default=150000, help="If reserve falls below this, grow streaming.")
  ap.add_argument("--reserve-high-clips", type=int, default=450000, help="If reserve exceeds this, pause streaming.")
  ap.add_argument("--disk-min-free-gb", type=float, default=20.0)
  ap.add_argument("--lake-max-gb", type=float, default=0.0, help="Max non-validation lake size in GB (0 disables cap).")
  ap.add_argument(
    "--lake-resume-fraction",
    type=float,
    default=0.5,
    help="Resume streaming when non-validation lake drops below this fraction of --lake-max-gb.",
  )
  ap.add_argument("--inventory-refresh-sec", type=float, default=10.0, help="Seconds between shard inventory refreshes.")
  ap.add_argument(
    "--prune-consumed",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Delete old non-validation shards as training consumes data while respecting reserve floor.",
  )
  ap.add_argument("--prune-every-sec", type=float, default=15.0, help="Seconds between prune checks.")
  ap.add_argument("--prune-max-shards", type=int, default=16, help="Max shards deleted per prune cycle.")
  ap.add_argument("--prune-keep-recent-shards", type=int, default=1, help="Keep this many latest shards per stream.")
  ap.add_argument(
    "--min-active-shards-before-pause",
    type=int,
    default=3,
    help="Minimum shards a worker should write after (re)start before being paused by target-workers logic.",
  )
  ap.add_argument("--status-every-sec", type=float, default=20.0)
  ap.add_argument("--poll-sec", type=float, default=1.0)
  ap.add_argument("--max-stream-failures", type=int, default=10)
  ap.add_argument("--exact-start-count", action=argparse.BooleanOptionalAction, default=True)
  ap.add_argument("--val-shards-file", type=Path, default=None, help="Validation shard protection JSON file from trainer.")

  ap.add_argument("--clip-seconds", type=float, default=2.0)
  ap.add_argument("--sample-rate", type=int, default=16000)
  ap.add_argument("--shuffle-buffer", type=int, default=20000)
  ap.add_argument("--shard-size", type=int, default=1000)
  ap.add_argument("--hf-transfer", action="store_true")
  ap.add_argument("--hf-token", type=str, default=None)
  ap.add_argument("--stream-extra-args", type=str, default="")

  ap.add_argument("--train-batch-size", type=int, default=64)
  ap.add_argument("--train-grad-accum", type=int, default=1)
  ap.add_argument("--train-num-workers", type=int, default=4)
  ap.add_argument("--train-out", type=Path, default=Path("checkpoints/hear_vit_s_lake"))
  ap.add_argument("--shard-refresh-sec", type=float, default=20.0)
  ap.add_argument("--train-extra-args", type=str, default="")
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  if args.reserve_low_clips < 0 or args.reserve_high_clips <= 0:
    _die("Reserve thresholds must be positive.")
  if args.reserve_low_clips >= args.reserve_high_clips:
    _die("--reserve-low-clips must be < --reserve-high-clips")
  if args.chunk_clips_per_stream <= 0:
    _die("--chunk-clips-per-stream must be > 0")
  if args.num_streams <= 0:
    _die("--num-streams must be > 0")
  if args.min_streams < 0 or args.min_streams > args.num_streams:
    _die("--min-streams must be in [0, --num-streams]")
  if args.stream_headroom <= 0:
    _die("--stream-headroom must be > 0")
  if not (0.0 < args.rate_ema_alpha <= 1.0):
    _die("--rate-ema-alpha must be in (0, 1]")
  if args.lake_max_gb < 0:
    _die("--lake-max-gb must be >= 0")
  if not (0.0 < args.lake_resume_fraction <= 1.0):
    _die("--lake-resume-fraction must be in (0, 1]")
  if args.inventory_refresh_sec <= 0:
    _die("--inventory-refresh-sec must be > 0")
  if args.prune_every_sec <= 0:
    _die("--prune-every-sec must be > 0")
  if args.prune_max_shards <= 0:
    _die("--prune-max-shards must be > 0")
  if args.prune_keep_recent_shards < 0:
    _die("--prune-keep-recent-shards must be >= 0")
  if args.min_active_shards_before_pause < 0:
    _die("--min-active-shards-before-pause must be >= 0")

  repo_root = args.repo_root.resolve()
  env_file = args.env_file
  if not env_file.is_absolute():
    env_file = (repo_root / env_file).resolve()
  else:
    env_file = env_file.resolve()

  loaded_env = _load_env_file(env_file)
  if loaded_env:
    for key, value in loaded_env.items():
      os.environ.setdefault(key, value)
    print(f"[lake] loaded env file: {env_file} ({len(loaded_env)} vars)", flush=True)

  if args.hf_token is None:
    env_hf_token = os.environ.get("HF_TOKEN")
    if env_hf_token:
      args.hf_token = env_hf_token
  if _truthy(os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")):
    args.hf_transfer = True

  data_dir = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()
  train_script = (repo_root / args.train_script).resolve() if not args.train_script.is_absolute() else args.train_script.resolve()
  stream_script = (repo_root / args.stream_script).resolve() if not args.stream_script.is_absolute() else args.stream_script.resolve()
  train_out = (repo_root / args.train_out).resolve() if not args.train_out.is_absolute() else args.train_out.resolve()
  val_shards_file = args.val_shards_file
  if val_shards_file is None:
    val_shards_file = train_out / "val_shards.json"
  elif not val_shards_file.is_absolute():
    val_shards_file = (repo_root / val_shards_file).resolve()
  else:
    val_shards_file = val_shards_file.resolve()

  if not train_script.exists():
    _die(f"Training script not found: {train_script}")
  if not stream_script.exists():
    _die(f"Streaming script not found: {stream_script}")

  data_dir.mkdir(parents=True, exist_ok=True)
  train_out.mkdir(parents=True, exist_ok=True)

  train_extra_tokens: List[str] = []
  if args.train_extra_args:
    try:
      train_extra_tokens = shlex.split(args.train_extra_args)
    except ValueError as exc:
      _die(f"Failed to parse --train-extra-args: {exc}")

  resume_ckpt_path = _resolve_resume_checkpoint(
    repo_root=repo_root,
    train_out=train_out,
    train_extra_tokens=train_extra_tokens,
  )
  initial_step = 0
  if resume_ckpt_path is not None:
    initial_step = max(0, _checkpoint_step_hint(resume_ckpt_path))
    print(
      f"[lake] resume checkpoint={resume_ckpt_path} step_hint={initial_step}",
      flush=True,
    )

  print("[lake] counting existing clips (startup)", flush=True)
  stream_written = _initial_counts(data_dir, exact=args.exact_start_count)
  for i in range(args.num_streams):
    stream_written.setdefault(i, 0)

  q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
  current_step = int(initial_step)
  resume_guard_active = bool(resume_ckpt_path is not None and current_step > 0)
  trainer_resume_confirmed = False
  train_proc: Optional[subprocess.Popen] = None
  stream_workers: Dict[int, subprocess.Popen] = {}
  paused_workers: Set[int] = set()
  stream_stop_requested: Set[int] = set()
  stream_resume_blocked: Set[int] = set()
  worker_active_start_written: Dict[int, int] = {}
  stream_failures = 0

  train_rate_ema = 0.0
  write_rate_ema = 0.0
  per_worker_write_rate_ema = 0.0
  deleted_clips_total = 0
  deleted_bytes_total = 0
  last_rate_t = time.time()
  last_consumed_total = current_step * args.train_batch_size * args.train_grad_accum
  last_written_total = sum(stream_written.values())
  clip_count_cache: Dict[str, Tuple[int, int, int]] = {}
  inventory_records: List[Tuple[Path, str, int, int, int, int, int]] = []
  protected_val_shards: Set[str] = set()
  last_inventory_t = 0.0
  last_prune_t = 0.0
  lake_blocked = False

  train_cmd = [
    args.python,
    str(train_script),
    "--data-dir",
    str(data_dir),
    "--out",
    str(train_out),
    "--batch-size",
    str(args.train_batch_size),
    "--grad-accum",
    str(args.train_grad_accum),
    "--num-workers",
    str(args.train_num_workers),
    "--repeat",
    "--live-shard-refresh",
    "--shard-refresh-sec",
    str(args.shard_refresh_sec),
    "--val-shards-file",
    str(val_shards_file),
  ]
  if train_extra_tokens:
    train_cmd.extend(train_extra_tokens)

  def _start_training() -> subprocess.Popen:
    print(f"[lake] starting training: {' '.join(train_cmd)}", flush=True)
    proc = _start_process(train_cmd, cwd=repo_root)
    t = threading.Thread(target=_reader, args=("train", proc, q), daemon=True)
    t.start()
    return proc

  def _start_stream_worker(worker_idx: int) -> None:
    if worker_idx in stream_resume_blocked:
      return
    if worker_idx in stream_workers and stream_workers[worker_idx].poll() is None:
      return
    target = stream_written.get(worker_idx, 0) + args.chunk_clips_per_stream
    cmd = [
      args.python,
      str(stream_script),
      "--out",
      str(data_dir),
      "--num-clips",
      str(target),
      "--clip-seconds",
      str(args.clip_seconds),
      "--sample-rate",
      str(args.sample_rate),
      "--shuffle-buffer",
      str(args.shuffle_buffer),
      "--shard-size",
      str(args.shard_size),
      "--num-streams",
      str(args.num_streams),
      "--stream-index",
      str(worker_idx),
      "--resume",
    ]
    if args.hf_transfer:
      cmd.append("--hf-transfer")
    if args.hf_token:
      cmd.extend(["--hf-token", args.hf_token])
    if args.stream_extra_args:
      cmd.extend(shlex.split(args.stream_extra_args))
    print(
      f"[lake] start stream worker={worker_idx} target={target} current={stream_written.get(worker_idx, 0)}",
      flush=True,
    )
    proc = _start_process(cmd, cwd=repo_root)
    stream_workers[worker_idx] = proc
    paused_workers.discard(worker_idx)
    stream_stop_requested.discard(worker_idx)
    worker_active_start_written[worker_idx] = int(stream_written.get(worker_idx, 0))
    t = threading.Thread(target=_reader, args=(f"stream:{worker_idx}", proc, q), daemon=True)
    t.start()

  def _pause_stream_worker(worker_idx: int, reason: str) -> None:
    proc = stream_workers.get(worker_idx)
    if proc is None or proc.poll() is not None or worker_idx in paused_workers:
      return
    print(f"[lake] pause stream worker={worker_idx} reason={reason}", flush=True)
    try:
      proc.send_signal(signal.SIGSTOP)
      paused_workers.add(worker_idx)
    except Exception:
      pass

  def _resume_stream_worker(worker_idx: int, reason: str) -> None:
    proc = stream_workers.get(worker_idx)
    if proc is None or proc.poll() is not None or worker_idx not in paused_workers:
      return
    print(f"[lake] resume stream worker={worker_idx} reason={reason}", flush=True)
    try:
      proc.send_signal(signal.SIGCONT)
      paused_workers.discard(worker_idx)
      worker_active_start_written[worker_idx] = int(stream_written.get(worker_idx, 0))
    except Exception:
      pass

  def _stop_stream_worker(worker_idx: int, reason: str) -> None:
    proc = stream_workers.get(worker_idx)
    if proc is None or proc.poll() is not None or worker_idx in stream_stop_requested:
      return
    print(f"[lake] stop stream worker={worker_idx} reason={reason}", flush=True)
    try:
      if worker_idx in paused_workers:
        proc.send_signal(signal.SIGCONT)
        paused_workers.discard(worker_idx)
      proc.send_signal(signal.SIGINT)
      stream_stop_requested.add(worker_idx)
    except Exception:
      pass

  def _running_workers() -> List[int]:
    out: List[int] = []
    for idx, proc in stream_workers.items():
      if proc.poll() is None:
        out.append(idx)
    return sorted(out)

  def _active_workers() -> List[int]:
    return [idx for idx in _running_workers() if idx not in paused_workers]

  def _paused_alive_workers() -> List[int]:
    return [idx for idx in _running_workers() if idx in paused_workers]

  def _reap_workers() -> None:
    nonlocal stream_failures
    finished: List[int] = []
    for idx, proc in stream_workers.items():
      rc = proc.poll()
      if rc is None:
        continue
      finished.append(idx)
      was_requested = idx in stream_stop_requested
      stream_stop_requested.discard(idx)
      if rc != 0 and not was_requested:
        if idx in stream_resume_blocked:
          print(
            f"[lake] stream worker blocked idx={idx} rc={rc} reason=unsafe_resume",
            flush=True,
          )
        else:
          stream_failures += 1
          print(
            f"[lake] stream worker failed idx={idx} rc={rc} (failures={stream_failures}/{args.max_stream_failures})",
            flush=True,
          )
      else:
        if stream_failures > 0:
          stream_failures -= 1
    for idx in finished:
      stream_workers.pop(idx, None)
      paused_workers.discard(idx)
      worker_active_start_written.pop(idx, None)
    if stream_failures >= args.max_stream_failures:
      _die("Too many stream worker failures.")

  def _desired_workers(
    reserve: int,
    consumed_rate: float,
    write_rate: float,
    running_count: int,
  ) -> int:
    if reserve >= args.reserve_high_clips:
      return 0

    if not args.auto_tune_streams:
      if reserve < args.reserve_low_clips:
        return max(1, args.min_streams)
      return running_count

    if consumed_rate <= 1e-6:
      base_required = max(1, args.min_streams)
    else:
      per_worker = per_worker_write_rate_ema
      if per_worker <= 1e-6:
        if running_count > 0 and write_rate > 1e-6:
          per_worker = write_rate / running_count
      if per_worker <= 1e-6:
        base_required = max(1, args.min_streams)
      else:
        base_required = max(1, int(math.ceil((consumed_rate * args.stream_headroom) / per_worker)))

    base_required = max(args.min_streams, min(args.num_streams, base_required))
    reserve_mid = int((args.reserve_low_clips + args.reserve_high_clips) / 2)

    if reserve <= args.reserve_low_clips:
      return base_required
    if reserve < reserve_mid:
      if write_rate < consumed_rate * 1.05:
        return base_required
      return max(args.min_streams, min(args.num_streams, running_count))
    if write_rate > consumed_rate * 1.10:
      return max(0, base_required - 1)
    return max(0, min(args.num_streams, running_count))

  def _reconcile_worker_count(desired: int, free_gb: float, *, force_pause: bool) -> None:
    active = _active_workers()
    paused = _paused_alive_workers()
    active_count = len(active)
    min_pause_clips = int(args.min_active_shards_before_pause) * int(args.shard_size)

    if desired < active_count:
      to_pause: List[int] = []
      if force_pause or min_pause_clips <= 0:
        to_pause = active[desired:]
      else:
        for idx in active[desired:]:
          started_written = int(worker_active_start_written.get(idx, stream_written.get(idx, 0)))
          produced = int(stream_written.get(idx, 0)) - started_written
          if produced >= min_pause_clips:
            to_pause.append(idx)
      for idx in to_pause:
        _pause_stream_worker(idx, reason="target_workers")
      return

    if desired == active_count:
      return

    needed = desired - active_count
    if needed > 0 and paused:
      for idx in paused[:needed]:
        _resume_stream_worker(idx, reason="target_workers")
      active = _active_workers()
      needed = max(0, desired - len(active))
    if needed <= 0:
      return

    if free_gb < args.disk_min_free_gb:
      return

    running = _running_workers()
    order = _worker_indices_by_need(stream_written, set(running), args.num_streams)
    for idx in order[:needed]:
      _start_stream_worker(idx)

  def _bootstrap_streams_until_first_shard() -> None:
    bootstrap_workers = max(1, args.min_streams)
    free_gb = _disk_free_gb(data_dir)
    if free_gb < args.disk_min_free_gb:
      _die(
        f"No initial shards and insufficient free space ({free_gb:.1f} GB < {args.disk_min_free_gb:.1f} GB). "
        "Cannot bootstrap stream safely."
      )
    order = _worker_indices_by_need(stream_written, set(), args.num_streams)
    for idx in order[:bootstrap_workers]:
      _start_stream_worker(idx)
    print("[lake] waiting for first shard before starting training", flush=True)
    while not _has_visible_shards(data_dir):
      _reap_workers()
      if not _running_workers():
        _die("Bootstrap streaming stopped before first shard was written.")
      try:
        src, line = q.get(timeout=args.poll_sec)
      except queue.Empty:
        continue
      if not src.startswith("stream:"):
        continue
      worker_idx = int(src.split(":")[1])
      print(f"[stream {worker_idx}] {line}", flush=True)
      m = STREAM_RESUME_RE.match(line)
      if m:
        stream_written[worker_idx] = max(stream_written.get(worker_idx, 0), int(m.group(1)))
        continue
      m = STREAM_PROGRESS_RE.match(line)
      if m:
        stream_written[worker_idx] = max(stream_written.get(worker_idx, 0), int(m.group(1)))
        continue
      m = STREAM_DONE_RE.match(line)
      if m:
        stream_written[worker_idx] = max(stream_written.get(worker_idx, 0), int(m.group(1)))

  if not _has_visible_shards(data_dir):
    _bootstrap_streams_until_first_shard()

  train_proc = _start_training()
  last_status_t = 0.0

  try:
    while True:
      deadline = time.time() + args.poll_sec
      while True:
        timeout = max(0.0, deadline - time.time())
        if timeout <= 0:
          break
        try:
          src, line = q.get(timeout=timeout)
        except queue.Empty:
          break

        if src == "train":
          print(f"[train] {line}", flush=True)
          m_resumed = TRAIN_RESUMED_RE.match(line)
          if m_resumed:
            trainer_resume_confirmed = True
            try:
              resumed_step = int(m_resumed.group(2))
              current_step = max(current_step, resumed_step)
              if resume_guard_active and resumed_step < initial_step:
                _die(f"Resume mismatch: expected step >= {initial_step}, trainer reported {resumed_step}.")
            except Exception:
              pass
          m = TRAIN_STEP_RE.search(line)
          if m:
            observed_step = int(m.group(1))
            if resume_guard_active and (not trainer_resume_confirmed) and observed_step < initial_step:
              _die(
                "Resume mismatch before trainer confirmation: "
                f"first observed step {observed_step} < expected {initial_step}."
              )
            current_step = max(current_step, observed_step)
          continue

        if src.startswith("stream:"):
          worker_idx = int(src.split(":")[1])
          print(f"[stream {worker_idx}] {line}", flush=True)
          m = STREAM_RESUME_RE.match(line)
          if m:
            stream_written[worker_idx] = max(stream_written.get(worker_idx, 0), int(m.group(1)))
            continue
          m = STREAM_PROGRESS_RE.match(line)
          if m:
            stream_written[worker_idx] = max(stream_written.get(worker_idx, 0), int(m.group(1)))
            continue
          m = STREAM_DONE_RE.match(line)
          if m:
            stream_written[worker_idx] = max(stream_written.get(worker_idx, 0), int(m.group(1)))
            continue
          if UNSAFE_RESUME_RE.match(line):
            stream_resume_blocked.add(worker_idx)
            print(
              f"[lake] mark worker={worker_idx} blocked (unsafe resume guard). "
              "Use a fresh --data-dir or pass --stream-extra-args \"--allow-unsafe-resume\".",
              flush=True,
            )
            continue

      if train_proc.poll() is not None:
        rc = int(train_proc.returncode or 0)
        for idx in _running_workers():
          _stop_stream_worker(idx, reason="train_exit")
        print(f"[lake] training exited rc={rc}", flush=True)
        raise SystemExit(rc)

      _reap_workers()

      now = time.time()
      if (now - last_inventory_t) >= args.inventory_refresh_sec:
        protected_val_shards = _load_val_shard_protection(val_shards_file)
        inventory_records = _build_shard_inventory(data_dir, clip_count_cache)
        last_inventory_t = now

      consumed_total = current_step * args.train_batch_size * args.train_grad_accum
      written_total = sum(stream_written.values())
      available_total_clips = sum(rec[2] for rec in inventory_records)
      non_val_bytes = sum(rec[3] for rec in inventory_records if rec[1] not in protected_val_shards)
      non_val_clips = sum(rec[2] for rec in inventory_records if rec[1] not in protected_val_shards)
      reserve = max(0, available_total_clips - consumed_total)

      dt = max(1e-9, now - last_rate_t)
      if dt >= 1.0:
        consumed_delta = max(0, consumed_total - last_consumed_total)
        written_delta = max(0, written_total - last_written_total)
        train_rate_inst = consumed_delta / dt
        write_rate_inst = written_delta / dt
        train_rate_ema = _ema(train_rate_ema, train_rate_inst, args.rate_ema_alpha)
        write_rate_ema = _ema(write_rate_ema, write_rate_inst, args.rate_ema_alpha)
        running_count = max(1, len(_active_workers()))
        if write_rate_inst > 0:
          per_worker_write_rate_ema = _ema(
            per_worker_write_rate_ema,
            write_rate_inst / float(running_count),
            args.rate_ema_alpha,
          )
        last_rate_t = now
        last_consumed_total = consumed_total
        last_written_total = written_total

      free_gb = _disk_free_gb(data_dir)
      hard_free_blocked = free_gb < args.disk_min_free_gb
      lake_max_bytes = int(args.lake_max_gb * (1024.0 ** 3))
      lake_resume_bytes = int(args.lake_max_gb * args.lake_resume_fraction * (1024.0 ** 3))
      if args.lake_max_gb > 0:
        if (not lake_blocked) and non_val_bytes >= lake_max_bytes:
          lake_blocked = True
          print(
            f"[lake] pause-by-cap non_val_gb={non_val_bytes/(1024.0**3):.1f} cap_gb={args.lake_max_gb:.1f}",
            flush=True,
          )
        elif lake_blocked and non_val_bytes <= lake_resume_bytes and (not hard_free_blocked):
          lake_blocked = False
          print(
            f"[lake] resume-by-cap non_val_gb={non_val_bytes/(1024.0**3):.1f} "
            f"resume_gb={args.lake_max_gb*args.lake_resume_fraction:.1f}",
            flush=True,
          )

      if args.prune_consumed and (now - last_prune_t) >= args.prune_every_sec and inventory_records:
        should_prune = bool(lake_blocked) or (reserve > args.reserve_high_clips)
        if should_prune:
          target_non_val_bytes = non_val_bytes
          if args.lake_max_gb > 0 and lake_blocked:
            target_non_val_bytes = lake_resume_bytes
          deleted_shards, deleted_clips, deleted_bytes = _prune_old_shards(
            inventory_records,
            protected_keys=protected_val_shards,
            reserve_clips=reserve,
            reserve_floor_clips=args.reserve_low_clips,
            target_non_val_bytes=target_non_val_bytes,
            keep_recent_shards_per_stream=args.prune_keep_recent_shards,
            max_delete_shards=args.prune_max_shards,
          )
          if deleted_shards > 0:
            deleted_clips_total += deleted_clips
            deleted_bytes_total += deleted_bytes
            print(
              f"[lake] prune shards={deleted_shards} clips={_fmt_int(deleted_clips)} "
              f"bytes_gb={deleted_bytes/(1024.0**3):.2f}",
              flush=True,
            )
            protected_val_shards = _load_val_shard_protection(val_shards_file)
            inventory_records = _build_shard_inventory(data_dir, clip_count_cache)
            available_total_clips = sum(rec[2] for rec in inventory_records)
            non_val_bytes = sum(rec[3] for rec in inventory_records if rec[1] not in protected_val_shards)
            non_val_clips = sum(rec[2] for rec in inventory_records if rec[1] not in protected_val_shards)
            reserve = max(0, available_total_clips - consumed_total)
        last_prune_t = now

      running = _active_workers()
      blocked_now = bool(lake_blocked or hard_free_blocked)
      desired = _desired_workers(
        reserve=reserve,
        consumed_rate=train_rate_ema,
        write_rate=write_rate_ema,
        running_count=len(running),
      )
      if blocked_now:
        desired = 0
      _reconcile_worker_count(desired, free_gb, force_pause=blocked_now)

      if (now - last_status_t) >= args.status_every_sec:
        active_after = _active_workers()
        paused_after = _paused_alive_workers()
        print(
          "[lake] "
          f"step={current_step} written={_fmt_int(written_total)} available={_fmt_int(available_total_clips)} "
          f"consumed~={_fmt_int(consumed_total)} "
          f"reserve~={_fmt_int(reserve)} "
          f"train_rate~={train_rate_ema:.2f}/s write_rate~={write_rate_ema:.2f}/s "
          f"workers={len(active_after)}a+{len(paused_after)}p/{desired}/{args.num_streams} "
          f"non_val_gb={non_val_bytes/(1024.0**3):.1f} non_val_clips={_fmt_int(non_val_clips)} "
          f"free_gb={free_gb:.1f} blocked={int(blocked_now)}",
          flush=True,
        )
        last_status_t = now

  except KeyboardInterrupt:
    print("[lake] interrupt received; shutting down children", flush=True)
    for idx in _running_workers():
      _stop_stream_worker(idx, reason="keyboard_interrupt")
    if train_proc is not None and train_proc.poll() is None:
      try:
        train_proc.send_signal(signal.SIGINT)
      except Exception:
        pass
    raise


if __name__ == "__main__":
  main()
