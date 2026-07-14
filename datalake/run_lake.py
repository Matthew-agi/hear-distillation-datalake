#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
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
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple


TRAIN_STEP_RE = re.compile(r"\bstep=(\d+)\b")
TRAIN_RESUMED_RE = re.compile(r"^Resumed training from\s+(.+)\s+at step=(\d+)\.")
STREAM_RESUME_RE = re.compile(r"^Resume:\s+found\s+clips=(\d+)\s+bytes=(\d+)\.")
STREAM_PROGRESS_RE = re.compile(
  r"^PROGRESS written=(\d+)/(\d+) written_bytes=(\d+)/(\d+) seen=(\d+) skipped=(\d+) "
  r"errors=(\d+) rate=([0-9.]+) clips/s byte_rate=([0-9.]+) MiB/s elapsed=([0-9.]+)s"
)
STREAM_DONE_RE = re.compile(
  r"^DONE written=(\d+) written_bytes=(\d+) seen=(\d+) skipped=(\d+) "
  r"elapsed=([0-9.]+)s rate=([0-9.]+) clips/s byte_rate=([0-9.]+) MiB/s"
)
RESUME_BLOCK_RE = re.compile(r"^Refusing (?:unsafe )?resume:")
CKPT_STEP_FILENAME_RE = re.compile(r"^ckpt_(\d+)\.pt$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

GIB = 1024.0 ** 3


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
  return usage.free / GIB


def _logical_reserve_bytes(
  *,
  active_bytes: int,
  origin_active_bytes: int,
  produced_since_origin_bytes: int,
  consumed_since_origin_bytes: int,
  pruned_since_origin_bytes: int = 0,
) -> int:
  logical = max(
    0,
    int(origin_active_bytes)
    + int(produced_since_origin_bytes)
    - int(consumed_since_origin_bytes)
    - int(pruned_since_origin_bytes),
  )
  return min(max(0, int(active_bytes)), logical)


def _prune_target_bytes(*, active_bytes: int, reserve_bytes: int, reserve_low_bytes: int) -> int:
  surplus = max(0, int(reserve_bytes) - int(reserve_low_bytes))
  return max(int(reserve_low_bytes), int(active_bytes) - surplus)


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


def _load_json(path: Path) -> Dict:
  if not path.exists():
    return {}
  try:
    payload = json.loads(path.read_text(encoding="utf-8"))
  except Exception:
    return {}
  return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: Dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  os.replace(tmp, path)


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
  stem = path.stem
  if stem.startswith("shard-"):
    try:
      shard_idx = int(stem.split("-")[-1])
    except Exception:
      shard_idx = -1
  return stream_idx, shard_idx


def _discover_final_shards(data_dir: Path) -> List[Path]:
  if not data_dir.exists():
    return []
  shards = sorted(data_dir.glob("stream-*/shard-*.tar"))
  shards.extend(sorted(data_dir.glob("shard-*.tar")))
  dedup: Dict[str, Path] = {}
  for p in shards:
    dedup[str(p.resolve())] = p
  out = list(dedup.values())
  out.sort(key=lambda p: (_parse_stream_shard(p)[0], _parse_stream_shard(p)[1], str(p)))
  return out


def _build_area_inventory(data_dir: Path) -> List[Tuple[Path, str, int, int, int, int]]:
  records: List[Tuple[Path, str, int, int, int, int]] = []
  for p in _discover_final_shards(data_dir):
    try:
      st = p.stat()
    except FileNotFoundError:
      continue
    except OSError:
      continue
    key = str(p.resolve())
    sidx, shidx = _parse_stream_shard(p)
    records.append((p, key, int(st.st_size), int(st.st_mtime_ns), sidx, shidx))
  records.sort(key=lambda r: (r[3], r[4], r[5], r[1]))
  return records


def _prune_old_shards(
  records: List[Tuple[Path, str, int, int, int, int]],
  *,
  target_total_bytes: int,
  keep_recent_shards_per_stream: int,
  max_delete_shards: int,
) -> Tuple[int, int]:
  if not records or max_delete_shards <= 0:
    return 0, 0

  keep_keys: Set[str] = set()
  if keep_recent_shards_per_stream > 0:
    by_stream: Dict[int, List[Tuple[Path, str, int, int, int, int]]] = {}
    for rec in records:
      by_stream.setdefault(rec[4], []).append(rec)
    for items in by_stream.values():
      for rec in items[-keep_recent_shards_per_stream:]:
        keep_keys.add(rec[1])

  current_total_bytes = sum(rec[2] for rec in records)
  deleted_shards = 0
  deleted_bytes = 0
  for path, key, nbytes, _mtime_ns, _sidx, _shidx in records:
    if deleted_shards >= max_delete_shards:
      break
    if key in keep_keys:
      continue
    if current_total_bytes <= target_total_bytes:
      break
    try:
      path.unlink()
    except FileNotFoundError:
      continue
    except OSError:
      continue
    current_total_bytes = max(0, current_total_bytes - nbytes)
    deleted_shards += 1
    deleted_bytes += nbytes
  return deleted_shards, deleted_bytes


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


def _load_stream_progress(data_dir: Path, *, num_streams: int, exact: bool) -> Tuple[Dict[int, int], Dict[int, int]]:
  clips_out: Dict[int, int] = {}
  bytes_out: Dict[int, int] = {}
  for idx in range(num_streams):
    stream_dir = data_dir / f"stream-{idx:03d}" if num_streams > 1 else data_dir
    manifest = _load_json(stream_dir / "manifest.json")
    resume_state = _load_json(stream_dir / "resume_state.json")
    manifest_clips = int(manifest.get("num_clips", 0) or 0)
    manifest_bytes = int(manifest.get("num_bytes", 0) or 0)
    state_clips = int(resume_state.get("written", 0) or 0)
    state_bytes = int(resume_state.get("written_bytes", 0) or 0)
    clips = max(manifest_clips, state_clips)
    nbytes = max(manifest_bytes, state_bytes)
    if exact and stream_dir.exists():
      existing_clips = 0
      existing_bytes = 0
      for tar_path in sorted(stream_dir.glob("shard-*.tar")):
        try:
          existing_bytes += int(tar_path.stat().st_size)
        except Exception:
          pass
        existing_clips += _count_wavs_in_tar(tar_path)
      clips = max(clips, existing_clips)
      nbytes = max(nbytes, existing_bytes)
    clips_out[idx] = clips
    bytes_out[idx] = nbytes
  return clips_out, bytes_out


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


def _checkpoint_optimizer_lr(path: Path) -> Optional[float]:
  try:
    import torch

    try:
      ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
      ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
      return None
    optim_state = ckpt.get("optim")
    if not isinstance(optim_state, dict):
      return None
    param_groups = optim_state.get("param_groups")
    if not isinstance(param_groups, list):
      return None
    for group in param_groups:
      if not isinstance(group, dict):
        continue
      lr = group.get("lr")
      if lr is None:
        continue
      try:
        lr_f = float(lr)
      except Exception:
        continue
      if math.isfinite(lr_f) and lr_f >= 0.0:
        return lr_f
  except Exception:
    return None
  return None


def _checkpoint_trainer_defaults(path: Path) -> Dict[str, int]:
  defaults: Dict[str, int] = {}
  try:
    import torch

    try:
      ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
      ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
      return defaults
    arg_state = ckpt.get("args")
    if isinstance(arg_state, dict):
      for src_key, dst_key in (
        ("batch_size", "batch_size"),
        ("grad_accum", "grad_accum"),
        ("num_workers", "num_workers"),
      ):
        raw_value = arg_state.get(src_key)
        if raw_value is None:
          continue
        try:
          value = int(raw_value)
        except Exception:
          continue
        if value > 0:
          defaults[dst_key] = value
    auto_warmup_state = ckpt.get("auto_warmup_state")
    if isinstance(auto_warmup_state, dict):
      for key in ("handoff_batch_size", "current_batch_size"):
        raw_value = auto_warmup_state.get(key)
        if raw_value is None:
          continue
        try:
          value = int(raw_value)
        except Exception:
          continue
        if value > 0:
          defaults["batch_size"] = value
          break
  except Exception:
    return defaults
  return defaults


def _token_has_flag(tokens: Sequence[str], flag: str) -> bool:
  prefix = f"{flag}="
  return any(tok == flag or tok.startswith(prefix) for tok in tokens)


def _reject_managed_flags(tokens: Sequence[str], *, source: str, flags: Sequence[str]) -> None:
  bad = [flag for flag in flags if _token_has_flag(tokens, flag)]
  if bad:
    joined = ", ".join(bad)
    _die(f"{source} cannot set {joined}; those are managed by run_lake.py.")


def _reject_managed_prefixes(tokens: Sequence[str], *, source: str, prefixes: Sequence[str]) -> None:
  bad: List[str] = []
  for tok in tokens:
    key = tok.split("=", 1)[0]
    for prefix in prefixes:
      if key == prefix or key.startswith(prefix):
        bad.append(key)
        break
  if bad:
    joined = ", ".join(sorted(set(bad)))
    _die(f"{source} cannot set {joined}; those are managed by run_lake.py.")


def _inherit_decay_tokens(train_extra_tokens: Sequence[str]) -> List[str]:
  inherited: List[str] = []
  skip_value_flags = {
    "--data-dir",
    "--out",
    "--resume-from",
    "--max-steps",
    "--val-manifest",
    "--lr",
    "--lr-schedule",
    "--lr-schedule-start-step",
    "--lr-warmup-steps",
  }
  skip_bool_flags = {
    "--resume-latest",
    "--auto-warmup",
  }
  i = 0
  while i < len(train_extra_tokens):
    tok = train_extra_tokens[i]
    key = tok.split("=", 1)[0]
    if key.startswith("--auto-warmup-"):
      i += 1 if "=" in tok else 2
      continue
    if key in skip_bool_flags:
      i += 1
      continue
    if key in skip_value_flags:
      i += 1 if "=" in tok else 2
      continue
    inherited.append(tok)
    i += 1
  return inherited


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
  stream_written_bytes: Dict[int, int],
  running_workers: Set[int],
  max_workers: int,
) -> List[int]:
  candidates = [i for i in range(max_workers) if i not in running_workers]
  return sorted(candidates, key=lambda i: (stream_written_bytes.get(i, 0), i))


def _estimated_clip_bytes(clip_seconds: float, sample_rate: int) -> int:
  wav_bytes = int(round(float(clip_seconds) * float(sample_rate) * 2.0))
  return max(4096, wav_bytes + 4096)


def _parse_train_val_settings(tokens: List[str]) -> Tuple[float, int]:
  val_fraction = 0.001
  val_target_clips = 0
  i = 0
  while i < len(tokens):
    tok = tokens[i]
    if tok.startswith("--val-fraction="):
      try:
        val_fraction = float(tok.split("=", 1)[1])
      except Exception:
        pass
      i += 1
      continue
    if tok == "--val-fraction" and i + 1 < len(tokens):
      try:
        val_fraction = float(tokens[i + 1])
      except Exception:
        pass
      i += 2
      continue
    if tok.startswith("--val-target-clips="):
      try:
        val_target_clips = int(tok.split("=", 1)[1])
      except Exception:
        pass
      i += 1
      continue
    if tok == "--val-target-clips" and i + 1 < len(tokens):
      try:
        val_target_clips = int(tokens[i + 1])
      except Exception:
        pass
      i += 2
      continue
    i += 1
  return val_fraction, val_target_clips


def _parse_train_max_steps(tokens: Sequence[str], *, default: int) -> int:
  max_steps = int(default)
  i = 0
  while i < len(tokens):
    tok = tokens[i]
    if tok.startswith("--max-steps="):
      try:
        max_steps = int(tok.split("=", 1)[1])
      except Exception:
        pass
      i += 1
      continue
    if tok == "--max-steps" and i + 1 < len(tokens):
      try:
        max_steps = int(tokens[i + 1])
      except Exception:
        pass
      i += 2
      continue
    i += 1
  return max_steps


def _stable_score(tag: str, clip_id: str, seed: int) -> float:
  digest = hashlib.sha1(f"{tag}:{seed}:{clip_id}".encode("utf-8")).digest()
  return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _iter_tar_records(tar_path: Path) -> Iterator[Tuple[str, bytes, bytes]]:
  try:
    tf = tarfile.open(tar_path, mode="r")
  except FileNotFoundError:
    return
  except tarfile.TarError:
    return
  try:
    with tf:
      pending: Dict[str, Dict[str, bytes]] = {}
      for member in tf:
        if not member.isfile():
          continue
        name = Path(member.name).name
        stem, ext = os.path.splitext(name)
        if ext not in (".wav", ".json"):
          continue
        extracted = tf.extractfile(member)
        if extracted is None:
          continue
        try:
          data = extracted.read()
        except Exception:
          return
        entry = pending.setdefault(stem, {})
        entry[ext] = data
        if ".wav" in entry and ".json" in entry:
          yield stem, entry[".wav"], entry[".json"]
          pending.pop(stem, None)
  except tarfile.TarError:
    return


def _extract_entry_from_tar(tar_path: Path, stem: str) -> Optional[Tuple[bytes, bytes]]:
  wav_name = f"{stem}.wav"
  json_name = f"{stem}.json"
  wav_bytes: Optional[bytes] = None
  json_bytes: Optional[bytes] = None
  try:
    with tarfile.open(tar_path, mode="r") as tf:
      for member in tf:
        if not member.isfile():
          continue
        name = Path(member.name).name
        if name not in (wav_name, json_name):
          continue
        extracted = tf.extractfile(member)
        if extracted is None:
          continue
        data = extracted.read()
        if name == wav_name:
          wav_bytes = data
        else:
          json_bytes = data
        if wav_bytes is not None and json_bytes is not None:
          return wav_bytes, json_bytes
  except Exception:
    return None
  return None


def _write_tar_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
  info = tarfile.TarInfo(name=name)
  info.size = len(data)
  info.mtime = int(time.time())
  tar.addfile(info, io.BytesIO(data))


class _TarShardWriter:
  def __init__(self, out_dir: Path, shard_idx: int) -> None:
    self.out_dir = out_dir
    self.shard_idx = int(shard_idx)
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.final_path = self.out_dir / f"shard-{self.shard_idx:06d}.tar"
    self.tmp_path = self.out_dir / f"shard-{self.shard_idx:06d}.tar.tmp"
    if self.tmp_path.exists():
      try:
        self.tmp_path.unlink()
      except Exception:
        pass
    self.tar = tarfile.open(self.tmp_path, mode="w")
    self.count = 0

  def write(self, stem: str, wav_bytes: bytes, json_bytes: bytes) -> None:
    _write_tar_member(self.tar, f"{stem}.wav", wav_bytes)
    _write_tar_member(self.tar, f"{stem}.json", json_bytes)
    self.count += 1

  def finalize(self) -> Tuple[Optional[Path], int, int]:
    try:
      self.tar.close()
    finally:
      if self.count > 0:
        os.replace(self.tmp_path, self.final_path)
        try:
          nbytes = int(self.final_path.stat().st_size)
        except Exception:
          nbytes = 0
        return self.final_path, nbytes, self.count
      try:
        self.tmp_path.unlink()
      except Exception:
        pass
    return None, 0, 0


def _sum_entry_bytes(entries: Sequence[Dict]) -> int:
  total = 0
  for entry in entries:
    try:
      total += int(entry.get("nbytes", 0))
    except Exception:
      continue
  return total


def _max_score_index(entries: Sequence[Dict]) -> int:
  best_idx = 0
  best_score = float(entries[0].get("score", 0.0))
  for idx in range(1, len(entries)):
    score = float(entries[idx].get("score", 0.0))
    if score > best_score:
      best_idx = idx
      best_score = score
  return best_idx


def _normalize_active_entries(entries: Sequence[Dict]) -> List[Dict]:
  out: List[Dict] = []
  for entry in entries:
    if not isinstance(entry, dict):
      continue
    tar_path_raw = entry.get("tar_path")
    stem = entry.get("stem")
    if not isinstance(tar_path_raw, str) or not isinstance(stem, str) or not stem:
      continue
    tar_path = Path(tar_path_raw)
    if not tar_path.exists():
      continue
    try:
      nbytes = int(entry.get("nbytes", 0))
    except Exception:
      nbytes = 0
    try:
      score = float(entry.get("score", 0.0))
    except Exception:
      score = 0.0
    out.append(
      {
        "tar_path": str(tar_path.resolve()),
        "stem": stem,
        "nbytes": max(0, nbytes),
        "score": score,
      }
    )
  return out


def _manifest_payload(entries: Sequence[Dict], *, max_bytes: int) -> Dict:
  active_entries = list(entries)
  return {
    "updated_unix": int(time.time()),
    "entry_count": len(active_entries),
    "active_bytes": _sum_entry_bytes(active_entries),
    "max_bytes": int(max_bytes),
    "entries": active_entries,
  }


def _next_shard_index(area_dir: Path) -> int:
  next_idx = 0
  for shard in _discover_final_shards(area_dir):
    _stream_idx, shard_idx = _parse_stream_shard(shard)
    if shard_idx >= 0:
      next_idx = max(next_idx, shard_idx + 1)
  return next_idx


def _next_train_shard_indices(train_lake_dir: Path, *, num_streams: int) -> Dict[str, int]:
  out: Dict[str, int] = {}
  for idx in range(num_streams):
    stream_dir = train_lake_dir / f"stream-{idx:03d}" if num_streams > 1 else train_lake_dir
    out[str(idx)] = _next_shard_index(stream_dir)
  return out


def _load_curation_state(
  path: Path,
  *,
  train_lake_dir: Path,
  val_lake_dir: Path,
  decay_lake_dir: Path,
  num_streams: int,
  val_capacity_entries: int,
  decay_max_entries: int,
) -> Dict:
  raw = _load_json(path)
  state = raw if isinstance(raw, dict) else {}
  train_next = state.get("train_next_shard_by_stream")
  if not isinstance(train_next, dict):
    train_next = _next_train_shard_indices(train_lake_dir, num_streams=num_streams)
  for idx, next_idx in _next_train_shard_indices(train_lake_dir, num_streams=num_streams).items():
    try:
      train_next[idx] = max(int(train_next.get(idx, 0)), int(next_idx))
    except Exception:
      train_next[idx] = int(next_idx)
  state["train_next_shard_by_stream"] = train_next
  try:
    state["val_next_shard"] = max(int(state.get("val_next_shard", 0)), _next_shard_index(val_lake_dir))
  except Exception:
    state["val_next_shard"] = _next_shard_index(val_lake_dir)
  try:
    state["decay_next_shard"] = max(int(state.get("decay_next_shard", 0)), _next_shard_index(decay_lake_dir))
  except Exception:
    state["decay_next_shard"] = _next_shard_index(decay_lake_dir)
  state["val_capacity_entries"] = int(max(0, val_capacity_entries))
  state["decay_max_entries"] = int(max(0, decay_max_entries))
  state["val_active"] = _normalize_active_entries(state.get("val_active", []))
  state["decay_active"] = _normalize_active_entries(state.get("decay_active", []))
  for key in (
    "train_committed_clips",
    "train_committed_member_bytes",
    "train_committed_tar_bytes",
    "val_seen_candidates",
    "val_replacements",
    "val_replacements_since_compaction",
    "val_compactions",
    "decay_seen_candidates",
    "decay_replacements",
    "decay_replacements_since_compaction",
    "decay_compactions",
  ):
    try:
      state[key] = int(state.get(key, 0))
    except Exception:
      state[key] = 0
  state["updated_unix"] = int(time.time())
  return state


def _write_curation_state(path: Path, state: Dict) -> None:
  payload = dict(state)
  payload["updated_unix"] = int(time.time())
  _write_json_atomic(path, payload)


def _compact_active_entries(
  *,
  store_dir: Path,
  active_entries: Sequence[Dict],
  next_shard_idx: int,
  shard_size: int,
) -> Tuple[List[Dict], int]:
  active_entries = list(active_entries)
  if not active_entries:
    for shard in _discover_final_shards(store_dir):
      try:
        shard.unlink()
      except Exception:
        pass
    return [], int(next_shard_idx)

  new_entries: List[Dict] = []
  writer: Optional[_TarShardWriter] = None
  new_paths: Set[str] = set()
  current_idx = int(next_shard_idx)
  for entry in active_entries:
    tar_path = Path(str(entry["tar_path"]))
    stem = str(entry["stem"])
    extracted = _extract_entry_from_tar(tar_path, stem)
    if extracted is None:
      continue
    wav_bytes, json_bytes = extracted
    if writer is None:
      writer = _TarShardWriter(store_dir, current_idx)
      current_idx += 1
    writer.write(stem, wav_bytes, json_bytes)
    new_entries.append(
      {
        "tar_path": str(writer.final_path.resolve()),
        "stem": stem,
        "nbytes": int(entry.get("nbytes", len(wav_bytes) + len(json_bytes))),
        "score": float(entry.get("score", 0.0)),
      }
    )
    if writer.count >= shard_size:
      final_path, _nbytes, _count = writer.finalize()
      if final_path is not None:
        new_paths.add(str(final_path.resolve()))
      writer = None
  if writer is not None:
    final_path, _nbytes, _count = writer.finalize()
    if final_path is not None:
      new_paths.add(str(final_path.resolve()))

  for shard in _discover_final_shards(store_dir):
    if str(shard.resolve()) not in new_paths:
      try:
        shard.unlink()
      except Exception:
        pass
  return new_entries, current_idx


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Run streaming + training as a bounded adaptive curated data lake pipeline.")
  ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
  ap.add_argument("--env-file", type=Path, default=Path(".env"), help="Optional env file to load before launching workers.")
  ap.add_argument("--data-dir", type=Path, default=Path("data/laion_audio_lake"))
  ap.add_argument("--incoming-dir", type=Path, default=None, help="Raw ingest area (default: <data-dir>/incoming).")
  ap.add_argument("--train-lake-dir", type=Path, default=None, help="Curated train lake (default: <data-dir>/train).")
  ap.add_argument("--val-lake-dir", type=Path, default=None, help="Curated validation store (default: <data-dir>/val).")
  ap.add_argument("--decay-lake-dir", type=Path, default=None, help="Curated decay store (default: <data-dir>/decay).")
  ap.add_argument("--train-script", type=Path, default=Path("distill_hear_vit_s_canon2d.py"))
  ap.add_argument("--stream-script", type=Path, default=Path("stream_laion_audio_clips.py"))
  ap.add_argument("--python", type=str, default="python3")

  ap.add_argument("--num-streams", type=int, default=3, help="Maximum stream workers.")
  ap.add_argument("--min-streams", type=int, default=1, help="Minimum stream workers when streaming is needed.")
  ap.add_argument("--auto-tune-streams", action=argparse.BooleanOptionalAction, default=True, help="Adapt active stream workers to training throughput.")
  ap.add_argument("--stream-headroom", type=float, default=1.10, help="Target write/consume ratio when auto tuning.")
  ap.add_argument("--rate-ema-alpha", type=float, default=0.25, help="EMA alpha for throughput estimates.")
  ap.add_argument("--chunk-gb-per-stream", type=float, default=0.0, help="Per-worker chunk target in GiB.")
  ap.add_argument("--reserve-low-gb", type=float, default=0.0, help="If train reserve falls below this, grow streaming.")
  ap.add_argument("--reserve-high-gb", type=float, default=0.0, help="If train reserve exceeds this, retire optional workers and prune.")
  ap.add_argument("--chunk-clips-per-stream", type=int, default=20000, help="Deprecated fallback for chunk target when --chunk-gb-per-stream is unset.")
  ap.add_argument("--reserve-low-clips", type=int, default=150000, help="Deprecated fallback when --reserve-low-gb is unset.")
  ap.add_argument("--reserve-high-clips", type=int, default=450000, help="Deprecated fallback when --reserve-high-gb is unset.")
  ap.add_argument("--disk-min-free-gb", type=float, default=20.0)
  ap.add_argument("--lake-max-gb", type=float, default=0.0, help="Max curated train lake size in GiB (0 disables cap).")
  ap.add_argument(
    "--lake-resume-fraction",
    type=float,
    default=0.5,
    help="Resume scheduling new chunks when train lake drops below this fraction of --lake-max-gb.",
  )
  ap.add_argument("--inventory-refresh-sec", type=float, default=10.0, help="Seconds between curated train inventory refreshes.")
  ap.add_argument(
    "--prune-consumed",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Delete old curated train shards while respecting the reserve floor.",
  )
  ap.add_argument("--prune-every-sec", type=float, default=15.0, help="Seconds between prune checks.")
  ap.add_argument("--prune-max-shards", type=int, default=16, help="Max train shards deleted per prune cycle.")
  ap.add_argument("--prune-keep-recent-shards", type=int, default=1, help="Keep this many latest train shards per stream.")
  ap.add_argument("--retire-workers-at-chunk-boundary", action=argparse.BooleanOptionalAction, default=True, help="Let optional workers finish their current chunk before retirement.")
  ap.add_argument("--status-every-sec", type=float, default=20.0)
  ap.add_argument("--poll-sec", type=float, default=1.0)
  ap.add_argument("--max-stream-failures", type=int, default=10)
  ap.add_argument("--exact-start-count", action=argparse.BooleanOptionalAction, default=True)

  ap.add_argument("--curation-state-file", type=Path, default=None, help="Persistent curation state JSON.")
  ap.add_argument("--val-manifest", type=Path, default=None, help="Validation manifest path (default: <data-dir>/manifests/val_active.json).")
  ap.add_argument("--decay-manifest", type=Path, default=None, help="Decay manifest path (default: <data-dir>/manifests/decay_active.json).")
  ap.add_argument("--curation-shard-size", type=int, default=0, help="Curated shard size (<=0 reuses --shard-size).")
  ap.add_argument("--val-max-gb", type=float, default=0.0, help="Validation reservoir cap in GiB (<=0 derives from current validation config).")
  ap.add_argument("--decay-retain-ratio", type=float, default=0.10, help="Approximate retain ratio for decay sampling.")
  ap.add_argument("--decay-max-gb", type=float, default=0.0, help="Decay reservoir cap in GiB (<=0 uses reserve high water mark).")
  ap.add_argument("--curation-compaction-factor", type=float, default=2.0, help="Compact val/decay stores when physical bytes exceed this multiple of active bytes.")
  ap.add_argument("--curation-compaction-min-replacements", type=int, default=64, help="Minimum replacements before compaction is eligible.")

  ap.add_argument("--clip-seconds", type=float, default=2.0)
  ap.add_argument("--sample-rate", type=int, default=16000)
  ap.add_argument("--shuffle-buffer", type=int, default=2000)
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
  ap.add_argument("--decay-steps", type=int, default=-1, help="Decay phase steps: 0 disables, >0 uses an explicit length, <0 uses --decay-fraction of completed stable training.")
  ap.add_argument("--decay-fraction", type=float, default=0.10, help="Default decay length as a fraction of completed stable training when --decay-steps < 0.")
  ap.add_argument("--decay-out", type=Path, default=None, help="Decay phase output dir (default: <train-out>/decay_phase).")
  ap.add_argument("--decay-extra-args", type=str, default="", help="Additional trainer args for the decay phase.")
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  if args.num_streams <= 0:
    _die("--num-streams must be > 0")
  if args.min_streams < 0 or args.min_streams > args.num_streams:
    _die("--min-streams must be in [0, --num-streams]")
  if args.stream_headroom <= 0:
    _die("--stream-headroom must be > 0")
  if not (0.0 < args.rate_ema_alpha <= 1.0):
    _die("--rate-ema-alpha must be in (0, 1]")
  if args.chunk_clips_per_stream <= 0:
    _die("--chunk-clips-per-stream must be > 0")
  if args.reserve_low_clips < 0 or args.reserve_high_clips <= 0:
    _die("Legacy reserve thresholds must be positive.")
  if args.reserve_low_clips >= args.reserve_high_clips:
    _die("--reserve-low-clips must be < --reserve-high-clips")
  if args.reserve_low_gb < 0 or args.reserve_high_gb < 0:
    _die("--reserve-low-gb and --reserve-high-gb must be >= 0")
  if args.reserve_high_gb > 0 and args.reserve_low_gb >= args.reserve_high_gb:
    _die("--reserve-low-gb must be < --reserve-high-gb")
  if args.chunk_gb_per_stream < 0:
    _die("--chunk-gb-per-stream must be >= 0")
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
  if args.status_every_sec <= 0 or args.poll_sec <= 0:
    _die("--status-every-sec and --poll-sec must be > 0")
  if args.max_stream_failures <= 0:
    _die("--max-stream-failures must be > 0")
  if args.curation_shard_size < 0:
    _die("--curation-shard-size must be >= 0")
  if args.val_max_gb < 0 or args.decay_max_gb < 0:
    _die("--val-max-gb and --decay-max-gb must be >= 0")
  if not (0.0 <= args.decay_retain_ratio <= 1.0):
    _die("--decay-retain-ratio must be in [0, 1]")
  if not (0.0 < args.decay_fraction <= 1.0):
    _die("--decay-fraction must be in (0, 1]")
  if args.curation_compaction_factor < 1.0:
    _die("--curation-compaction-factor must be >= 1.0")
  if args.curation_compaction_min_replacements < 0:
    _die("--curation-compaction-min-replacements must be >= 0")
  if args.decay_steps < -1:
    _die("--decay-steps must be >= -1")

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

  data_root = (repo_root / args.data_dir).resolve() if not args.data_dir.is_absolute() else args.data_dir.resolve()
  incoming_dir = (
    (repo_root / args.incoming_dir).resolve()
    if args.incoming_dir is not None and not args.incoming_dir.is_absolute()
    else (args.incoming_dir.resolve() if args.incoming_dir is not None else (data_root / "incoming").resolve())
  )
  train_lake_dir = (
    (repo_root / args.train_lake_dir).resolve()
    if args.train_lake_dir is not None and not args.train_lake_dir.is_absolute()
    else (args.train_lake_dir.resolve() if args.train_lake_dir is not None else (data_root / "train").resolve())
  )
  val_lake_dir = (
    (repo_root / args.val_lake_dir).resolve()
    if args.val_lake_dir is not None and not args.val_lake_dir.is_absolute()
    else (args.val_lake_dir.resolve() if args.val_lake_dir is not None else (data_root / "val").resolve())
  )
  decay_lake_dir = (
    (repo_root / args.decay_lake_dir).resolve()
    if args.decay_lake_dir is not None and not args.decay_lake_dir.is_absolute()
    else (args.decay_lake_dir.resolve() if args.decay_lake_dir is not None else (data_root / "decay").resolve())
  )
  manifests_dir = data_root / "manifests"
  curation_state_file = args.curation_state_file
  if curation_state_file is None:
    curation_state_file = manifests_dir / "curation_state.json"
  elif not curation_state_file.is_absolute():
    curation_state_file = (repo_root / curation_state_file).resolve()
  else:
    curation_state_file = curation_state_file.resolve()
  val_manifest_path = args.val_manifest
  if val_manifest_path is None:
    val_manifest_path = manifests_dir / "val_active.json"
  elif not val_manifest_path.is_absolute():
    val_manifest_path = (repo_root / val_manifest_path).resolve()
  else:
    val_manifest_path = val_manifest_path.resolve()
  decay_manifest_path = args.decay_manifest
  if decay_manifest_path is None:
    decay_manifest_path = manifests_dir / "decay_active.json"
  elif not decay_manifest_path.is_absolute():
    decay_manifest_path = (repo_root / decay_manifest_path).resolve()
  else:
    decay_manifest_path = decay_manifest_path.resolve()

  train_script = (repo_root / args.train_script).resolve() if not args.train_script.is_absolute() else args.train_script.resolve()
  stream_script = (repo_root / args.stream_script).resolve() if not args.stream_script.is_absolute() else args.stream_script.resolve()
  train_out = (repo_root / args.train_out).resolve() if not args.train_out.is_absolute() else args.train_out.resolve()
  decay_out = args.decay_out
  if decay_out is None:
    decay_out = (train_out / "decay_phase").resolve()
  elif not decay_out.is_absolute():
    decay_out = (repo_root / decay_out).resolve()
  else:
    decay_out = decay_out.resolve()
  if not train_script.exists():
    _die(f"Training script not found: {train_script}")
  if not stream_script.exists():
    _die(f"Streaming script not found: {stream_script}")

  mkdir_paths = [data_root, incoming_dir, train_lake_dir, val_lake_dir, decay_lake_dir, manifests_dir, train_out]
  if args.decay_steps != 0:
    mkdir_paths.append(decay_out)
  for path in mkdir_paths:
    path.mkdir(parents=True, exist_ok=True)

  train_extra_tokens: List[str] = []
  if args.train_extra_args:
    try:
      train_extra_tokens = shlex.split(args.train_extra_args)
    except ValueError as exc:
      _die(f"Failed to parse --train-extra-args: {exc}")
  if train_extra_tokens:
    _reject_managed_flags(
      train_extra_tokens,
      source="--train-extra-args",
      flags=("--data-dir", "--out", "--val-manifest"),
    )
  decay_extra_tokens: List[str] = []
  if args.decay_extra_args:
    try:
      decay_extra_tokens = shlex.split(args.decay_extra_args)
    except ValueError as exc:
      _die(f"Failed to parse --decay-extra-args: {exc}")
  if decay_extra_tokens:
    _reject_managed_flags(
      decay_extra_tokens,
      source="--decay-extra-args",
      flags=(
        "--data-dir",
        "--out",
        "--resume-from",
        "--resume-latest",
        "--val-manifest",
        "--max-steps",
        "--lr",
        "--lr-schedule",
        "--lr-schedule-start-step",
      ),
    )
    _reject_managed_prefixes(
      decay_extra_tokens,
      source="--decay-extra-args",
      prefixes=("--auto-warmup",),
    )
  inherited_decay_tokens = _inherit_decay_tokens(train_extra_tokens)
  effective_decay_extra_tokens = list(inherited_decay_tokens)
  if decay_extra_tokens:
    effective_decay_extra_tokens.extend(decay_extra_tokens)

  resume_ckpt_path = _resolve_resume_checkpoint(
    repo_root=repo_root,
    train_out=train_out,
    train_extra_tokens=train_extra_tokens,
  )
  initial_step = 0
  if resume_ckpt_path is not None:
    initial_step = max(0, _checkpoint_step_hint(resume_ckpt_path))
    print(f"[lake] resume checkpoint={resume_ckpt_path} step_hint={initial_step}", flush=True)
  stable_target_max_steps = _parse_train_max_steps(train_extra_tokens, default=20000)

  estimated_clip_bytes = _estimated_clip_bytes(args.clip_seconds, args.sample_rate)
  reserve_low_bytes = int(args.reserve_low_gb * GIB) if args.reserve_low_gb > 0 else int(args.reserve_low_clips * estimated_clip_bytes)
  reserve_high_bytes = int(args.reserve_high_gb * GIB) if args.reserve_high_gb > 0 else int(args.reserve_high_clips * estimated_clip_bytes)
  if reserve_low_bytes >= reserve_high_bytes:
    _die("Resolved reserve low water mark must be < reserve high water mark.")
  chunk_bytes_per_stream = int(args.chunk_gb_per_stream * GIB) if args.chunk_gb_per_stream > 0 else int(args.chunk_clips_per_stream * estimated_clip_bytes)
  if chunk_bytes_per_stream <= 0:
    _die("Resolved chunk target must be > 0.")
  curation_shard_size = int(args.curation_shard_size) if args.curation_shard_size > 0 else int(args.shard_size)

  parsed_val_fraction, parsed_val_target_clips = _parse_train_val_settings(train_extra_tokens)
  val_base_bytes = max(reserve_high_bytes, int(args.lake_max_gb * GIB) if args.lake_max_gb > 0 else reserve_high_bytes)
  if args.val_max_gb > 0:
    val_max_bytes = int(args.val_max_gb * GIB)
  elif parsed_val_target_clips > 0:
    val_max_bytes = int(parsed_val_target_clips * estimated_clip_bytes)
  elif parsed_val_fraction > 0:
    val_max_bytes = max(estimated_clip_bytes, int(parsed_val_fraction * val_base_bytes))
  else:
    val_max_bytes = 0
  if parsed_val_fraction > 0:
    val_sample_ratio = min(1.0, max(0.0, float(parsed_val_fraction)))
  elif val_max_bytes > 0 and val_base_bytes > 0:
    val_sample_ratio = min(1.0, max(0.0, float(val_max_bytes) / float(val_base_bytes)))
  else:
    val_sample_ratio = 0.0
  decay_max_bytes = int(args.decay_max_gb * GIB) if args.decay_max_gb > 0 else max(reserve_high_bytes, chunk_bytes_per_stream)
  val_capacity_entries = int(val_max_bytes // estimated_clip_bytes) if val_max_bytes > 0 else 0
  decay_max_entries = int(decay_max_bytes // estimated_clip_bytes) if decay_max_bytes > 0 else 0

  curation_state = _load_curation_state(
    curation_state_file,
    train_lake_dir=train_lake_dir,
    val_lake_dir=val_lake_dir,
    decay_lake_dir=decay_lake_dir,
    num_streams=args.num_streams,
    val_capacity_entries=val_capacity_entries,
    decay_max_entries=decay_max_entries,
  )
  _write_json_atomic(val_manifest_path, _manifest_payload(curation_state["val_active"], max_bytes=val_max_bytes))
  _write_json_atomic(decay_manifest_path, _manifest_payload(curation_state["decay_active"], max_bytes=decay_max_bytes))
  _write_curation_state(curation_state_file, curation_state)

  print(
    "[lake] bytes-first control "
    f"reserve_low_gb={reserve_low_bytes/GIB:.2f} reserve_high_gb={reserve_high_bytes/GIB:.2f} "
    f"chunk_gb_per_stream={chunk_bytes_per_stream/GIB:.2f} "
    f"val_max_gb={val_max_bytes/GIB:.2f} val_sample_ratio={val_sample_ratio:.4f} "
    f"decay_max_gb={decay_max_bytes/GIB:.2f}",
    flush=True,
  )

  def _build_train_cmd(
    *,
    data_dir: Path,
    out_dir: Path,
    batch_size: int,
    grad_accum: int,
    num_workers: int,
    extra_tokens: Sequence[str],
    resume_from: Optional[Path],
    max_steps_override: Optional[int],
    live_refresh: bool,
  ) -> List[str]:
    cmd = [
      args.python,
      str(train_script),
      "--data-dir",
      str(data_dir),
      "--out",
      str(out_dir),
      "--batch-size",
      str(int(batch_size)),
      "--grad-accum",
      str(int(grad_accum)),
      "--num-workers",
      str(int(num_workers)),
    ]
    if live_refresh:
      cmd.extend(["--live-shard-refresh", "--shard-refresh-sec", str(args.shard_refresh_sec)])
    if resume_from is not None:
      cmd.extend(["--resume-from", str(resume_from)])
    if max_steps_override is not None:
      cmd.extend(["--max-steps", str(int(max_steps_override))])
    if extra_tokens:
      cmd.extend(extra_tokens)
    cmd.extend(["--val-fraction", "0", "--val-target-clips", "0"])
    if val_capacity_entries > 0:
      cmd.extend(["--val-manifest", str(val_manifest_path)])
      if live_refresh:
        cmd.extend(["--val-live-refresh", "--val-shard-refresh-sec", str(args.shard_refresh_sec)])
    return cmd

  stable_train_cmd = _build_train_cmd(
    data_dir=train_lake_dir,
    out_dir=train_out,
    batch_size=args.train_batch_size,
    grad_accum=args.train_grad_accum,
    num_workers=args.train_num_workers,
    extra_tokens=train_extra_tokens,
    resume_from=None,
    max_steps_override=None,
    live_refresh=True,
  )

  print("[lake] counting existing incoming stream progress (startup)", flush=True)
  stream_written_clips, stream_written_bytes = _load_stream_progress(incoming_dir, num_streams=args.num_streams, exact=args.exact_start_count)

  q: "queue.Queue[Tuple[str, str]]" = queue.Queue()
  current_step = int(initial_step)
  resume_guard_active = bool(resume_ckpt_path is not None and current_step > 0)
  trainer_resume_confirmed = False
  train_proc: Optional[subprocess.Popen] = None
  stream_workers: Dict[int, subprocess.Popen] = {}
  draining_workers: Set[int] = set()
  stream_stop_requested: Set[int] = set()
  stream_resume_blocked: Set[int] = set()
  worker_target_bytes: Dict[int, int] = {}
  stream_failures = 0

  train_rate_ema = 0.0
  write_rate_ema = 0.0
  per_worker_write_rate_ema = 0.0
  last_rate_t = time.time()
  avg_train_clip_bytes = float(
    curation_state["train_committed_member_bytes"] / max(1, curation_state["train_committed_clips"])
    if curation_state["train_committed_clips"] > 0
    else estimated_clip_bytes
  )
  last_consumed_total_bytes = 0
  last_written_total_bytes = int(curation_state["train_committed_tar_bytes"])
  deleted_bytes_total = 0
  train_inventory: List[Tuple[Path, str, int, int, int, int]] = []
  last_inventory_t = 0.0
  last_prune_t = 0.0
  lake_blocked = False
  training_phase = "stable"

  def _start_training(phase_name: str, cmd: Sequence[str]) -> subprocess.Popen:
    print(f"[lake] starting {phase_name} training: {' '.join(cmd)}", flush=True)
    proc = _start_process(list(cmd), cwd=repo_root)
    t = threading.Thread(target=_reader, args=("train", proc, q), daemon=True)
    t.start()
    return proc

  def _handle_train_line(line: str) -> None:
    nonlocal trainer_resume_confirmed, current_step
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
      return
    m = TRAIN_STEP_RE.search(line)
    if m:
      observed_step = int(m.group(1))
      if resume_guard_active and (not trainer_resume_confirmed) and observed_step < initial_step:
        _die(
          "Resume mismatch before trainer confirmation: "
          f"first observed step {observed_step} < expected {initial_step}."
        )
      current_step = max(current_step, observed_step)

  def _handle_stream_line(worker_idx: int, line: str) -> None:
    print(f"[stream {worker_idx}] {line}", flush=True)
    m = STREAM_RESUME_RE.match(line)
    if m:
      stream_written_clips[worker_idx] = max(stream_written_clips.get(worker_idx, 0), int(m.group(1)))
      stream_written_bytes[worker_idx] = max(stream_written_bytes.get(worker_idx, 0), int(m.group(2)))
      return
    m = STREAM_PROGRESS_RE.match(line)
    if m:
      stream_written_clips[worker_idx] = max(stream_written_clips.get(worker_idx, 0), int(m.group(1)))
      stream_written_bytes[worker_idx] = max(stream_written_bytes.get(worker_idx, 0), int(m.group(3)))
      return
    m = STREAM_DONE_RE.match(line)
    if m:
      stream_written_clips[worker_idx] = max(stream_written_clips.get(worker_idx, 0), int(m.group(1)))
      stream_written_bytes[worker_idx] = max(stream_written_bytes.get(worker_idx, 0), int(m.group(2)))
      return
    if RESUME_BLOCK_RE.match(line):
      stream_resume_blocked.add(worker_idx)
      print(
        f"[lake] mark worker={worker_idx} blocked (resume compatibility guard). "
        "Use a fresh incoming dir or restore the original stream configuration.",
        flush=True,
      )

  def _pump_messages(deadline: float) -> None:
    while True:
      timeout = max(0.0, deadline - time.time())
      if timeout <= 0:
        return
      try:
        src, line = q.get(timeout=timeout)
      except queue.Empty:
        return
      if src == "train":
        _handle_train_line(line)
      elif src.startswith("stream:"):
        _handle_stream_line(int(src.split(":")[1]), line)

  def _start_stream_worker(worker_idx: int) -> None:
    if worker_idx in stream_resume_blocked:
      return
    if worker_idx in stream_workers and stream_workers[worker_idx].poll() is None:
      return
    target_bytes = int(stream_written_bytes.get(worker_idx, 0)) + int(chunk_bytes_per_stream)
    cmd = [
      args.python,
      str(stream_script),
      "--out",
      str(incoming_dir),
      "--num-clips",
      "0",
      "--target-bytes",
      str(target_bytes),
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
      f"[lake] start stream worker={worker_idx} target_bytes={target_bytes} current_bytes={stream_written_bytes.get(worker_idx, 0)}",
      flush=True,
    )
    proc = _start_process(cmd, cwd=repo_root)
    stream_workers[worker_idx] = proc
    stream_stop_requested.discard(worker_idx)
    worker_target_bytes[worker_idx] = target_bytes
    t = threading.Thread(target=_reader, args=(f"stream:{worker_idx}", proc, q), daemon=True)
    t.start()

  def _hard_stop_worker(worker_idx: int, reason: str) -> None:
    proc = stream_workers.get(worker_idx)
    if proc is None or proc.poll() is not None or worker_idx in stream_stop_requested:
      return
    print(f"[lake] stop stream worker={worker_idx} reason={reason}", flush=True)
    try:
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
          print(f"[lake] stream worker blocked idx={idx} rc={rc} reason=unsafe_resume", flush=True)
        else:
          stream_failures += 1
          print(f"[lake] stream worker failed idx={idx} rc={rc} (failures={stream_failures}/{args.max_stream_failures})", flush=True)
      else:
        if stream_failures > 0:
          stream_failures -= 1
    for idx in finished:
      stream_workers.pop(idx, None)
      draining_workers.discard(idx)
      worker_target_bytes.pop(idx, None)
    if stream_failures >= args.max_stream_failures:
      _die("Too many stream worker failures.")

  def _desired_workers(
    reserve_bytes: int,
    consumed_rate_bps: float,
    write_rate_bps: float,
    running_count: int,
  ) -> int:
    if reserve_bytes >= reserve_high_bytes:
      return 0

    if not args.auto_tune_streams:
      if reserve_bytes < reserve_low_bytes:
        return max(1, args.min_streams)
      return running_count

    if consumed_rate_bps <= 1e-6:
      base_required = max(1, args.min_streams)
    else:
      per_worker = per_worker_write_rate_ema
      if per_worker <= 1e-6 and running_count > 0 and write_rate_bps > 1e-6:
        per_worker = write_rate_bps / float(running_count)
      if per_worker <= 1e-6:
        base_required = max(1, args.min_streams)
      else:
        base_required = max(1, int(math.ceil((consumed_rate_bps * args.stream_headroom) / per_worker)))
    base_required = max(args.min_streams, min(args.num_streams, base_required))
    reserve_mid = int((reserve_low_bytes + reserve_high_bytes) / 2)

    if reserve_bytes <= reserve_low_bytes:
      return base_required
    if reserve_bytes < reserve_mid:
      if write_rate_bps < consumed_rate_bps * 1.05:
        return base_required
      return max(args.min_streams, min(args.num_streams, running_count))
    if write_rate_bps > consumed_rate_bps * 1.10:
      return max(0, base_required - 1)
    return max(0, min(args.num_streams, running_count))

  def _reconcile_worker_count(desired: int, free_gb: float, *, force_blocked: bool) -> None:
    running = _running_workers()
    running_count = len(running)
    if free_gb < args.disk_min_free_gb:
      desired = 0
      force_blocked = True

    if not force_blocked and desired >= running_count and draining_workers:
      for idx in list(sorted(draining_workers)):
        if len(_running_workers()) <= desired:
          draining_workers.discard(idx)

    if desired < running_count:
      candidates = [idx for idx in reversed(running) if (force_blocked or idx >= args.min_streams)]
      excess = running_count - desired
      for idx in candidates[:excess]:
        if args.retire_workers_at_chunk_boundary and not force_blocked:
          if idx not in draining_workers:
            draining_workers.add(idx)
            print(f"[lake] mark worker={idx} draining at chunk boundary", flush=True)
        else:
          _hard_stop_worker(idx, reason="target_workers")
      return

    if desired == running_count:
      return

    needed = desired - running_count
    if needed <= 0:
      return
    running_set = set(running)
    order = _worker_indices_by_need(stream_written_bytes, running_set, args.num_streams)
    for idx in order[:needed]:
      _start_stream_worker(idx)

  def _maybe_compact_store(kind: str, store_dir: Path, manifest_path: Path, max_bytes: int) -> None:
    if kind == "val":
      active_entries = list(curation_state["val_active"])
      replacements = int(curation_state["val_replacements_since_compaction"])
      next_idx_key = "val_next_shard"
      compactions_key = "val_compactions"
      replacements_key = "val_replacements_since_compaction"
    else:
      active_entries = list(curation_state["decay_active"])
      replacements = int(curation_state["decay_replacements_since_compaction"])
      next_idx_key = "decay_next_shard"
      compactions_key = "decay_compactions"
      replacements_key = "decay_replacements_since_compaction"
    if replacements < args.curation_compaction_min_replacements:
      return
    active_bytes = _sum_entry_bytes(active_entries)
    physical_bytes = sum(rec[2] for rec in _build_area_inventory(store_dir))
    if physical_bytes <= max(active_bytes, 1) * args.curation_compaction_factor:
      return
    new_entries, next_idx = _compact_active_entries(
      store_dir=store_dir,
      active_entries=active_entries,
      next_shard_idx=int(curation_state[next_idx_key]),
      shard_size=curation_shard_size,
    )
    curation_state[next_idx_key] = int(next_idx)
    curation_state[compactions_key] = int(curation_state[compactions_key]) + 1
    curation_state[replacements_key] = 0
    if kind == "val":
      curation_state["val_active"] = new_entries
    else:
      curation_state["decay_active"] = new_entries
    _write_json_atomic(manifest_path, _manifest_payload(new_entries, max_bytes=max_bytes))

  def _process_incoming_shard(shard_path: Path) -> None:
    stream_idx, _incoming_shard_idx = _parse_stream_shard(shard_path)
    if stream_idx < 0:
      stream_idx = 0
    train_stream_dir = train_lake_dir / f"stream-{stream_idx:03d}" if args.num_streams > 1 else train_lake_dir
    train_writer: Optional[_TarShardWriter] = None
    val_writer: Optional[_TarShardWriter] = None
    decay_writer: Optional[_TarShardWriter] = None

    val_active = list(curation_state["val_active"])
    decay_active = list(curation_state["decay_active"])
    val_capacity = int(curation_state["val_capacity_entries"])
    decay_capacity = int(curation_state["decay_max_entries"])

    for stem, wav_bytes, json_bytes in _iter_tar_records(shard_path):
      try:
        meta = json.loads(json_bytes.decode("utf-8", "replace"))
      except Exception:
        meta = {}
      clip_id = str(meta.get("clip_id", stem))
      clip_nbytes = int(len(wav_bytes) + len(json_bytes))

      routed_to_val = False
      if val_capacity > 0 and val_sample_ratio > 0.0:
        curation_state["val_seen_candidates"] = int(curation_state["val_seen_candidates"]) + 1
        val_score = _stable_score("val", clip_id, seed=1337 + args.num_streams)
        replace_idx: Optional[int] = None
        accept_val = val_score < val_sample_ratio
        if accept_val and len(val_active) >= val_capacity:
          replace_idx = _max_score_index(val_active)
          accept_val = val_score < float(val_active[replace_idx]["score"])
        if accept_val:
          if replace_idx is not None and len(val_active) >= val_capacity:
            val_active.pop(replace_idx)
            curation_state["val_replacements"] = int(curation_state["val_replacements"]) + 1
            curation_state["val_replacements_since_compaction"] = int(curation_state["val_replacements_since_compaction"]) + 1
          if val_writer is None:
            val_writer = _TarShardWriter(val_lake_dir, int(curation_state["val_next_shard"]))
            curation_state["val_next_shard"] = int(curation_state["val_next_shard"]) + 1
          val_writer.write(stem, wav_bytes, json_bytes)
          val_active.append(
            {
              "tar_path": str(val_writer.final_path.resolve()),
              "stem": stem,
              "nbytes": clip_nbytes,
              "score": val_score,
            }
          )
          routed_to_val = True
      if routed_to_val:
        continue

      if train_writer is None:
        train_next = int(curation_state["train_next_shard_by_stream"].get(str(stream_idx), 0))
        train_writer = _TarShardWriter(train_stream_dir, train_next)
        curation_state["train_next_shard_by_stream"][str(stream_idx)] = train_next + 1
      train_writer.write(stem, wav_bytes, json_bytes)
      curation_state["train_committed_clips"] = int(curation_state["train_committed_clips"]) + 1
      curation_state["train_committed_member_bytes"] = int(curation_state["train_committed_member_bytes"]) + clip_nbytes

      if args.decay_retain_ratio > 0.0 and decay_capacity > 0:
        curation_state["decay_seen_candidates"] = int(curation_state["decay_seen_candidates"]) + 1
        decay_score = _stable_score("decay", clip_id, seed=2026 + args.num_streams)
        if decay_score < args.decay_retain_ratio:
          replace_idx = None
          accept_decay = False
          if len(decay_active) < decay_capacity:
            accept_decay = True
          else:
            replace_idx = _max_score_index(decay_active)
            accept_decay = decay_score < float(decay_active[replace_idx]["score"])
          if accept_decay:
            if replace_idx is not None and len(decay_active) >= decay_capacity:
              decay_active.pop(replace_idx)
              curation_state["decay_replacements"] = int(curation_state["decay_replacements"]) + 1
              curation_state["decay_replacements_since_compaction"] = int(curation_state["decay_replacements_since_compaction"]) + 1
            if decay_writer is None:
              decay_writer = _TarShardWriter(decay_lake_dir, int(curation_state["decay_next_shard"]))
              curation_state["decay_next_shard"] = int(curation_state["decay_next_shard"]) + 1
            decay_writer.write(stem, wav_bytes, json_bytes)
            decay_active.append(
              {
                "tar_path": str(decay_writer.final_path.resolve()),
                "stem": stem,
                "nbytes": clip_nbytes,
                "score": decay_score,
              }
            )

    for writer in (train_writer, val_writer, decay_writer):
      if writer is None:
        continue
      final_path, nbytes, count = writer.finalize()
      if final_path is None or count <= 0:
        continue
      if writer is train_writer:
        curation_state["train_committed_tar_bytes"] = int(curation_state["train_committed_tar_bytes"]) + int(nbytes)

    curation_state["val_active"] = val_active
    curation_state["decay_active"] = decay_active
    _maybe_compact_store("val", val_lake_dir, val_manifest_path, val_max_bytes)
    _maybe_compact_store("decay", decay_lake_dir, decay_manifest_path, decay_max_bytes)
    _write_json_atomic(val_manifest_path, _manifest_payload(curation_state["val_active"], max_bytes=val_max_bytes))
    _write_json_atomic(decay_manifest_path, _manifest_payload(curation_state["decay_active"], max_bytes=decay_max_bytes))
    _write_curation_state(curation_state_file, curation_state)
    try:
      shard_path.unlink()
    except FileNotFoundError:
      pass
    except OSError as exc:
      print(f"[lake] warning: failed to delete processed incoming shard {shard_path}: {exc}", flush=True)

  def _process_incoming_shards(limit: Optional[int] = None) -> int:
    processed = 0
    for shard_path in _discover_final_shards(incoming_dir):
      _process_incoming_shard(shard_path)
      processed += 1
      if limit is not None and processed >= limit:
        break
    return processed

  def _drain_stream_workers(reason: str) -> None:
    for idx in _running_workers():
      _hard_stop_worker(idx, reason=reason)

    grace_deadline = time.time() + max(10.0, args.poll_sec * 20.0)
    while _running_workers() and time.time() < grace_deadline:
      _pump_messages(time.time() + args.poll_sec)
      _reap_workers()
      _process_incoming_shards()

    if _running_workers():
      stuck = _running_workers()
      print(f"[lake] forcing stream termination after drain timeout workers={stuck}", flush=True)
      for idx in stuck:
        proc = stream_workers.get(idx)
        if proc is None or proc.poll() is not None:
          continue
        try:
          proc.terminate()
        except Exception:
          pass
      term_deadline = time.time() + max(5.0, args.poll_sec * 10.0)
      while _running_workers() and time.time() < term_deadline:
        _pump_messages(time.time() + args.poll_sec)
        _reap_workers()
        _process_incoming_shards()

    if _running_workers():
      stuck = _running_workers()
      print(f"[lake] forcing stream kill workers={stuck}", flush=True)
      for idx in stuck:
        proc = stream_workers.get(idx)
        if proc is None or proc.poll() is not None:
          continue
        try:
          proc.kill()
        except Exception:
          pass
      kill_deadline = time.time() + max(2.0, args.poll_sec * 5.0)
      while _running_workers() and time.time() < kill_deadline:
        _pump_messages(time.time() + args.poll_sec)
        _reap_workers()
      _reap_workers()

    while _process_incoming_shards() > 0:
      pass

  def _bootstrap_until_first_train_shard() -> None:
    free_gb = _disk_free_gb(data_root)
    if free_gb < args.disk_min_free_gb:
      _die(
        f"No initial curated train shards and insufficient free space ({free_gb:.1f} GiB < {args.disk_min_free_gb:.1f} GiB). "
        "Cannot bootstrap stream safely."
      )
    order = _worker_indices_by_need(stream_written_bytes, set(), args.num_streams)
    for idx in order[: max(1, args.min_streams)]:
      _start_stream_worker(idx)
    print("[lake] waiting for first curated train shard before starting training", flush=True)
    while not _has_visible_shards(train_lake_dir):
      _pump_messages(time.time() + args.poll_sec)
      _reap_workers()
      _process_incoming_shards()
      if _has_visible_shards(train_lake_dir):
        break
      if not _running_workers():
        _die("Bootstrap streaming stopped before first curated train shard was written.")

  if not _has_visible_shards(train_lake_dir):
    _bootstrap_until_first_train_shard()

  # Anchor reserve accounting at this invocation. Subtracting lifetime training
  # steps from the current on-disk lake double-counts bytes already pruned and
  # makes reserve collapse to zero on long/resumed runs.
  train_inventory = _build_area_inventory(train_lake_dir)
  reserve_origin_active_bytes = sum(rec[2] for rec in train_inventory)
  reserve_origin_written_bytes = int(curation_state["train_committed_tar_bytes"])
  reserve_origin_step = int(current_step)
  last_written_total_bytes = reserve_origin_written_bytes

  train_proc = _start_training("stable", stable_train_cmd)
  last_status_t = 0.0

  try:
    while True:
      _pump_messages(time.time() + args.poll_sec)

      if train_proc.poll() is not None:
        rc = int(train_proc.returncode or 0)
        decay_requested = (args.decay_steps != 0)
        if training_phase == "stable" and rc == 0 and decay_requested:
          print(f"[lake] stable training exited rc={rc}; preparing decay handoff", flush=True)
          _drain_stream_workers("stable_phase_complete")
          stable_final_ckpt = train_out / "ckpt_final.pt"
          if not stable_final_ckpt.exists():
            latest_ckpt = _find_latest_checkpoint(train_out)
            if latest_ckpt is None:
              _die(f"Stable training finished but no checkpoint was found in {train_out}.")
            stable_final_ckpt = latest_ckpt.resolve()
          stable_final_step = max(current_step, _checkpoint_step_hint(stable_final_ckpt))
          stable_final_lr = _checkpoint_optimizer_lr(stable_final_ckpt)
          if stable_final_lr is None:
            _die(f"Stable training finished but optimizer LR could not be read from {stable_final_ckpt}.")
          stable_runtime_defaults = _checkpoint_trainer_defaults(stable_final_ckpt)
          decay_batch_size = int(stable_runtime_defaults.get("batch_size", args.train_batch_size))
          decay_grad_accum = int(stable_runtime_defaults.get("grad_accum", args.train_grad_accum))
          decay_num_workers = int(stable_runtime_defaults.get("num_workers", args.train_num_workers))
          stable_resume_ckpt = (train_out / "ckpt_stable_final.pt").resolve()
          shutil.copy2(stable_final_ckpt, stable_resume_ckpt)
          if not _has_visible_shards(decay_lake_dir):
            print(
              f"[lake] stable phase complete but decay lake is empty at {decay_lake_dir}; skipping decay phase",
              flush=True,
            )
            raise SystemExit(0)
          resolved_decay_steps = int(args.decay_steps)
          if resolved_decay_steps < 0:
            basis_steps = max(stable_target_max_steps, stable_final_step)
            resolved_decay_steps = max(1, int(math.ceil(float(basis_steps) * float(args.decay_fraction))))
          if resolved_decay_steps <= 0:
            print("[lake] decay disabled after resolving decay length; exiting after stable phase", flush=True)
            raise SystemExit(0)
          decay_total_max_steps = max(0, stable_final_step) + resolved_decay_steps
          decay_managed_tokens = [
            "--lr",
            f"{stable_final_lr:.12g}",
            "--lr-schedule",
            "linear",
            "--lr-schedule-start-step",
            str(int(stable_final_step)),
          ]
          decay_cmd = _build_train_cmd(
            data_dir=decay_lake_dir,
            out_dir=decay_out,
            batch_size=decay_batch_size,
            grad_accum=decay_grad_accum,
            num_workers=decay_num_workers,
            extra_tokens=(effective_decay_extra_tokens + decay_managed_tokens),
            resume_from=stable_resume_ckpt,
            max_steps_override=decay_total_max_steps,
            live_refresh=False,
          )
          training_phase = "decay"
          current_step = int(stable_final_step)
          initial_step = int(stable_final_step)
          resume_ckpt_path = stable_resume_ckpt
          resume_guard_active = bool(initial_step > 0)
          trainer_resume_confirmed = False
          print(
            "[lake] starting decay phase "
            f"resume={stable_resume_ckpt} start_step={stable_final_step} "
            f"decay_steps={resolved_decay_steps} total_max_steps={decay_total_max_steps} "
            f"start_lr={stable_final_lr:.3e} schedule=linear "
            f"batch={decay_batch_size} grad_accum={decay_grad_accum} num_workers={decay_num_workers} "
            f"data_dir={decay_lake_dir} out={decay_out}",
            flush=True,
          )
          train_proc = _start_training("decay", decay_cmd)
          last_status_t = 0.0
          continue

        if training_phase == "stable":
          for idx in _running_workers():
            _hard_stop_worker(idx, reason="train_exit")
        print(f"[lake] {training_phase} training exited rc={rc}", flush=True)
        raise SystemExit(rc)

      if training_phase == "decay":
        now = time.time()
        if (now - last_status_t) >= args.status_every_sec:
          decay_inventory = _build_area_inventory(decay_lake_dir)
          decay_bytes = sum(rec[2] for rec in decay_inventory)
          print(
            f"[lake] phase=decay step={current_step} decay_gb={decay_bytes/GIB:.2f} out={decay_out}",
            flush=True,
          )
          last_status_t = now
        continue

      _reap_workers()
      processed_now = _process_incoming_shards()
      now = time.time()

      if (now - last_inventory_t) >= args.inventory_refresh_sec or processed_now > 0:
        train_inventory = _build_area_inventory(train_lake_dir)
        last_inventory_t = now

      active_train_bytes = sum(rec[2] for rec in train_inventory)
      avg_train_clip_bytes = (
        float(curation_state["train_committed_member_bytes"]) / max(1, int(curation_state["train_committed_clips"]))
        if int(curation_state["train_committed_clips"]) > 0
        else float(estimated_clip_bytes)
      )
      consumed_total_clips = max(0, current_step - reserve_origin_step) * args.train_batch_size * args.train_grad_accum
      consumed_total_bytes = int(consumed_total_clips * avg_train_clip_bytes)
      written_total_bytes = int(curation_state["train_committed_tar_bytes"])
      produced_since_origin_bytes = max(0, written_total_bytes - reserve_origin_written_bytes)
      reserve_bytes = _logical_reserve_bytes(
        active_bytes=active_train_bytes,
        origin_active_bytes=reserve_origin_active_bytes,
        produced_since_origin_bytes=produced_since_origin_bytes,
        consumed_since_origin_bytes=consumed_total_bytes,
        pruned_since_origin_bytes=deleted_bytes_total,
      )

      dt = max(1e-9, now - last_rate_t)
      if dt >= 1.0:
        consumed_delta = max(0, consumed_total_bytes - last_consumed_total_bytes)
        written_delta = max(0, written_total_bytes - last_written_total_bytes)
        train_rate_inst = consumed_delta / dt
        write_rate_inst = written_delta / dt
        train_rate_ema = _ema(train_rate_ema, train_rate_inst, args.rate_ema_alpha)
        write_rate_ema = _ema(write_rate_ema, write_rate_inst, args.rate_ema_alpha)
        running_count = max(1, len(_running_workers()))
        if write_rate_inst > 0:
          per_worker_write_rate_ema = _ema(per_worker_write_rate_ema, write_rate_inst / float(running_count), args.rate_ema_alpha)
        last_rate_t = now
        last_consumed_total_bytes = consumed_total_bytes
        last_written_total_bytes = written_total_bytes

      free_gb = _disk_free_gb(data_root)
      hard_free_blocked = free_gb < args.disk_min_free_gb
      lake_max_bytes = int(args.lake_max_gb * GIB)
      lake_resume_bytes = int(args.lake_max_gb * args.lake_resume_fraction * GIB)
      if args.lake_max_gb > 0:
        if (not lake_blocked) and active_train_bytes >= lake_max_bytes:
          lake_blocked = True
          print(f"[lake] pause-by-cap train_gb={active_train_bytes/GIB:.2f} cap_gb={args.lake_max_gb:.2f}", flush=True)
        elif lake_blocked and active_train_bytes <= lake_resume_bytes and (not hard_free_blocked):
          lake_blocked = False
          print(
            f"[lake] resume-by-cap train_gb={active_train_bytes/GIB:.2f} resume_gb={args.lake_max_gb*args.lake_resume_fraction:.2f}",
            flush=True,
          )

      if args.prune_consumed and (now - last_prune_t) >= args.prune_every_sec and train_inventory:
        should_prune = bool(lake_blocked) or (reserve_bytes > reserve_high_bytes)
        if should_prune:
          target_train_bytes = _prune_target_bytes(
            active_bytes=active_train_bytes,
            reserve_bytes=reserve_bytes,
            reserve_low_bytes=reserve_low_bytes,
          )
          if args.lake_max_gb > 0 and lake_blocked:
            target_train_bytes = min(target_train_bytes, lake_resume_bytes)
          deleted_shards, deleted_bytes = _prune_old_shards(
            train_inventory,
            target_total_bytes=max(0, target_train_bytes),
            keep_recent_shards_per_stream=args.prune_keep_recent_shards,
            max_delete_shards=args.prune_max_shards,
          )
          if deleted_shards > 0:
            deleted_bytes_total += deleted_bytes
            print(f"[lake] prune shards={deleted_shards} bytes_gb={deleted_bytes/GIB:.2f}", flush=True)
            train_inventory = _build_area_inventory(train_lake_dir)
            active_train_bytes = sum(rec[2] for rec in train_inventory)
            reserve_bytes = _logical_reserve_bytes(
              active_bytes=active_train_bytes,
              origin_active_bytes=reserve_origin_active_bytes,
              produced_since_origin_bytes=produced_since_origin_bytes,
              consumed_since_origin_bytes=consumed_total_bytes,
              pruned_since_origin_bytes=deleted_bytes_total,
            )
        last_prune_t = now

      blocked_now = bool(lake_blocked or hard_free_blocked)
      desired = _desired_workers(
        reserve_bytes=reserve_bytes,
        consumed_rate_bps=train_rate_ema,
        write_rate_bps=write_rate_ema,
        running_count=len(_running_workers()),
      )
      if blocked_now:
        desired = 0
      _reconcile_worker_count(desired, free_gb, force_blocked=blocked_now)

      if (now - last_status_t) >= args.status_every_sec:
        active_entries_val = len(curation_state["val_active"])
        active_entries_decay = len(curation_state["decay_active"])
        print(
          "[lake] "
          f"step={current_step} train_gb={active_train_bytes/GIB:.2f} "
          f"reserve_gb={reserve_bytes/GIB:.2f} consumed_gb~={consumed_total_bytes/GIB:.2f} "
          f"produced_gb~={produced_since_origin_bytes/GIB:.2f} "
          f"train_rate~={train_rate_ema/GIB:.3f} GiB/s write_rate~={write_rate_ema/GIB:.3f} GiB/s "
          f"workers={len(_running_workers())}/{desired}/{args.num_streams} draining={len(draining_workers)} "
          f"val_entries={active_entries_val} decay_entries={active_entries_decay} "
          f"free_gb={free_gb:.1f} blocked={int(blocked_now)}",
          flush=True,
        )
        last_status_t = now

  except KeyboardInterrupt:
    print("[lake] interrupt received; shutting down children", flush=True)
    for idx in _running_workers():
      _hard_stop_worker(idx, reason="keyboard_interrupt")
    if train_proc is not None and train_proc.poll() is None:
      try:
        train_proc.send_signal(signal.SIGINT)
      except Exception:
        pass
    raise


if __name__ == "__main__":
  main()
