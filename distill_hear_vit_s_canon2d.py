#!/usr/bin/env python3
from __future__ import annotations

"""
Distill Google HeAR (teacher) into a ViT-S student using locally saved clips.

Loss = w_mse * MSE(student, teacher)
     + w_contrastive * InfoNCE(student↔teacher)
     + w_relational * MSE(sim_matrix_student, sim_matrix_teacher)

Expected data layout (from stream_laion_audio_clips.py):
  data/laion_audio_2s/
    shard-000000.tar
    shard-000001.tar
    ...
    manifest.json

Each shard contains:
  <clip_id>.wav
  <clip_id>.json

Example:
  python3 distill_hear_vit_s_canon2d.py \
    --data-dir data/laion_audio_2s \
    --out checkpoints/hear_vit_s \
    --batch-size 64 \
    --max-steps 20000 \
    --num-workers 8 \
    --device cuda
"""

import argparse
from collections import deque
import io
import json
import math
import os
import random
import re
import tarfile
import time
import contextlib
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from adaptive_warmup import CriticalLREstimate, estimate_critical_learning_rate


def _die(msg: str) -> "None":
  raise SystemExit(msg)


def _import_preprocess_audio(repo_root: Path):
  import sys

  import importlib

  src_root = repo_root / "src"
  if src_root.exists() and str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
  try:
    from hear_distill.audio import AudioPreprocessor

    return AudioPreprocessor()
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


def _decode_wav_bytes(wav_bytes: bytes, target_sr: int) -> Optional[torch.Tensor]:
  try:
    from hear_distill.audio import decode_wav_bytes

    return decode_wav_bytes(wav_bytes, target_sr)
  except ImportError:
    pass
  try:
    import soundfile as sf
  except Exception as exc:  # noqa: BLE001
    _die(f"soundfile is required to decode .wav clips: {exc}")

  with io.BytesIO(wav_bytes) as bio:
    audio, sr = sf.read(bio, dtype="float32", always_2d=False)

  if audio is None:
    return None

  # Convert to mono if needed (soundfile returns shape [frames, channels]).
  if audio.ndim == 2:
    audio = audio.mean(axis=1)

  if sr != target_sr:
    try:
      from scipy import signal
    except Exception as exc:  # noqa: BLE001
      _die(f"Resampling requires scipy: {exc}")
    new_len = int(round(audio.shape[0] * (target_sr / sr)))
    audio = signal.resample(audio, new_len)

  return torch.from_numpy(audio).float()


def _iter_tar_pairs(tar_path: Path) -> Iterator[Tuple[bytes, Dict]]:
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
          # Most commonly truncated tar shards; skip the rest of this shard.
          return
        entry = pending.setdefault(stem, {})
        entry[ext] = data
        if ".wav" in entry and ".json" in entry:
          try:
            meta = json.loads(entry[".json"].decode("utf-8", "replace"))
          except Exception:
            meta = {}
          yield entry[".wav"], meta
          pending.pop(stem, None)
  except tarfile.TarError:
    return


def _iter_selected_tar_pairs(tar_path: Path, allowed_stems: Set[str]) -> Iterator[Tuple[bytes, Dict]]:
  if not allowed_stems:
    return
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
        if stem not in allowed_stems or ext not in (".wav", ".json"):
          continue
        f = tf.extractfile(member)
        if f is None:
          continue
        try:
          data = f.read()
        except Exception:
          return
        entry = pending.setdefault(stem, {})
        entry[ext] = data
        if ".wav" in entry and ".json" in entry:
          try:
            meta = json.loads(entry[".json"].decode("utf-8", "replace"))
          except Exception:
            meta = {}
          yield entry[".wav"], meta
          pending.pop(stem, None)
  except tarfile.TarError:
    return


def _load_manifest_entry_groups(manifest_path: Path) -> List[Tuple[Path, Set[str]]]:
  if not manifest_path.exists():
    return []
  try:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
  except Exception:
    return []
  if not isinstance(payload, dict):
    return []
  raw_entries = payload.get("entries")
  if not isinstance(raw_entries, list):
    return []
  grouped: Dict[str, Set[str]] = {}
  for item in raw_entries:
    if not isinstance(item, dict):
      continue
    tar_path_raw = item.get("tar_path")
    stem = item.get("stem")
    if not isinstance(tar_path_raw, str) or not isinstance(stem, str) or not stem:
      continue
    try:
      tar_path = str(Path(tar_path_raw).resolve())
    except Exception:
      continue
    grouped.setdefault(tar_path, set()).add(stem)
  out: List[Tuple[Path, Set[str]]] = []
  for tar_path, stems in grouped.items():
    out.append((Path(tar_path), stems))
  out.sort(key=lambda item: str(item[0]))
  return out


def _discover_shards(data_dir: Path, shards_glob: str, streams_glob: str) -> List[Path]:
  shards: List[Path] = []
  stream_dirs = sorted([p for p in data_dir.glob(streams_glob) if p.is_dir()])
  if stream_dirs:
    for d in stream_dirs:
      shards.extend(sorted(d.glob(shards_glob)))
  else:
    shards = sorted(data_dir.glob(shards_glob))
  return shards


def _count_wavs_in_tar(path: Path) -> int:
  count = 0
  try:
    with tarfile.open(path, mode="r") as tf:
      for m in tf:
        if m.isfile() and m.name.endswith(".wav"):
          count += 1
  except Exception:
    return 0
  return count


def _write_val_shards_file(
  path: Path,
  *,
  val_shards: Sequence[Path],
  step: int,
  deferred: bool,
) -> None:
  payload = {
    "updated_step": int(step),
    "deferred": bool(deferred),
    "val_shards": [str(p.resolve()) for p in val_shards],
  }
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  os.replace(tmp, path)


def _select_val_shards(
  shards: List[Path],
  *,
  seed: int,
  val_fraction: float,
  val_target_clips: int,
  clip_count_cache: Dict[Path, int],
) -> Tuple[List[Path], int]:
  if len(shards) < 2:
    return [], 0
  order = list(shards)
  rng = random.Random(int(seed) + 99991)
  rng.shuffle(order)

  if val_target_clips > 0:
    selected: List[Path] = []
    total = 0
    max_select = max(1, len(order) - 1)
    for p in order:
      if len(selected) >= max_select:
        break
      c = clip_count_cache.get(p)
      if c is None:
        c = _count_wavs_in_tar(p)
        clip_count_cache[p] = c
      if c <= 0:
        continue
      selected.append(p)
      total += c
      if total >= val_target_clips:
        break
    if not selected:
      selected = [order[0]]
      c = clip_count_cache.get(order[0])
      if c is None:
        c = _count_wavs_in_tar(order[0])
        clip_count_cache[order[0]] = c
      total = max(0, c)
    if len(selected) >= len(order):
      selected = selected[:-1]
    return selected, total

  val_count = int(round(len(order) * val_fraction)) if val_fraction > 0 else 0
  if val_fraction > 0 and val_count == 0 and len(order) >= 2:
    val_count = 1
  if val_count >= len(order):
    val_count = max(1, len(order) - 1)
  selected = order[:val_count]
  total = 0
  for p in selected:
    c = clip_count_cache.get(p)
    if c is None:
      c = _count_wavs_in_tar(p)
      clip_count_cache[p] = c
    total += max(0, c)
  return selected, total


class ClipDataset(IterableDataset):
  def __init__(
    self,
    shards: List[Path],
    *,
    clip_samples: int,
    sample_rate: int,
    shuffle_shards: bool,
    seed: int,
    repeat: bool,
    live_data_dir: Optional[Path] = None,
    shards_glob: str = "shard-*.tar",
    streams_glob: str = "stream-*",
    refresh_interval_sec: float = 30.0,
    exclude_shards: Optional[List[Path]] = None,
  ) -> None:
    super().__init__()
    self.shards = shards
    self.clip_samples = int(clip_samples)
    self.sample_rate = int(sample_rate)
    self.shuffle_shards = bool(shuffle_shards)
    self.seed = int(seed)
    self.repeat = bool(repeat)
    self.live_data_dir = live_data_dir
    self.shards_glob = str(shards_glob)
    self.streams_glob = str(streams_glob)
    self.refresh_interval_sec = float(max(1.0, refresh_interval_sec))
    self._last_refresh_t = 0.0
    self.exclude_shards = set(str(p.resolve()) for p in (exclude_shards or []))

  def _current_shards(self) -> List[Path]:
    if self.live_data_dir is None:
      if not self.exclude_shards:
        return self.shards
      return [p for p in self.shards if str(p.resolve()) not in self.exclude_shards]
    now = time.time()
    if (now - self._last_refresh_t) >= self.refresh_interval_sec:
      fresh = _discover_shards(self.live_data_dir, self.shards_glob, self.streams_glob)
      if fresh:
        self.shards = fresh
      self._last_refresh_t = now
    if not self.exclude_shards:
      return self.shards
    return [p for p in self.shards if str(p.resolve()) not in self.exclude_shards]

  def __iter__(self):
    worker = get_worker_info()
    if worker is None:
      worker_id = 0
      num_workers = 1
    else:
      worker_id = worker.id
      num_workers = worker.num_workers

    rng = random.Random(self.seed + worker_id)

    while True:
      all_shards = self._current_shards()
      shard_list = all_shards[worker_id::num_workers]
      if not shard_list:
        time.sleep(min(2.0, self.refresh_interval_sec))
        if not self.repeat:
          break
        continue
      order = list(shard_list)
      if self.shuffle_shards:
        rng.shuffle(order)

      for shard in order:
        try:
          for wav_bytes, _meta in _iter_tar_pairs(shard):
            audio = _decode_wav_bytes(wav_bytes, self.sample_rate)
            if audio is None:
              continue

            # Ensure fixed length.
            if audio.numel() < self.clip_samples:
              pad = self.clip_samples - audio.numel()
              audio = torch.nn.functional.pad(audio, (0, pad))
            elif audio.numel() > self.clip_samples:
              start = rng.randint(0, audio.numel() - self.clip_samples)
              audio = audio[start : start + self.clip_samples]

            yield audio
        except (tarfile.TarError, FileNotFoundError, OSError):
          continue

      if not self.repeat:
        break


class ManifestClipDataset(IterableDataset):
  def __init__(
    self,
    manifest_path: Path,
    *,
    clip_samples: int,
    sample_rate: int,
    shuffle_shards: bool,
    seed: int,
    repeat: bool,
    refresh_interval_sec: float,
  ) -> None:
    super().__init__()
    self.manifest_path = Path(manifest_path)
    self.clip_samples = int(clip_samples)
    self.sample_rate = int(sample_rate)
    self.shuffle_shards = bool(shuffle_shards)
    self.seed = int(seed)
    self.repeat = bool(repeat)
    self.refresh_interval_sec = float(max(1.0, refresh_interval_sec))
    self._last_refresh_t = 0.0
    self._groups: List[Tuple[Path, Set[str]]] = []

  def _current_groups(self) -> List[Tuple[Path, Set[str]]]:
    now = time.time()
    if (not self._groups) or ((now - self._last_refresh_t) >= self.refresh_interval_sec):
      self._groups = _load_manifest_entry_groups(self.manifest_path)
      self._last_refresh_t = now
    return self._groups

  def __iter__(self):
    worker = get_worker_info()
    if worker is None:
      worker_id = 0
      num_workers = 1
    else:
      worker_id = worker.id
      num_workers = worker.num_workers

    rng = random.Random(self.seed + worker_id)

    while True:
      all_groups = self._current_groups()
      group_list = all_groups[worker_id::num_workers]
      if not group_list:
        time.sleep(min(2.0, self.refresh_interval_sec))
        if not self.repeat:
          break
        continue
      order = list(group_list)
      if self.shuffle_shards:
        rng.shuffle(order)

      for shard_path, allowed_stems in order:
        try:
          for wav_bytes, _meta in _iter_selected_tar_pairs(shard_path, allowed_stems):
            audio = _decode_wav_bytes(wav_bytes, self.sample_rate)
            if audio is None:
              continue

            if audio.numel() < self.clip_samples:
              pad = self.clip_samples - audio.numel()
              audio = torch.nn.functional.pad(audio, (0, pad))
            elif audio.numel() > self.clip_samples:
              start = rng.randint(0, audio.numel() - self.clip_samples)
              audio = audio[start : start + self.clip_samples]

            yield audio
        except (tarfile.TarError, FileNotFoundError, OSError):
          continue

      if not self.repeat:
        break


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Distill HeAR into a Canon-adapted ViT student.")
  ap.add_argument("--data-dir", type=Path, default=Path("data/laion_audio_2s"), help="Directory with shard-*.tar files.")
  ap.add_argument("--shards-glob", type=str, default="shard-*.tar", help="Glob pattern for shards.")
  ap.add_argument("--streams-glob", type=str, default="stream-*", help="Glob for stream subfolders inside data-dir.")
  ap.add_argument("--out", type=Path, default=Path("checkpoints/hear_vit_s"), help="Output/checkpoint directory.")
  ap.add_argument(
    "--model-size",
    choices=["tiny", "small", "base", "large"],
    default="small",
    help="ViT family size. Canon widths and patch-grid adapters are inferred from the model.",
  )
  ap.add_argument("--max-steps", type=int, default=20000, help="Number of training steps.")
  ap.add_argument("--batch-size", type=int, default=64, help="Batch size.")
  ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
  ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
  ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device.")
  ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
  ap.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine", "linear"], help="LR schedule.")
  ap.add_argument("--lr-schedule-start-step", type=int, default=0, help="Global step at which LR scheduling begins (used for resumed phase-local decay).")
  ap.add_argument("--lr-warmup-steps", type=int, default=0, help="Linear warmup steps for LR.")
  ap.add_argument("--lr-min-ratio", type=float, default=0.1, help="Final LR ratio for cosine schedule.")
  ap.add_argument(
    "--auto-warmup",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable automatic LR and batch warmup (default: enabled).",
  )
  ap.add_argument("--auto-warmup-init-lr", type=float, default=0.0, help="Initial LR for auto warmup (<=0 uses 0.01 * --lr).")
  ap.add_argument("--auto-warmup-probe-batch-size", type=int, default=0, help="Initial probe batch size for auto warmup (<=0 uses max(8, batch_size//4)).")
  ap.add_argument("--auto-warmup-steps", type=int, default=1000, help="Number of warmup steps when --auto-warmup is enabled.")
  ap.add_argument("--auto-warmup-metric-every", type=int, default=5, help="Warmup metric cadence in steps.")
  ap.add_argument("--auto-warmup-lr-safety-frac", type=float, default=0.8, help="Safety fraction applied to estimated critical LR.")
  ap.add_argument("--auto-warmup-ema-beta", type=float, default=0.9, help="EMA beta for auto-warmup LR smoothing.")
  ap.add_argument("--auto-warmup-cbs-ema-beta", type=float, default=0.995, help="EMA beta for auto-warmup CBS smoothing.")
  ap.add_argument(
    "--gns-batch-window",
    type=int,
    default=9,
    help="Recent raw CBS window used for distribution-aware batch selection.",
  )
  ap.add_argument(
    "--gns-batch-target-utility",
    type=float,
    default=0.5,
    help="Target fraction of asymptotic large-batch utility used to select a CBS-controlled batch.",
  )
  ap.add_argument(
    "--batch-opt-mult",
    type=float,
    default=2.0,
    help="Post-warmup WSD multiplier applied to the selected critical batch (default: 2x).",
  )
  ap.add_argument(
    "--batch-opt-oom-buffer-frac",
    type=float,
    default=0.10,
    help="Free-memory headroom fraction used for near-OOM and OOM batch-opt backoff.",
  )
  ap.add_argument("--auto-warmup-lr-outlier-factor", type=float, default=4.0, help="Clamp LR critical-LR samples to this multiplicative factor around the LR EMA before updating it.")
  ap.add_argument("--auto-warmup-cbs-outlier-factor", type=float, default=4.0, help="Clamp CBS samples to this multiplicative factor around the CBS EMA before updating it.")
  ap.add_argument("--auto-warmup-batch-round-to", type=int, default=8, help="Round auto-warmup batch updates down to this multiple.")
  ap.add_argument("--auto-warmup-max-lr", type=float, default=0.0, help="Optional hard cap on auto-warmup LR (<=0 disables cap).")
  ap.add_argument("--auto-warmup-max-batch-size", type=int, default=0, help="Optional hard cap on auto-warmup batch size (<=0 disables cap).")
  ap.add_argument(
    "--lr-gns-mode",
    type=str,
    default="stable",
    choices=["stable", "sqrt"],
    help="Post-warmup LR policy: keep LR stable or use sqrt GNS adaptation.",
  )
  ap.add_argument("--lr-gns-adapt", action="store_true", help="Deprecated alias for --lr-gns-mode sqrt.")
  ap.add_argument("--lr-gns-ema-beta", type=float, default=0.99, help="EMA beta for GNS optimal batch smoothing.")
  ap.add_argument("--lr-gns-min-samples", type=int, default=20, help="Minimum GNS samples before LR adaptation starts.")
  ap.add_argument("--lr-gns-update-every", type=int, default=50, help="LR adaptation cadence in steps.")
  ap.add_argument("--lr-gns-min-factor", type=float, default=0.1, help="Minimum LR factor from GNS adaptation.")
  ap.add_argument("--lr-gns-max-factor", type=float, default=1.0, help="Maximum LR factor from GNS adaptation.")
  ap.add_argument("--lr-gns-ref-batch", type=float, default=0.0, help="Reference batch for LR adaptation (<=0 uses batch_size*grad_accum).")
  ap.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay.")
  ap.add_argument(
    "--compile-teacher",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Compile the teacher forward path with torch.compile.",
  )
  ap.add_argument(
    "--compile-student",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Compile the student+projection+loss path with torch.compile.",
  )
  ap.add_argument(
    "--compile-preprocess",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Compile the cached mel-PCEN preprocessing module.",
  )
  ap.add_argument("--compile-mode", type=str, default="default", help="torch.compile mode.")
  ap.add_argument(
    "--compile-dynamic",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable dynamic-shape torch.compile (off by default for faster fixed batches).",
  )
  ap.add_argument(
    "--fused-adamw",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use fused AdamW on CUDA when available.",
  )
  ap.add_argument(
    "--optimizer-mode",
    type=str,
    default="default",
    choices=["default", "diagnostic", "teacher-superbatch"],
    help="Training loop mode: baseline, timing diagnostics, or teacher-side superbatching.",
  )
  ap.add_argument("--optimizer-log-every", type=int, default=50, help="Log optimizer / timing diagnostics every N steps.")
  ap.add_argument("--teacher-batch-factor", type=int, default=1, help="Number of student microbatches to group into one teacher pass.")
  ap.add_argument("--teacher-max-batch", type=int, default=0, help="Optional cap on the total teacher batch size (0 disables cap).")
  ap.add_argument("--log-every", type=int, default=50, help="Log every N steps.")
  ap.add_argument("--save-every", type=int, default=1000, help="Checkpoint every N steps.")
  ap.add_argument("--max-checkpoints", type=int, default=20, help="Max numeric checkpoints to keep (0 disables pruning).")
  ap.add_argument("--resume-from", type=Path, default=None, help="Resume training from a checkpoint file.")
  ap.add_argument("--resume-latest", action="store_true", help="Resume from latest checkpoint in --out (ckpt_*.pt, fallback ckpt_final.pt).")
  ap.add_argument(
    "--resume-require-optim",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Require optimizer state to be present when resuming (recommended for true continuation).",
  )
  ap.add_argument("--gns-every", type=int, default=0, help="Estimate gradient noise scale every N steps (0 disables).")
  ap.add_argument("--gns-param-sample", type=int, default=200000, help="Max gradient elements to sample for GNS estimate.")
  ap.add_argument("--teacher-id", type=str, default="google/hear-pytorch", help="Teacher model id.")
  ap.add_argument("--clip-seconds", type=float, default=2.0, help="Clip length in seconds.")
  ap.add_argument("--sample-rate", type=int, default=16000, help="Sample rate.")
  ap.add_argument("--shuffle-shards", action="store_true", help="Shuffle shard order per epoch.")
  ap.add_argument("--repeat", action="store_true", help="Repeat over shards indefinitely (recommended).")
  ap.add_argument("--live-shard-refresh", action="store_true", help="Refresh shard list while training to ingest newly written shards.")
  ap.add_argument("--shard-refresh-sec", type=float, default=30.0, help="Seconds between shard list refreshes when --live-shard-refresh is enabled.")
  ap.add_argument(
    "--amp",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Use mixed precision (defaults on for CUDA, off for CPU).",
  )
  ap.add_argument(
    "--amp-dtype",
    choices=["auto", "float16", "bfloat16"],
    default="auto",
    help="CUDA autocast dtype; auto prefers bfloat16 when supported.",
  )
  ap.add_argument("--seed", type=int, default=1337, help="RNG seed.")
  ap.add_argument("--loss-mse-weight", type=float, default=1.0, help="Weight for MSE loss.")
  ap.add_argument("--loss-contrastive-weight", type=float, default=0.5, help="Weight for contrastive loss.")
  ap.add_argument("--loss-relational-weight", type=float, default=0.5, help="Weight for relational loss.")
  ap.add_argument("--contrastive-temp", type=float, default=0.07, help="Temperature for contrastive loss.")
  ap.add_argument("--canon", action="store_true", help="Enable Canon layers in the student.")
  ap.add_argument("--cannon", action="store_true", dest="canon", help=argparse.SUPPRESS)
  ap.add_argument("--canon-2d", action="store_true", help="Use 2D Canon (depthwise Conv2d) on patch tokens.")
  ap.add_argument("--canon-kernel", type=int, default=4, help="Canon conv kernel size (default: 4).")
  ap.add_argument("--canon-a", action="store_true", help="Canon-A: after norm1, before attention.")
  ap.add_argument("--canon-b", action="store_true", help="Canon-B: after attention output projection (post-attn).")
  ap.add_argument("--canon-b-qkv", action="store_true", help="(Ablation) Use legacy Canon-B on QKV projection output.")
  ap.add_argument("--canon-c", action="store_true", help="Canon-C: after norm2, before MLP.")
  ap.add_argument("--canon-d", action="store_true", help="Canon-D: after MLP FC1, before activation.")
  ap.add_argument("--canon-abcd", action="store_true", help="Enable Canon A/B/C/D placements.")
  ap.add_argument("--canon-no-pos-enc", action="store_true", help="Disable positional encodings when Canon is enabled.")
  ap.add_argument("--canon-pre", action="store_true", help="(Deprecated) Use --canon-a instead.")
  ap.add_argument("--canon-post", action="store_true", help="(Deprecated) Use --canon-c instead.")
  ap.add_argument("--canon-causal", action="store_true", help="Use causal padding in Canon layer (time axis for 2D).")
  ap.add_argument("--val-fraction", type=float, default=0.001, help="Fraction of shards for validation (default 0.05).")
  ap.add_argument("--val-manifest", type=Path, default=None, help="Optional orchestrator-managed validation manifest.")
  ap.add_argument(
    "--val-live-refresh",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Refresh orchestrator-managed validation manifest during training.",
  )
  ap.add_argument(
    "--val-shard-refresh-sec",
    type=float,
    default=30.0,
    help="Seconds between validation manifest refreshes when --val-manifest is set.",
  )
  ap.add_argument("--val-shards-file", type=Path, default=None, help="Optional path to write selected validation shard paths as JSON.")
  ap.add_argument(
    "--val-target-clips",
    type=int,
    default=0,
    help="If >0, choose validation shards to reach at least this many clips (deferred in live mode until available).",
  )
  ap.add_argument(
    "--val-defer-start-steps",
    type=int,
    default=0,
    help="Earliest step to start attempting deferred validation setup.",
  )
  ap.add_argument(
    "--val-defer-check-every",
    type=int,
    default=50,
    help="Check interval (steps) for deferred validation setup.",
  )
  ap.add_argument("--val-every", type=int, default=100, help="Run validation every N steps (default 100).")
  ap.add_argument("--val-batches", type=int, default=20, help="Number of validation batches per eval (default 20).")
  ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
  ap.add_argument("--wandb-project", type=str, default="hear-distill", help="wandb project name.")
  ap.add_argument("--wandb-entity", type=str, default=None, help="wandb entity/team.")
  ap.add_argument("--wandb-run-name", type=str, default=None, help="wandb run name.")
  ap.add_argument("--wandb-tags", type=str, default=None, help="wandb tags (comma-separated).")
  return ap.parse_args()


def _disable_positional_embeddings(model: nn.Module) -> None:
  if not hasattr(model, "pos_embed"):
    print("Warning: model has no pos_embed; cannot disable positional encodings.", flush=True)
    return
  pos = getattr(model, "pos_embed")
  if pos is None:
    print("Warning: model pos_embed is None; positional encodings already disabled.", flush=True)
    return
  if isinstance(pos, nn.Parameter):
    with torch.no_grad():
      new_pos = torch.zeros_like(pos)
    model.pos_embed = nn.Parameter(new_pos, requires_grad=False)
    return
  if torch.is_tensor(pos):
    with torch.no_grad():
      pos.zero_()
    return
  print("Warning: model pos_embed has unsupported type; cannot disable positional encodings.", flush=True)


def _build_student(
  *,
  model_size: str = "small",
  use_canon: bool,
  canon_2d: bool,
  canon_no_pos_enc: bool,
  canon_kernel: int,
  canon_a: bool,
  canon_b: bool,
  canon_b_qkv: bool,
  canon_c: bool,
  canon_d: bool,
  canon_causal: bool,
) -> nn.Module:
  import sys

  src_root = Path(__file__).resolve().parent / "src"
  if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
  from hear_distill.models import CanonConfig, build_audio_vit

  try:
    return build_audio_vit(
      model_size,
      canon=CanonConfig(
        enabled=use_canon,
        use_2d=canon_2d,
        disable_positional_encoding=canon_no_pos_enc,
        kernel_size=canon_kernel,
        a=canon_a,
        b=canon_b,
        b_qkv=canon_b_qkv,
        c=canon_c,
        d=canon_d,
        causal=canon_causal,
      ),
    )
  except Exception as exc:  # noqa: BLE001
    _die(f"Failed to build {model_size} ViT student: {exc}")


def _student_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
  if hasattr(model, "forward_features"):
    feats = model.forward_features(x)
  else:
    feats = model(x)

  if isinstance(feats, (list, tuple)):
    feats = feats[-1]

  if feats.ndim == 3:
    # [B, N, C] -> take CLS token if present
    feats = feats[:, 0, :]
  elif feats.ndim == 4:
    feats = feats.mean(dim=(-2, -1))

  return feats


class CanonLayer(nn.Module):
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
    # x: [B, N, C]
    y = x.transpose(1, 2)  # [B, C, N]
    if self.causal:
      pad_left = self.kernel_size - 1
      pad_right = 0
    else:
      pad_left = (self.kernel_size - 1) // 2
      pad_right = self.kernel_size // 2
    y = F.pad(y, (pad_left, pad_right))
    y = self.conv(y)
    y = y.transpose(1, 2)
    return x + y


class Canon2DLayer(nn.Module):
  def __init__(self, dim: int, kernel_h: int, kernel_w: int, causal_time: bool = False) -> None:
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
    self.grid_size: Optional[Tuple[int, int]] = None
    self.expect_cls: Optional[bool] = None
    self._warned = False
    self._fallback = CanonLayer(dim, kernel_size=self.kernel_h, causal=self.causal_time)

  def _warn_once(self, msg: str) -> None:
    if not self._warned:
      print(msg, flush=True)
      self._warned = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, N, C]
    if x.ndim != 3:
      raise RuntimeError("Canon2D expects input of shape [B, N, C].")

    if self.grid_size is None:
      self._warn_once("Warning: Canon2D missing grid_size; falling back to 1D Canon.")
      return self._fallback(x)

    h, w = self.grid_size
    if not (isinstance(h, int) and isinstance(w, int)):
      self._warn_once("Warning: Canon2D grid_size is invalid; falling back to 1D Canon.")
      return self._fallback(x)

    b, n, c = x.shape
    expected = int(h) * int(w)
    if self.expect_cls is True and n != expected + 1:
      self._warn_once(
        f"Warning: Canon2D expected CLS token with N=1+H*W ({expected + 1}) but got N={n}; falling back to 1D Canon."
      )
      return self._fallback(x)
    has_cls = False
    if n == expected + 1:
      has_cls = True
      cls = x[:, :1, :]
      patches = x[:, 1:, :]
    elif n == expected:
      cls = None
      patches = x
    else:
      self._warn_once(
        f"Warning: Canon2D token count mismatch (N={n}, H*W={expected}); falling back to 1D Canon."
      )
      return self._fallback(x)

    patches = patches.transpose(1, 2).contiguous().view(b, c, int(h), int(w))
    if self.causal_time:
      pad_h_top = self.kernel_h - 1
      pad_h_bottom = 0
    else:
      pad_h_top = (self.kernel_h - 1) // 2
      pad_h_bottom = self.kernel_h // 2
    pad_w_left = (self.kernel_w - 1) // 2
    pad_w_right = self.kernel_w // 2
    patches = F.pad(patches, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom))
    y = self.conv(patches)
    y = y.view(b, c, int(h) * int(w)).transpose(1, 2)
    if has_cls:
      zero_cls = torch.zeros_like(cls)
      y = torch.cat([zero_cls, y], dim=1)
    return x + y


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
    y = self.qkv(x, *args, **kwargs)
    if y.ndim != 3:
      raise RuntimeError("Canon-B expects QKV output of shape [B, N, 3*D].")
    return self.canon(y)


class CanonFC1Wrapper(nn.Module):
  def __init__(self, fc1: nn.Module, canon: nn.Module) -> None:
    super().__init__()
    self.fc1 = fc1
    self.canon = canon

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    y = self.fc1(x, *args, **kwargs)
    if y.ndim != 3:
      raise RuntimeError("Canon-D expects MLP FC1 output of shape [B, N, M].")
    return self.canon(y)


class CanonBlockWrapper(nn.Module):
  def __init__(
    self,
    block: nn.Module,
    dim: int,
    *,
    kernel_size: int = 4,
    canon_a: bool = False,
    canon_b: bool = False,
    canon_b_qkv: bool = False,
    canon_c: bool = False,
    canon_d: bool = False,
    causal: bool = False,
    use_2d: bool = False,
    grid_size: Optional[Tuple[int, int]] = None,
    expect_cls: Optional[bool] = None,
  ) -> None:
    super().__init__()
    self.block = block
    self.use_2d = bool(use_2d)
    self.grid_size = tuple(grid_size) if grid_size is not None else None
    self.canon_b_qkv = bool(canon_b_qkv)
    self.expect_cls = expect_cls
    self._insert_canon(
      dim=dim,
      kernel_size=kernel_size,
      canon_a=canon_a,
      canon_b=canon_b,
      canon_b_qkv=canon_b_qkv,
      canon_c=canon_c,
      canon_d=canon_d,
      causal=causal,
    )

  def _make_canon(self, dim: int, kernel_size: int, causal: bool) -> nn.Module:
    if self.use_2d:
      canon = Canon2DLayer(int(dim), int(kernel_size), int(kernel_size), causal_time=causal)
      canon.grid_size = self.grid_size
      canon.expect_cls = self.expect_cls
      return canon
    return CanonLayer(int(dim), kernel_size=kernel_size, causal=causal)

  def _insert_canon(
    self,
    *,
    dim: int,
    kernel_size: int,
    canon_a: bool,
    canon_b: bool,
    canon_b_qkv: bool,
    canon_c: bool,
    canon_d: bool,
    causal: bool,
  ) -> None:
    block = self.block

    if canon_b:
      if not hasattr(block, "attn"):
        _die("Canon-B requested but block has no `.attn`.")
      attn = block.attn
      if canon_b_qkv:
        if not hasattr(attn, "qkv"):
          _die("Canon-B(QKV) requested but attention has no `.qkv`.")
        qkv = attn.qkv
        qkv_dim = getattr(qkv, "out_features", None)
        if qkv_dim is None:
          _die("Canon-B(QKV) requested but could not read qkv out_features.")
        attn.qkv = CanonQKVWrapper(qkv, self._make_canon(int(qkv_dim), kernel_size, causal))
      else:
        if not hasattr(attn, "proj"):
          _die("Canon-B requested but attention has no `.proj`.")
        attn.proj = nn.Sequential(attn.proj, self._make_canon(int(dim), kernel_size, causal))

    if canon_a:
      if not hasattr(block, "attn"):
        _die("Canon-A requested but block has no `.attn`.")
      block.attn = CanonInputWrapper(block.attn, self._make_canon(int(dim), kernel_size, causal))

    if canon_d:
      if not hasattr(block, "mlp"):
        _die("Canon-D requested but block has no `.mlp`.")
      mlp = block.mlp
      if not hasattr(mlp, "fc1"):
        _die("Canon-D requested but MLP has no `.fc1`.")
      fc1 = mlp.fc1
      hidden_dim = getattr(fc1, "out_features", None)
      if hidden_dim is None:
        _die("Canon-D requested but could not read MLP fc1 out_features.")
      mlp.fc1 = CanonFC1Wrapper(fc1, self._make_canon(int(hidden_dim), kernel_size, causal))

    if canon_c:
      if not hasattr(block, "mlp"):
        _die("Canon-C requested but block has no `.mlp`.")
      block.mlp = CanonInputWrapper(block.mlp, self._make_canon(int(dim), kernel_size, causal))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
  return x / (x.norm(dim=-1, keepdim=True) + eps)


def _format_bytes(n: int) -> str:
  units = ["B", "KB", "MB", "GB", "TB"]
  size = float(n)
  for u in units:
    if size < 1024.0:
      return f"{size:.2f} {u}"
    size /= 1024.0
  return f"{size:.2f} PB"


def _param_count(model: nn.Module) -> Tuple[int, int]:
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total, trainable


def _param_bytes(model: nn.Module) -> int:
  return sum(p.numel() * p.element_size() for p in model.parameters())


def _resolve_canon_flags(args: argparse.Namespace) -> Tuple[bool, bool, bool, bool, bool, bool, bool]:
  canon_a = bool(getattr(args, "canon_a", False))
  canon_b = bool(getattr(args, "canon_b", False))
  canon_c = bool(getattr(args, "canon_c", False))
  canon_d = bool(getattr(args, "canon_d", False))
  legacy_pre = bool(getattr(args, "canon_pre", False))
  legacy_post = bool(getattr(args, "canon_post", False))

  if getattr(args, "canon_abcd", False):
    canon_a = canon_b = canon_c = canon_d = True

  if legacy_pre:
    canon_a = True
  if legacy_post:
    canon_c = True

  if args.canon and not (canon_a or canon_b or canon_c or canon_d or legacy_pre or legacy_post):
    canon_a = canon_b = canon_c = canon_d = True

  use_canon = args.canon or getattr(args, "canon_abcd", False) or canon_a or canon_b or canon_c or canon_d or legacy_pre or legacy_post
  return use_canon, canon_a, canon_b, canon_c, canon_d, legacy_pre, legacy_post


def _cached_checkpoint_bytes(repo_or_path: str) -> Optional[int]:
  path = Path(repo_or_path)
  if path.exists():
    # Local path: sum known weight files if present.
    index_files = list(path.glob("*.index.json"))
    if index_files:
      try:
        idx = json.loads(index_files[0].read_text())
        shards = sorted(set(idx.get("weight_map", {}).values()))
        total = 0
        for s in shards:
          p = path / s
          if p.exists():
            total += p.stat().st_size
        return total if total > 0 else None
      except Exception:
        pass
    candidates = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if candidates:
      return sum(p.stat().st_size for p in candidates if p.exists())
    return None

  try:
    from huggingface_hub import hf_hub_download
  except Exception:
    return None

  def _try_file(fname: str) -> List[Path]:
    try:
      p = hf_hub_download(repo_or_path, fname, local_files_only=True)
      return [Path(p)]
    except Exception:
      return []

  def _try_index(fname: str) -> List[Path]:
    try:
      idx_path = hf_hub_download(repo_or_path, fname, local_files_only=True)
    except Exception:
      return []
    try:
      idx = json.loads(Path(idx_path).read_text())
      shards = sorted(set(idx.get("weight_map", {}).values()))
    except Exception:
      return []
    out: List[Path] = []
    for s in shards:
      try:
        out.append(Path(hf_hub_download(repo_or_path, s, local_files_only=True)))
      except Exception:
        continue
    return out

  paths = (
    _try_file("model.safetensors")
    or _try_file("pytorch_model.bin")
    or _try_index("model.safetensors.index.json")
    or _try_index("pytorch_model.bin.index.json")
  )
  if not paths:
    return None
  return sum(p.stat().st_size for p in paths if p.exists())


def _contrastive_loss(
  student_emb: torch.Tensor,
  teacher_emb: torch.Tensor,
  *,
  temperature: float,
) -> torch.Tensor:
  # Symmetric cross-entropy on similarity matrix (InfoNCE-style).
  s = _l2_normalize(student_emb)
  t = _l2_normalize(teacher_emb)
  logits = (s @ t.t()) / temperature
  targets = torch.arange(logits.shape[0], device=logits.device)
  loss_st = F.cross_entropy(logits, targets)
  loss_ts = F.cross_entropy(logits.t(), targets)
  return 0.5 * (loss_st + loss_ts)


def _relational_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
  # Match pairwise cosine similarity matrices (exclude diagonal).
  if student_emb.shape[0] < 2:
    return torch.zeros((), device=student_emb.device)
  s = _l2_normalize(student_emb)
  t = _l2_normalize(teacher_emb)
  s_sim = s @ s.t()
  t_sim = t @ t.t()
  mask = ~torch.eye(s_sim.shape[0], dtype=torch.bool, device=s_sim.device)
  return F.mse_loss(s_sim[mask], t_sim[mask])


class _NoopScaler:
  def scale(self, loss: torch.Tensor) -> torch.Tensor:
    return loss

  def step(self, optim: torch.optim.Optimizer) -> None:
    optim.step()

  def update(self) -> None:
    return None

  def state_dict(self) -> Dict[str, float]:
    return {}

  def is_enabled(self) -> bool:
    return False


def _teacher_targets_from_spec(
  teacher: nn.Module,
  spec: torch.Tensor,
  *,
  teacher_autocast_ctx,
) -> torch.Tensor:
  # `torch.compile` + AOT autograd cannot save inference tensors for backward.
  # Use `no_grad()` here so teacher targets stay non-grad tensors without the
  # stricter inference-tensor semantics.
  with torch.no_grad():
    with teacher_autocast_ctx():
      return teacher(spec, return_dict=True).pooler_output.detach()


class _StepTimer:
  def __init__(self, *, device: torch.device, enabled: bool) -> None:
    self.device = device
    self.enabled = bool(enabled)
    self.cuda_enabled = bool(self.enabled and device.type == "cuda")
    self._starts: Dict[str, object] = {}
    self._pairs: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
    self._totals_ms: Dict[str, float] = {}

  def start(self, name: str) -> None:
    if not self.enabled:
      return
    if self.cuda_enabled:
      evt = torch.cuda.Event(enable_timing=True)
      evt.record()
      self._starts[name] = evt
    else:
      self._starts[name] = time.perf_counter()

  def stop(self, name: str) -> None:
    if not self.enabled:
      return
    start = self._starts.pop(name, None)
    if start is None:
      return
    if self.cuda_enabled:
      end = torch.cuda.Event(enable_timing=True)
      end.record()
      self._pairs.setdefault(name, []).append((start, end))  # type: ignore[arg-type]
    else:
      elapsed_ms = (time.perf_counter() - float(start)) * 1000.0
      self._totals_ms[name] = self._totals_ms.get(name, 0.0) + elapsed_ms

  def add_ms(self, name: str, elapsed_ms: float) -> None:
    if (not self.enabled) or (not math.isfinite(elapsed_ms)):
      return
    self._totals_ms[name] = self._totals_ms.get(name, 0.0) + float(elapsed_ms)

  def finish(self) -> Dict[str, float]:
    if not self.enabled:
      return {}
    if self.cuda_enabled:
      torch.cuda.synchronize()
      for name, pairs in self._pairs.items():
        total = self._totals_ms.get(name, 0.0)
        for start, end in pairs:
          total += float(start.elapsed_time(end))
        self._totals_ms[name] = total
    out = dict(self._totals_ms)
    self._starts.clear()
    self._pairs.clear()
    self._totals_ms.clear()
    return out


class _PerfWindow:
  def __init__(self) -> None:
    self.count = 0
    self.totals: Dict[str, float] = {}

  def add(self, metrics: Dict[str, float]) -> None:
    if not metrics:
      return
    self.count += 1
    for key, value in metrics.items():
      if not math.isfinite(value):
        continue
      self.totals[key] = self.totals.get(key, 0.0) + float(value)

  def means(self) -> Dict[str, float]:
    if self.count <= 0:
      return {}
    denom = float(self.count)
    return {key: (value / denom) for key, value in self.totals.items()}

  def reset(self) -> None:
    self.count = 0
    self.totals.clear()


def _classify_perf_bottleneck(metrics: Dict[str, float]) -> str:
  step_ms = float(metrics.get("step_ms", 0.0))
  if step_ms <= 0.0:
    return "unknown"
  input_ms = float(metrics.get("loader_wait_ms", 0.0) + metrics.get("h2d_ms", 0.0) + metrics.get("preprocess_ms", 0.0))
  teacher_ms = float(metrics.get("teacher_ms", 0.0))
  student_ms = float(metrics.get("student_fwd_ms", 0.0) + metrics.get("backward_ms", 0.0) + metrics.get("optim_ms", 0.0))
  aux_ms = float(metrics.get("aux_ms", 0.0))
  dominant = max(input_ms, teacher_ms, student_ms, aux_ms)
  if dominant < (0.35 * step_ms):
    return "mixed"
  if dominant == teacher_ms:
    return "teacher_bound"
  if dominant == input_ms:
    return "input_bound"
  if dominant == student_ms:
    return "student_bound"
  return "aux_bound"


def _scalar_to_float(x: torch.Tensor) -> float:
  return float(x.detach().float().cpu().item())


def _set_lr(optim: torch.optim.Optimizer, lr: float) -> None:
  for group in optim.param_groups:
    group["lr"] = float(lr)


def _lr_multiplier(
  *,
  step: int,
  max_steps: int,
  schedule: str,
  warmup_steps: int,
  min_ratio: float,
) -> float:
  if step <= 0:
    return 0.0 if warmup_steps > 0 else 1.0
  if warmup_steps > 0 and step <= warmup_steps:
    return float(step) / float(max(1, warmup_steps))
  if schedule == "none":
    return 1.0
  if schedule == "cosine":
    denom = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    cos_term = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_ratio + (1.0 - min_ratio) * cos_term)
  if schedule == "linear":
    if max_steps <= warmup_steps + 1:
      return 0.0
    denom = max(1, max_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, float(step - warmup_steps - 1) / float(denom)))
    return float(max(0.0, 1.0 - progress))
  return 1.0


def _weighted_distill_loss(
  student_emb: torch.Tensor,
  teacher_emb: torch.Tensor,
  *,
  temperature: float,
  loss_mse_weight: float,
  loss_contrastive_weight: float,
  loss_relational_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  loss_mse = F.mse_loss(student_emb, teacher_emb)
  loss_con = _contrastive_loss(student_emb, teacher_emb, temperature=temperature)
  loss_rel = _relational_loss(student_emb, teacher_emb)
  loss_total = (
    loss_mse_weight * loss_mse
    + loss_contrastive_weight * loss_con
    + loss_relational_weight * loss_rel
  )
  return loss_total, loss_mse, loss_con, loss_rel


class _StudentLossWrapper(nn.Module):
  def __init__(
    self,
    *,
    student: nn.Module,
    proj: nn.Module,
    temperature: float,
    loss_mse_weight: float,
    loss_contrastive_weight: float,
    loss_relational_weight: float,
  ) -> None:
    super().__init__()
    self.student = student
    self.proj = proj
    self.temperature = float(temperature)
    self.loss_mse_weight = float(loss_mse_weight)
    self.loss_contrastive_weight = float(loss_contrastive_weight)
    self.loss_relational_weight = float(loss_relational_weight)

  def forward(
    self,
    spec: torch.Tensor,
    teacher_emb: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    student_feats = _student_features(self.student, spec)
    student_emb = self.proj(student_feats)
    loss_total, loss_mse, loss_con, loss_rel = _weighted_distill_loss(
      student_emb,
      teacher_emb,
      temperature=self.temperature,
      loss_mse_weight=self.loss_mse_weight,
      loss_contrastive_weight=self.loss_contrastive_weight,
      loss_relational_weight=self.loss_relational_weight,
    )
    return loss_total, loss_mse, loss_con, loss_rel, student_emb


def _make_data_loader(
  ds: IterableDataset,
  *,
  batch_size: int,
  num_workers: int,
  pin_memory: bool,
  drop_last: bool,
) -> DataLoader:
  kwargs = {
    "batch_size": int(batch_size),
    "num_workers": int(num_workers),
    "pin_memory": bool(pin_memory),
    "drop_last": bool(drop_last),
  }
  if int(num_workers) > 0:
    kwargs["prefetch_factor"] = 2
  return DataLoader(ds, **kwargs)


def _round_batch_size_down(value: float, multiple: int) -> int:
  mult = max(1, int(multiple))
  if not math.isfinite(value) or value <= 0:
    return 1
  rounded = int(math.floor(float(value) / float(mult))) * mult
  if rounded > 0:
    return rounded
  return max(1, int(math.floor(float(value))))


def _sanitize_positive_float_window(
  values: Sequence[float],
  *,
  max_len: int,
) -> List[float]:
  cleaned: List[float] = []
  for raw_value in values:
    try:
      value = float(raw_value)
    except Exception:
      continue
    if (not math.isfinite(value)) or value <= 0.0:
      continue
    cleaned.append(value)
  if max_len <= 0:
    return []
  return cleaned[-max_len:]


def _mean_large_batch_utility(
  batch_size: float,
  crit_batches: Sequence[float],
) -> float:
  valid = _sanitize_positive_float_window(crit_batches, max_len=max(1, len(crit_batches)))
  if not valid:
    return float("nan")
  batch = max(float(batch_size), 1e-12)
  total = 0.0
  for crit_batch in valid:
    total += batch / (batch + max(float(crit_batch), 1e-12))
  return total / float(len(valid))


def _expected_step_speedup_from_batches(
  *,
  crit_batches: Sequence[float],
  baseline_batch: float,
  candidate_batch: float,
) -> Optional[float]:
  if (not math.isfinite(baseline_batch)) or baseline_batch <= 0.0:
    return None
  if (not math.isfinite(candidate_batch)) or candidate_batch <= 0.0:
    return None
  baseline_utility = _mean_large_batch_utility(float(baseline_batch), crit_batches)
  candidate_utility = _mean_large_batch_utility(float(candidate_batch), crit_batches)
  if (
    (not math.isfinite(baseline_utility))
    or (not math.isfinite(candidate_utility))
    or baseline_utility <= 0.0
  ):
    return None
  return float(candidate_utility / baseline_utility)


def _select_cbs_from_recent_window(
  crit_batches: Sequence[float],
  *,
  target_utility: float,
  max_batch: Optional[float] = None,
) -> Optional[float]:
  valid = _sanitize_positive_float_window(crit_batches, max_len=max(1, len(crit_batches)))
  if not valid:
    return None
  target = min(1.0 - 1e-6, max(1e-6, float(target_utility)))
  lower = 1.0
  upper = max(valid)
  if target < 1.0:
    upper = max(upper, (target / max(1e-6, 1.0 - target)) * max(valid))
  if max_batch is not None and math.isfinite(max_batch) and max_batch > 0.0:
    upper = min(upper, float(max_batch))
  upper = max(lower, upper)
  if _mean_large_batch_utility(lower, valid) >= target:
    return lower
  if _mean_large_batch_utility(upper, valid) < target:
    return upper
  lo = lower
  hi = upper
  for _ in range(48):
    mid = 0.5 * (lo + hi)
    util = _mean_large_batch_utility(mid, valid)
    if not math.isfinite(util):
      return None
    if util >= target:
      hi = mid
    else:
      lo = mid
  return hi


def _resolve_batch_goal_from_cbs(
  selected_cbs: Optional[float],
  *,
  batch_opt_mult: float,
  round_to: int,
  gpu_batch_cap: Optional[int],
  max_batch_size: Optional[int],
) -> Tuple[Optional[int], Optional[float]]:
  if selected_cbs is None or (not math.isfinite(selected_cbs)) or selected_cbs <= 0.0:
    return None, None
  effective_cbs = float(selected_cbs) * max(1.0, float(batch_opt_mult))
  batch_goal = _round_batch_size_down(effective_cbs, round_to)
  if gpu_batch_cap is not None:
    batch_goal = min(batch_goal, int(gpu_batch_cap))
  if max_batch_size is not None:
    batch_goal = min(batch_goal, int(max_batch_size))
  batch_goal = max(1, batch_goal)
  return int(batch_goal), float(effective_cbs)


def _ramp_warmup_lr(
  current_lr: float,
  lr_goal: float,
  *,
  step: int,
  warmup_steps: int,
  metric_every: int,
) -> float:
  if (not math.isfinite(current_lr)) or current_lr <= 0.0:
    current_lr = 1e-8
  if (not math.isfinite(lr_goal)) or lr_goal <= 0.0:
    return current_lr
  updates_left = max(1, int(math.ceil(float(max(1, warmup_steps) - int(step)) / float(max(1, metric_every)))))
  return current_lr + ((lr_goal - current_lr) / float(updates_left))


def _filter_positive_ema_sample(
  value: float,
  ema_value: Optional[float],
  *,
  outlier_factor: float,
) -> float:
  if (not math.isfinite(value)) or value <= 0.0:
    return value
  if ema_value is None or (not math.isfinite(ema_value)) or ema_value <= 0.0:
    return value
  factor = max(1.0, float(outlier_factor))
  if factor <= 1.0:
    return value
  lower = float(ema_value) / factor
  upper = float(ema_value) * factor
  return min(max(float(value), lower), upper)


def _theil_sen_log_slope(
  crit_points: Sequence[Tuple[int, float]],
) -> Optional[float]:
  points: List[Tuple[int, float]] = []
  for raw_step, raw_value in crit_points:
    try:
      step = int(raw_step)
      value = float(raw_value)
    except Exception:
      continue
    if (not math.isfinite(value)) or value <= 0.0:
      continue
    points.append((step, math.log(max(value, 1e-12))))
  if len(points) < 2:
    return None
  slopes: List[float] = []
  for i in range(len(points) - 1):
    x0, y0 = points[i]
    for j in range(i + 1, len(points)):
      x1, y1 = points[j]
      dx = x1 - x0
      if dx <= 0:
        continue
      slopes.append((y1 - y0) / float(dx))
  if not slopes:
    return None
  slopes.sort()
  n = len(slopes)
  mid = n // 2
  if n % 2 == 1:
    return float(slopes[mid])
  return float(0.5 * (slopes[mid - 1] + slopes[mid]))


def _forecast_terminal_crit_lr(
  current_ema_crit_lr: Optional[float],
  crit_points: Sequence[Tuple[int, float]],
  *,
  step: int,
  warmup_steps: int,
) -> Optional[float]:
  if current_ema_crit_lr is None or (not math.isfinite(current_ema_crit_lr)) or current_ema_crit_lr <= 0.0:
    return None
  log_slope = _theil_sen_log_slope(crit_points)
  if log_slope is None or not math.isfinite(log_slope):
    return float(current_ema_crit_lr)
  steps_remaining = max(0, int(max(1, warmup_steps) - int(step + 1)))
  forecast_log = math.log(max(float(current_ema_crit_lr), 1e-12)) + (float(log_slope) * float(steps_remaining))
  forecast_log = min(700.0, max(math.log(1e-12), forecast_log))
  return float(math.exp(forecast_log))


def _cap_early_warmup_lr_increase(
  current_lr: float,
  proposed_lr: float,
  *,
  step: int,
  warmup_steps: int,
  early_fraction: float = 0.10,
  max_increase_fraction: float = 0.10,
) -> float:
  if (not math.isfinite(current_lr)) or current_lr <= 0.0:
    return proposed_lr
  if (not math.isfinite(proposed_lr)) or proposed_lr <= 0.0:
    return proposed_lr
  if proposed_lr <= current_lr:
    return proposed_lr
  early_steps = max(1, int(math.ceil(float(max(1, warmup_steps)) * float(early_fraction))))
  if (step + 1) > early_steps:
    return proposed_lr
  capped_lr = current_lr * (1.0 + float(max_increase_fraction))
  return min(proposed_lr, capped_lr)

def _is_cuda_oom(exc: RuntimeError) -> bool:
  msg = str(exc).lower()
  return "out of memory" in msg or "cuda error: out of memory" in msg


def _cuda_free_memory_fraction(device: torch.device) -> Optional[float]:
  if device.type != "cuda":
    return None
  try:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device=device)
  except Exception:
    return None
  if total_bytes <= 0:
    return None
  return float(free_bytes) / float(total_bytes)


def _probe_total_loss(
  spec: torch.Tensor,
  target: torch.Tensor,
  *,
  student: nn.Module,
  proj: nn.Module,
  contrastive_temp: float,
  loss_mse_weight: float,
  loss_contrastive_weight: float,
  loss_relational_weight: float,
  autocast_ctx,
) -> float:
  with torch.inference_mode():
    with autocast_ctx():
      student_feats = _student_features(student, spec)
      student_emb = proj(student_feats)
      loss_total, _loss_mse, _loss_con, _loss_rel = _weighted_distill_loss(
        student_emb,
        target,
        temperature=contrastive_temp,
        loss_mse_weight=loss_mse_weight,
        loss_contrastive_weight=loss_contrastive_weight,
        loss_relational_weight=loss_relational_weight,
      )
  return float(loss_total.detach().cpu())


def _estimate_critical_lr(
  *,
  loader: DataLoader,
  data_iter: Iterator[torch.Tensor],
  preprocess_audio,
  teacher: nn.Module,
  student: nn.Module,
  proj: nn.Module,
  optim: torch.optim.Optimizer,
  device: torch.device,
  contrastive_temp: float,
  loss_mse_weight: float,
  loss_contrastive_weight: float,
  loss_relational_weight: float,
  autocast_ctx,
  teacher_autocast_ctx,
  current_lr: float,
  prev_estimate: Optional[float],
  max_lr_cap: Optional[float],
) -> Tuple[Optional[CriticalLREstimate], Iterator[torch.Tensor]]:
  has_dense_gradient = any(
    parameter.grad is not None and not parameter.grad.is_sparse
    for group in optim.param_groups
    for parameter in group["params"]
  )
  if not has_dense_gradient:
    return None, data_iter
  batch, data_iter = _next_batch(loader, data_iter)
  student_was_training = student.training
  proj_was_training = proj.training
  student.eval()
  proj.eval()
  try:
    # The virtual LR steps only modify the student and projection. Cache the
    # invariant preprocessing and frozen-teacher target once for the entire
    # line search instead of repeating both for every candidate LR.
    batch = batch.to(device, non_blocking=True)
    with torch.inference_mode():
      spec = preprocess_audio(batch)
      target = _teacher_targets_from_spec(
        teacher,
        spec,
        teacher_autocast_ctx=teacher_autocast_ctx,
      )
    def _held_out_loss() -> float:
      return _probe_total_loss(
        spec,
        target,
        student=student,
        proj=proj,
        contrastive_temp=contrastive_temp,
        loss_mse_weight=loss_mse_weight,
        loss_contrastive_weight=loss_contrastive_weight,
        loss_relational_weight=loss_relational_weight,
        autocast_ctx=autocast_ctx,
      )

    try:
      estimate = estimate_critical_learning_rate(
        _held_out_loss,
        optimizer=optim,
        current_lr=current_lr,
        previous_estimate=prev_estimate,
        max_lr=max_lr_cap,
      )
    except FloatingPointError:
      return None, data_iter
    return estimate, data_iter
  finally:
    if student_was_training:
      student.train()
    if proj_was_training:
      proj.train()


def _next_batch(
  loader: DataLoader,
  data_iter: Iterator[torch.Tensor],
) -> Tuple[torch.Tensor, Iterator[torch.Tensor]]:
  try:
    batch = next(data_iter)
  except StopIteration:
    data_iter = iter(loader)
    batch = next(data_iter)
  return batch, data_iter


def _sample_grad_vector(params: Sequence[torch.nn.Parameter], max_elems: int) -> torch.Tensor:
  remaining = int(max(1, max_elems))
  chunks: List[torch.Tensor] = []
  for p in params:
    g = p.grad
    if g is None:
      continue
    flat = g.detach().float().reshape(-1)
    take = min(remaining, int(flat.numel()))
    if take <= 0:
      break
    chunks.append(flat[:take].cpu())
    remaining -= take
    if remaining <= 0:
      break
  if not chunks:
    return torch.empty((0,), dtype=torch.float32)
  return torch.cat(chunks, dim=0)


def _estimate_gns(
  *,
  loader: DataLoader,
  data_iter: Iterator[torch.Tensor],
  preprocess_audio,
  teacher: nn.Module,
  student: nn.Module,
  proj: nn.Module,
  device: torch.device,
  trainable_params: Sequence[torch.nn.Parameter],
  batch_size: int,
  gns_param_sample: int,
  contrastive_temp: float,
  loss_mse_weight: float,
  loss_contrastive_weight: float,
  loss_relational_weight: float,
  autocast_ctx,
  teacher_autocast_ctx,
) -> Tuple[Optional[Dict[str, float]], Iterator[torch.Tensor]]:
  grads: List[torch.Tensor] = []
  losses: List[float] = []
  student_was_training = student.training
  proj_was_training = proj.training
  student.train()
  proj.train()
  for _ in range(2):
    batch, data_iter = _next_batch(loader, data_iter)
    batch = batch.to(device, non_blocking=True)
    spec = preprocess_audio(batch)
    target = _teacher_targets_from_spec(
      teacher,
      spec,
      teacher_autocast_ctx=teacher_autocast_ctx,
    )
    student.zero_grad(set_to_none=True)
    proj.zero_grad(set_to_none=True)
    with autocast_ctx():
      student_feats = _student_features(student, spec)
      student_emb = proj(student_feats)
      loss_total, loss_mse, loss_con, loss_rel = _weighted_distill_loss(
        student_emb,
        target,
        temperature=contrastive_temp,
        loss_mse_weight=loss_mse_weight,
        loss_contrastive_weight=loss_contrastive_weight,
        loss_relational_weight=loss_relational_weight,
      )
    loss_value = float(loss_total.detach().cpu())
    if not math.isfinite(loss_value):
      student.zero_grad(set_to_none=True)
      proj.zero_grad(set_to_none=True)
      if not student_was_training:
        student.eval()
      if not proj_was_training:
        proj.eval()
      return None, data_iter
    loss_total.backward()
    g = _sample_grad_vector(trainable_params, gns_param_sample)
    if g.numel() == 0:
      student.zero_grad(set_to_none=True)
      proj.zero_grad(set_to_none=True)
      if not student_was_training:
        student.eval()
      if not proj_was_training:
        proj.eval()
      return None, data_iter
    if not torch.isfinite(g).all():
      student.zero_grad(set_to_none=True)
      proj.zero_grad(set_to_none=True)
      if not student_was_training:
        student.eval()
      if not proj_was_training:
        proj.eval()
      return None, data_iter
    grads.append(g)
    losses.append(loss_value)

  student.zero_grad(set_to_none=True)
  proj.zero_grad(set_to_none=True)
  if not student_was_training:
    student.eval()
  if not proj_was_training:
    proj.eval()

  min_elems = min(int(grads[0].numel()), int(grads[1].numel()))
  if min_elems <= 0:
    return None, data_iter
  g1 = grads[0][:min_elems]
  g2 = grads[1][:min_elems]
  diff = g1 - g2
  mean = 0.5 * (g1 + g2)
  noise_batch = 0.5 * float(torch.dot(diff, diff))
  signal = float(torch.dot(mean, mean))
  if (not math.isfinite(noise_batch)) or (not math.isfinite(signal)):
    return None, data_iter
  nsr = noise_batch / max(signal, 1e-12)
  if not math.isfinite(nsr):
    return None, data_iter
  optimal_batch = float(batch_size) * nsr if signal > 0.0 else float("nan")
  return {
    "gns_opt_batch": optimal_batch,
    "gns_noise_batch": noise_batch,
    "gns_signal": signal,
    "gns_nsr": nsr,
    "gns_loss": float(sum(losses) / max(1, len(losses))),
  }, data_iter


def _prune_old_step_checkpoints(out_dir: Path, max_keep: int) -> int:
  if max_keep <= 0:
    return 0
  rx = re.compile(r"^ckpt_(\d+)\.pt$")
  ckpts: List[Tuple[int, Path]] = []
  for p in out_dir.glob("ckpt_*.pt"):
    m = rx.match(p.name)
    if m:
      ckpts.append((int(m.group(1)), p))
  ckpts.sort(key=lambda kv: kv[0])
  extra = len(ckpts) - max_keep
  if extra <= 0:
    return 0
  removed = 0
  for _, p in ckpts[:extra]:
    try:
      p.unlink()
      removed += 1
    except Exception:
      continue
  return removed


def _find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
  rx = re.compile(r"^ckpt_(\d+)\.pt$")
  best: Optional[Tuple[int, Path]] = None
  for p in out_dir.glob("ckpt_*.pt"):
    m = rx.match(p.name)
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


def _load_training_checkpoint(path: Path) -> dict:
  if not path.exists():
    _die(f"Resume checkpoint not found: {path}")
  try:
    return torch.load(path, map_location="cpu", weights_only=False)
  except TypeError:
    return torch.load(path, map_location="cpu")


def main() -> None:
  args = _parse_args()
  torch.manual_seed(args.seed)
  if args.contrastive_temp <= 0:
    _die("--contrastive-temp must be > 0.")
  if args.loss_mse_weight < 0 or args.loss_contrastive_weight < 0 or args.loss_relational_weight < 0:
    _die("Loss weights must be >= 0.")
  if args.canon_kernel <= 0:
    _die("--canon-kernel must be > 0.")
  if not (0.0 <= args.val_fraction < 1.0):
    _die("--val-fraction must be in [0, 1).")
  if args.val_target_clips < 0:
    _die("--val-target-clips must be >= 0.")
  if args.val_defer_start_steps < 0:
    _die("--val-defer-start-steps must be >= 0.")
  if args.val_defer_check_every <= 0:
    _die("--val-defer-check-every must be > 0.")
  if args.val_every <= 0:
    _die("--val-every must be > 0.")
  if args.val_batches <= 0:
    _die("--val-batches must be > 0.")
  if args.val_shard_refresh_sec <= 0:
    _die("--val-shard-refresh-sec must be > 0.")
  if args.lr_warmup_steps < 0:
    _die("--lr-warmup-steps must be >= 0.")
  if not (0.0 <= args.lr_min_ratio <= 1.0):
    _die("--lr-min-ratio must be in [0, 1].")
  if args.auto_warmup_init_lr < 0:
    _die("--auto-warmup-init-lr must be >= 0.")
  if args.auto_warmup_probe_batch_size < 0:
    _die("--auto-warmup-probe-batch-size must be >= 0.")
  if args.auto_warmup_steps <= 0:
    _die("--auto-warmup-steps must be > 0.")
  if args.auto_warmup_metric_every <= 0:
    _die("--auto-warmup-metric-every must be > 0.")
  if args.auto_warmup_lr_safety_frac <= 0:
    _die("--auto-warmup-lr-safety-frac must be > 0.")
  if not (0.0 <= args.auto_warmup_ema_beta < 1.0):
    _die("--auto-warmup-ema-beta must be in [0, 1).")
  if not (0.0 <= args.auto_warmup_cbs_ema_beta < 1.0):
    _die("--auto-warmup-cbs-ema-beta must be in [0, 1).")
  if args.gns_batch_window <= 0:
    _die("--gns-batch-window must be > 0.")
  if not (0.0 < args.gns_batch_target_utility < 1.0):
    _die("--gns-batch-target-utility must be in (0, 1).")
  if args.batch_opt_mult < 1.0:
    _die("--batch-opt-mult must be >= 1.0.")
  if not (0.0 <= args.batch_opt_oom_buffer_frac < 1.0):
    _die("--batch-opt-oom-buffer-frac must be in [0, 1).")
  if args.auto_warmup_lr_outlier_factor < 1.0:
    _die("--auto-warmup-lr-outlier-factor must be >= 1.0.")
  if args.auto_warmup_cbs_outlier_factor < 1.0:
    _die("--auto-warmup-cbs-outlier-factor must be >= 1.0.")
  if args.auto_warmup_batch_round_to <= 0:
    _die("--auto-warmup-batch-round-to must be > 0.")
  if args.auto_warmup_max_lr < 0:
    _die("--auto-warmup-max-lr must be >= 0.")
  if args.auto_warmup_max_batch_size < 0:
    _die("--auto-warmup-max-batch-size must be >= 0.")
  if not (0.0 <= args.lr_gns_ema_beta < 1.0):
    _die("--lr-gns-ema-beta must be in [0, 1).")
  if args.lr_gns_min_samples <= 0:
    _die("--lr-gns-min-samples must be > 0.")
  if args.lr_gns_update_every <= 0:
    _die("--lr-gns-update-every must be > 0.")
  if args.lr_gns_min_factor <= 0:
    _die("--lr-gns-min-factor must be > 0.")
  if args.lr_gns_max_factor <= 0:
    _die("--lr-gns-max-factor must be > 0.")
  if args.lr_gns_min_factor > args.lr_gns_max_factor:
    _die("--lr-gns-min-factor must be <= --lr-gns-max-factor.")
  if args.lr_gns_ref_batch < 0:
    _die("--lr-gns-ref-batch must be >= 0.")
  if args.optimizer_log_every <= 0:
    _die("--optimizer-log-every must be > 0.")
  if args.teacher_batch_factor <= 0:
    _die("--teacher-batch-factor must be > 0.")
  if args.teacher_max_batch < 0:
    _die("--teacher-max-batch must be >= 0.")
  if args.optimizer_mode != "teacher-superbatch" and args.teacher_batch_factor != 1:
    _die("--teacher-batch-factor > 1 requires --optimizer-mode teacher-superbatch.")
  if args.max_checkpoints < 0:
    _die("--max-checkpoints must be >= 0.")
  if args.gns_every < 0:
    _die("--gns-every must be >= 0.")
  if args.gns_param_sample <= 0:
    _die("--gns-param-sample must be > 0.")
  lr_gns_mode = "sqrt" if args.lr_gns_adapt else str(args.lr_gns_mode)
  if lr_gns_mode == "sqrt" and args.gns_every <= 0:
    _die("--lr-gns-mode sqrt requires --gns-every > 0.")
  if args.resume_from is not None and args.resume_latest:
    _die("Use only one of --resume-from or --resume-latest.")
  if args.lr_schedule_start_step < 0:
    _die("--lr-schedule-start-step must be >= 0.")
  if args.auto_warmup and args.lr_warmup_steps > 0:
    _die("--auto-warmup cannot be combined with --lr-warmup-steps.")
  if args.auto_warmup and args.auto_warmup_steps >= args.max_steps:
    _die("--auto-warmup-steps must be < --max-steps so the run has a post-warmup phase.")

  data_dir = args.data_dir
  shards = _discover_shards(data_dir, args.shards_glob, args.streams_glob)
  if not shards:
    _die(f"No shards found in {data_dir} matching {args.shards_glob}")
  orchestrated_val_manifest = args.val_manifest.resolve() if args.val_manifest is not None else None
  val_enabled = bool(args.val_fraction > 0.0 or args.val_target_clips > 0 or orchestrated_val_manifest is not None)
  if len(shards) < 2 and val_enabled and orchestrated_val_manifest is None:
    if args.live_shard_refresh:
      print(
        "Warning: fewer than 2 shards at startup with --live-shard-refresh; validation setup will be deferred.",
        flush=True,
      )
    else:
      _die("Need at least 2 shards to create a validation split.")

  device = torch.device(args.device)
  if device.type == "cuda" and not torch.cuda.is_available():
    _die("CUDA requested but not available.")
  if device.type == "cuda":
    if hasattr(torch, "set_float32_matmul_precision"):
      torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
      torch.backends.cudnn.benchmark = True
      torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
      torch.backends.cuda.matmul.allow_tf32 = True

  repo_root = Path(__file__).resolve().parent
  preprocess_audio = _import_preprocess_audio(repo_root)
  if isinstance(preprocess_audio, nn.Module):
    preprocess_audio = preprocess_audio.eval().to(device)

  # Teacher
  from transformers import AutoModel

  # Avoid background safetensors conversion thread that can 403 on some repos.
  os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

  teacher = AutoModel.from_pretrained(args.teacher_id)
  teacher.eval().to(device)
  for p in teacher.parameters():
    p.requires_grad = False

  # Student
  use_canon, canon_a, canon_b, canon_c, canon_d, legacy_pre, legacy_post = _resolve_canon_flags(args)
  if (legacy_pre or legacy_post) and (args.canon_a or args.canon_b or args.canon_c or args.canon_d or args.canon_abcd):
    print("Warning: --canon-pre/--canon-post are deprecated and overridden by explicit A/B/C/D flags.", flush=True)
  student = _build_student(
    model_size=str(args.model_size),
    use_canon=bool(use_canon),
    canon_2d=bool(getattr(args, "canon_2d", False)),
    canon_no_pos_enc=bool(getattr(args, "canon_no_pos_enc", False)),
    canon_kernel=int(args.canon_kernel),
    canon_a=bool(canon_a),
    canon_b=bool(canon_b),
    canon_b_qkv=bool(getattr(args, "canon_b_qkv", False)),
    canon_c=bool(canon_c),
    canon_d=bool(canon_d),
    canon_causal=bool(args.canon_causal),
  ).to(device)

  if use_canon and getattr(args, "canon_2d", False):
    # Trigger Canon2D shape checks once with a dummy input.
    was_training = student.training
    student.eval()
    with torch.no_grad():
      dummy = torch.zeros((1, 1, 192, 128), device=device)
      _ = _student_features(student, dummy)
    if was_training:
      student.train()
  student_dim = getattr(student, "num_features", None)
  if student_dim is None:
    # try a dummy forward
    dummy = torch.zeros((1, 1, 192, 128), device=device)
    with torch.no_grad():
      student_dim = _student_features(student, dummy).shape[-1]
  proj = nn.Linear(int(student_dim), 512).to(device)

  # Model size/structure prints (as requested).
  teacher_total, teacher_trainable = _param_count(teacher)
  teacher_param_bytes = _param_bytes(teacher)
  teacher_ckpt_bytes = _cached_checkpoint_bytes(args.teacher_id)

  student_total, student_trainable = _param_count(student)
  proj_total, proj_trainable = _param_count(proj)
  student_param_bytes = _param_bytes(student) + _param_bytes(proj)

  print("=== Model Sizes ===", flush=True)
  print(
    f"Teacher: params={teacher_total:,} trainable={teacher_trainable:,} "
    f"param_bytes={_format_bytes(teacher_param_bytes)} "
    f"ckpt_bytes={_format_bytes(teacher_ckpt_bytes) if teacher_ckpt_bytes is not None else 'N/A'}",
    flush=True,
  )
  print(
    f"Student: params={student_total+proj_total:,} trainable={student_trainable+proj_trainable:,} "
    f"param_bytes={_format_bytes(student_param_bytes)}",
    flush=True,
  )
  if use_canon:
    canon_b_mode = "qkv" if getattr(args, "canon_b_qkv", False) else "post-attn"
    pos_enc_mode = "off" if getattr(args, "canon_no_pos_enc", False) else "on"
    print(
      f"(Canon enabled: A={canon_a}, B={canon_b}({canon_b_mode}), C={canon_c}, D={canon_d}, "
      f"kernel={args.canon_kernel}, causal={args.canon_causal}, 2d={bool(getattr(args, 'canon_2d', False))}, "
      f"pos_enc={pos_enc_mode})",
      flush=True,
    )

  # Optimizer
  params = list(student.parameters()) + list(proj.parameters())
  fused_adamw_active = False
  if args.fused_adamw and device.type == "cuda":
    try:
      optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
      fused_adamw_active = True
    except Exception as exc:
      print(f"Warning: fused AdamW unavailable ({exc}); falling back to eager AdamW.", flush=True)
      optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
  else:
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
  amp_enabled = bool(device.type == "cuda" if args.amp is None else args.amp)
  amp_dtype = torch.float32
  if amp_enabled and device.type == "cuda":
    use_bfloat16 = args.amp_dtype == "bfloat16" or (
      args.amp_dtype == "auto"
      and hasattr(torch.cuda, "is_bf16_supported")
      and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    if amp_dtype == torch.float16:
      try:
        scaler = torch.amp.GradScaler("cuda")
      except Exception:
        scaler = torch.cuda.amp.GradScaler()
    else:
      scaler = _NoopScaler()

    def _autocast():
      return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
  else:
    scaler = _NoopScaler()
    def _autocast():
      return contextlib.nullcontext()
  _teacher_autocast = _autocast

  optimizer_mode = str(args.optimizer_mode)
  # CUDA event timing synchronizes the device in _StepTimer.finish(). Keep it
  # exclusive to explicit diagnostics; teacher superbatching is a throughput
  # mode and must not introduce a full GPU barrier after every step.
  perf_enabled = optimizer_mode == "diagnostic"
  teacher_superbatch_enabled = optimizer_mode == "teacher-superbatch"
  teacher_superbatch_runtime_factor = int(args.teacher_batch_factor if teacher_superbatch_enabled else 1)
  perf_window = _PerfWindow() if perf_enabled else None
  compile_available = hasattr(torch, "compile")
  student_loss_module = _StudentLossWrapper(
    student=student,
    proj=proj,
    temperature=args.contrastive_temp,
    loss_mse_weight=args.loss_mse_weight,
    loss_contrastive_weight=args.loss_contrastive_weight,
    loss_relational_weight=args.loss_relational_weight,
  )
  student_loss_runner: nn.Module = student_loss_module
  teacher_compiled_active = False
  student_compiled_active = False
  preprocess_compiled_active = False

  def _compile_kwargs() -> Dict[str, object]:
    return {
      "mode": args.compile_mode,
      "dynamic": bool(args.compile_dynamic),
    }

  def _enable_teacher_compile() -> None:
    nonlocal teacher, teacher_compiled_active
    if teacher_compiled_active or not args.compile_teacher:
      return
    if not compile_available:
      print("Warning: torch.compile unavailable; leaving teacher eager.", flush=True)
      return
    try:
      teacher = torch.compile(teacher, **_compile_kwargs())
      teacher_compiled_active = True
      print(
        f"compile teacher=1 mode={args.compile_mode} dynamic={int(bool(args.compile_dynamic))}",
        flush=True,
      )
    except Exception as exc:
      print(f"Warning: failed to compile teacher ({exc}); leaving teacher eager.", flush=True)

  def _enable_preprocess_compile() -> None:
    nonlocal preprocess_audio, preprocess_compiled_active
    if preprocess_compiled_active or not args.compile_preprocess:
      return
    if not isinstance(preprocess_audio, nn.Module):
      return
    if not compile_available:
      print("Warning: torch.compile unavailable; leaving preprocessing eager.", flush=True)
      return
    try:
      preprocess_audio = torch.compile(preprocess_audio, **_compile_kwargs())
      preprocess_compiled_active = True
      print(
        f"compile preprocess=1 mode={args.compile_mode} dynamic={int(bool(args.compile_dynamic))}",
        flush=True,
      )
    except Exception as exc:
      print(f"Warning: failed to compile preprocessing ({exc}); leaving eager.", flush=True)

  def _enable_student_compile(reason: str) -> None:
    nonlocal student_loss_runner, student_compiled_active
    if student_compiled_active or not args.compile_student:
      return
    if not compile_available:
      print("Warning: torch.compile unavailable; leaving student loss path eager.", flush=True)
      return
    try:
      student_loss_runner = torch.compile(student_loss_module, **_compile_kwargs())
      student_compiled_active = True
      print(
        f"compile student=1 reason={reason} mode={args.compile_mode} dynamic={int(bool(args.compile_dynamic))}",
        flush=True,
      )
    except Exception as exc:
      print(f"Warning: failed to compile student loss path ({exc}); leaving eager.", flush=True)

  resume_ckpt_path: Optional[Path] = None
  if args.resume_from is not None:
    resume_ckpt_path = args.resume_from
  elif args.resume_latest:
    resume_ckpt_path = _find_latest_checkpoint(args.out)
    if resume_ckpt_path is None:
      _die(f"--resume-latest requested, but no checkpoint found in {args.out}.")

  auto_warmup_enabled = bool(args.auto_warmup)
  auto_warmup_init_lr = float(args.auto_warmup_init_lr) if args.auto_warmup_init_lr > 0 else max(float(args.lr) * 0.01, 1e-8)
  auto_warmup_probe_batch_size = int(args.auto_warmup_probe_batch_size) if args.auto_warmup_probe_batch_size > 0 else max(8, int(args.batch_size) // 4)
  auto_warmup_probe_batch_size = max(1, auto_warmup_probe_batch_size)
  auto_warmup_max_lr = float(args.auto_warmup_max_lr) if args.auto_warmup_max_lr > 0 else None
  auto_warmup_max_batch_size = int(args.auto_warmup_max_batch_size) if args.auto_warmup_max_batch_size > 0 else None
  auto_warmup_lr_freeze_step = max(1, int(args.auto_warmup_steps) // 2) if auto_warmup_enabled else 0
  if auto_warmup_max_batch_size is not None:
    auto_warmup_probe_batch_size = min(auto_warmup_probe_batch_size, auto_warmup_max_batch_size)
  current_train_batch = int(args.batch_size)
  auto_warmup_current_lr = float(args.lr)
  auto_warmup_handoff_lr: Optional[float] = None
  auto_warmup_handoff_batch_size: Optional[int] = None
  auto_warmup_ema_crit_lr: Optional[float] = None
  auto_warmup_first_crit_lr: Optional[float] = None
  auto_warmup_recent_crit_points: List[Tuple[int, float]] = []
  auto_warmup_last_crit_update_step: Optional[int] = None
  auto_warmup_last_crit_lr: Optional[float] = None
  auto_warmup_last_crit_sharpness: Optional[float] = None
  auto_warmup_last_lr_goal: Optional[float] = None
  auto_warmup_last_batch_goal: Optional[float] = None
  auto_warmup_gpu_batch_cap: Optional[int] = auto_warmup_max_batch_size
  auto_warmup_last_safe_batch_size = int(args.batch_size)
  auto_warmup_restored = False
  if auto_warmup_enabled:
    current_train_batch = auto_warmup_probe_batch_size
    auto_warmup_current_lr = auto_warmup_init_lr
    auto_warmup_last_safe_batch_size = int(current_train_batch)

  lr_gns_manual_ref_batch = bool(args.lr_gns_ref_batch > 0)
  lr_gns_ema_opt_batch: Optional[float] = None
  lr_gns_selected_opt_batch: Optional[float] = None
  lr_gns_recent_opt_batches: List[float] = []
  batch_opt_runtime_mult = max(1.0, float(args.batch_opt_mult))
  last_batch_opt_step_speedup: Optional[float] = None
  last_effective_opt_batch: Optional[float] = None
  lr_gns_samples = 0
  lr_gns_factor = 1.0
  lr_gns_last_update_step = 0
  step = 0

  def _resolve_lr_gns_ref_batch(batch_size: int) -> float:
    if lr_gns_manual_ref_batch:
      return float(args.lr_gns_ref_batch)
    if auto_warmup_enabled and step >= args.auto_warmup_steps:
      ref_batch_size = auto_warmup_handoff_batch_size if auto_warmup_handoff_batch_size is not None else batch_size
      return float(ref_batch_size * max(1, args.grad_accum))
    return float(batch_size * max(1, args.grad_accum))

  lr_gns_ref_batch = _resolve_lr_gns_ref_batch(current_train_batch)

  if resume_ckpt_path is not None:
    ckpt = _load_training_checkpoint(Path(resume_ckpt_path))
    if "student" not in ckpt or "proj" not in ckpt:
      _die(f"Checkpoint missing 'student' or 'proj': {resume_ckpt_path}")
    student.load_state_dict(ckpt["student"], strict=True)
    proj.load_state_dict(ckpt["proj"], strict=True)
    if "optim" in ckpt and ckpt["optim"] is not None:
      optim.load_state_dict(ckpt["optim"])
    elif args.resume_require_optim:
      _die(
        f"Checkpoint missing optimizer state while resuming: {resume_ckpt_path} "
        "(pass --no-resume-require-optim to allow weights-only resume)."
      )
    if scaler.is_enabled() and ckpt.get("scaler") is not None:
      try:
        scaler.load_state_dict(ckpt["scaler"])
      except Exception as exc:  # noqa: BLE001
        print(f"Warning: failed to load AMP scaler state ({exc}); continuing with fresh scaler.", flush=True)
    step = int(ckpt.get("step", 0))
    gns_state = ckpt.get("lr_gns_state")
    if isinstance(gns_state, dict):
      raw_ema = gns_state.get("ema_opt_batch")
      if raw_ema is not None:
        try:
          lr_gns_ema_opt_batch = float(raw_ema)
        except Exception:
          lr_gns_ema_opt_batch = None
      raw_selected = gns_state.get("selected_opt_batch")
      if raw_selected is not None:
        try:
          lr_gns_selected_opt_batch = float(raw_selected)
        except Exception:
          lr_gns_selected_opt_batch = None
      raw_batch_opt_mult = gns_state.get("batch_opt_runtime_mult")
      if raw_batch_opt_mult is not None:
        try:
          batch_opt_runtime_mult = max(1.0, float(raw_batch_opt_mult))
        except Exception:
          batch_opt_runtime_mult = batch_opt_runtime_mult
      raw_recent_opt_batches = gns_state.get("recent_opt_batches")
      if isinstance(raw_recent_opt_batches, list):
        lr_gns_recent_opt_batches = _sanitize_positive_float_window(
          raw_recent_opt_batches,
          max_len=int(args.gns_batch_window),
        )
      if (
        (lr_gns_selected_opt_batch is None or not math.isfinite(lr_gns_selected_opt_batch) or lr_gns_selected_opt_batch <= 0.0)
        and lr_gns_recent_opt_batches
      ):
        lr_gns_selected_opt_batch = _select_cbs_from_recent_window(
          lr_gns_recent_opt_batches,
          target_utility=float(args.gns_batch_target_utility),
        )
      raw_last_effective = gns_state.get("last_effective_opt_batch")
      if raw_last_effective is not None:
        try:
          last_effective_opt_batch = float(raw_last_effective)
        except Exception:
          last_effective_opt_batch = None
      raw_last_step_speedup = gns_state.get("last_batch_opt_step_speedup")
      if raw_last_step_speedup is not None:
        try:
          last_batch_opt_step_speedup = float(raw_last_step_speedup)
        except Exception:
          last_batch_opt_step_speedup = None
      try:
        lr_gns_samples = int(gns_state.get("samples", 0))
      except Exception:
        lr_gns_samples = 0
      try:
        lr_gns_factor = float(gns_state.get("factor", 1.0))
      except Exception:
        lr_gns_factor = 1.0
      try:
        lr_gns_last_update_step = int(gns_state.get("last_update_step", step))
      except Exception:
        lr_gns_last_update_step = step
    optimizer_mode_state = ckpt.get("optimizer_mode_state")
    if teacher_superbatch_enabled and isinstance(optimizer_mode_state, dict):
      raw_runtime_factor = optimizer_mode_state.get("teacher_superbatch_runtime_factor")
      if raw_runtime_factor is not None:
        try:
          saved_runtime_factor = max(1, int(raw_runtime_factor))
          teacher_superbatch_runtime_factor = min(int(args.teacher_batch_factor), saved_runtime_factor)
        except Exception:
          teacher_superbatch_runtime_factor = teacher_superbatch_runtime_factor
    auto_warmup_state = ckpt.get("auto_warmup_state")
    if auto_warmup_enabled and isinstance(auto_warmup_state, dict):
      auto_warmup_restored = True
      try:
        current_train_batch = int(auto_warmup_state.get("current_batch_size", current_train_batch))
      except Exception:
        current_train_batch = current_train_batch
      try:
        auto_warmup_current_lr = float(auto_warmup_state.get("current_lr", auto_warmup_current_lr))
      except Exception:
        auto_warmup_current_lr = auto_warmup_current_lr
      raw_ema_crit_lr = auto_warmup_state.get("ema_crit_lr")
      if raw_ema_crit_lr is not None:
        try:
          auto_warmup_ema_crit_lr = float(raw_ema_crit_lr)
        except Exception:
          auto_warmup_ema_crit_lr = None
      raw_first_crit_lr = auto_warmup_state.get("first_crit_lr")
      if raw_first_crit_lr is not None:
        try:
          auto_warmup_first_crit_lr = float(raw_first_crit_lr)
        except Exception:
          auto_warmup_first_crit_lr = None
      raw_recent_crit_points = auto_warmup_state.get("recent_crit_points")
      if isinstance(raw_recent_crit_points, list):
        restored_points: List[Tuple[int, float]] = []
        for item in raw_recent_crit_points[-7:]:
          if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
          try:
            restored_step = int(item[0])
            restored_value = float(item[1])
          except Exception:
            continue
          if (not math.isfinite(restored_value)) or restored_value <= 0.0:
            continue
          restored_points.append((restored_step, restored_value))
        auto_warmup_recent_crit_points = restored_points
      raw_last_crit_update_step = auto_warmup_state.get("last_crit_update_step")
      if raw_last_crit_update_step is not None:
        try:
          auto_warmup_last_crit_update_step = int(raw_last_crit_update_step)
        except Exception:
          auto_warmup_last_crit_update_step = None
      raw_last_crit_lr = auto_warmup_state.get("last_crit_lr")
      if raw_last_crit_lr is not None:
        try:
          auto_warmup_last_crit_lr = float(raw_last_crit_lr)
        except Exception:
          auto_warmup_last_crit_lr = None
      raw_last_crit_sharpness = auto_warmup_state.get("last_crit_sharpness")
      if raw_last_crit_sharpness is not None:
        try:
          auto_warmup_last_crit_sharpness = float(raw_last_crit_sharpness)
        except Exception:
          auto_warmup_last_crit_sharpness = None
      raw_last_lr_goal = auto_warmup_state.get("last_lr_goal")
      if raw_last_lr_goal is not None:
        try:
          auto_warmup_last_lr_goal = float(raw_last_lr_goal)
        except Exception:
          auto_warmup_last_lr_goal = None
      raw_last_batch_goal = auto_warmup_state.get("last_batch_goal")
      if raw_last_batch_goal is not None:
        try:
          auto_warmup_last_batch_goal = float(raw_last_batch_goal)
        except Exception:
          auto_warmup_last_batch_goal = None
      raw_handoff_lr = auto_warmup_state.get("handoff_lr")
      if raw_handoff_lr is not None:
        try:
          auto_warmup_handoff_lr = float(raw_handoff_lr)
        except Exception:
          auto_warmup_handoff_lr = None
      raw_handoff_batch = auto_warmup_state.get("handoff_batch_size")
      if raw_handoff_batch is not None:
        try:
          auto_warmup_handoff_batch_size = int(raw_handoff_batch)
        except Exception:
          auto_warmup_handoff_batch_size = None
      raw_gpu_cap = auto_warmup_state.get("gpu_batch_cap")
      if raw_gpu_cap is not None:
        try:
          auto_warmup_gpu_batch_cap = int(raw_gpu_cap)
        except Exception:
          auto_warmup_gpu_batch_cap = auto_warmup_gpu_batch_cap
      try:
        auto_warmup_last_safe_batch_size = int(auto_warmup_state.get("last_safe_batch_size", auto_warmup_last_safe_batch_size))
      except Exception:
        auto_warmup_last_safe_batch_size = auto_warmup_last_safe_batch_size
      if step >= auto_warmup_lr_freeze_step and auto_warmup_last_lr_goal is None:
        auto_warmup_last_lr_goal = float(auto_warmup_current_lr)
    elif auto_warmup_enabled:
      if step > 0:
        _die(
          "Checkpoint has no auto warmup state; cannot safely resume with --auto-warmup. "
          "Resume from a checkpoint produced by the auto-warmup trainer or restart from step 0."
        )
      print("Warning: checkpoint has no auto warmup state; warmup controller will start from fresh defaults.", flush=True)
    lr_gns_ref_batch = _resolve_lr_gns_ref_batch(current_train_batch)
    print(f"Resumed training from {Path(resume_ckpt_path)} at step={step}.", flush=True)

  _enable_preprocess_compile()
  _enable_teacher_compile()
  if (not auto_warmup_enabled) or step >= args.auto_warmup_steps:
    _enable_student_compile("startup")

  # Data
  rng = random.Random(args.seed)
  shard_list = list(shards)
  rng.shuffle(shard_list)
  clip_count_cache: Dict[Path, int] = {}
  val_shards: List[Path] = []
  val_clip_total = 0
  deferred_val_setup = False
  if orchestrated_val_manifest is not None:
    print(f"Validation manifest: {orchestrated_val_manifest}", flush=True)
  elif val_enabled:
    candidate_val_shards, candidate_val_clips = _select_val_shards(
      shard_list,
      seed=args.seed,
      val_fraction=args.val_fraction,
      val_target_clips=args.val_target_clips,
      clip_count_cache=clip_count_cache,
    )
    if not candidate_val_shards:
      deferred_val_setup = bool(args.live_shard_refresh)
      if not deferred_val_setup:
        _die("Unable to create validation split from current shards.")
    elif args.live_shard_refresh and args.val_target_clips > 0 and candidate_val_clips < args.val_target_clips:
      deferred_val_setup = True
      print(
        f"Deferred validation: startup has {candidate_val_clips} clips, waiting for target {args.val_target_clips}.",
        flush=True,
      )
    else:
      val_shards = candidate_val_shards
      val_clip_total = candidate_val_clips
  train_shards = [s for s in shard_list if s not in set(val_shards)]
  if val_enabled and orchestrated_val_manifest is None and not deferred_val_setup and not train_shards:
    if args.live_shard_refresh:
      deferred_val_setup = True
      val_shards = []
      val_clip_total = 0
      train_shards = list(shard_list)
      print("Deferred validation: no train shards left after split; waiting for more data.", flush=True)
    else:
      _die("Validation split left no training shards.")
  print(
    f"Train shards: {len(train_shards)} | Val shards: {len(val_shards)}"
    + (
      f" (manifest={orchestrated_val_manifest})"
      if orchestrated_val_manifest is not None
      else (f" (val_clips~{val_clip_total})" if val_shards else (" (deferred)" if deferred_val_setup else ""))
    ),
    flush=True,
  )
  val_shards_file = args.val_shards_file if args.val_shards_file is not None else (args.out / "val_shards.json")
  val_shards_file = Path(val_shards_file)
  val_shards_file.parent.mkdir(parents=True, exist_ok=True)
  try:
    _write_val_shards_file(
      val_shards_file,
      val_shards=val_shards,
      step=0,
      deferred=deferred_val_setup,
    )
  except Exception as exc:  # noqa: BLE001
    print(f"Warning: failed to write val shard file {val_shards_file}: {exc}", flush=True)

  clip_samples = int(round(args.clip_seconds * args.sample_rate))
  def _build_train_loader(excluded_shards: List[Path], batch_size: int) -> Tuple[ClipDataset, DataLoader]:
    train_source = shard_list if args.live_shard_refresh else [s for s in shard_list if s not in set(excluded_shards)]
    ds = ClipDataset(
      train_source,
      clip_samples=clip_samples,
      sample_rate=args.sample_rate,
      shuffle_shards=args.shuffle_shards,
      seed=args.seed,
      repeat=args.repeat,
      live_data_dir=(data_dir if args.live_shard_refresh else None),
      shards_glob=args.shards_glob,
      streams_glob=args.streams_glob,
      refresh_interval_sec=args.shard_refresh_sec,
      exclude_shards=excluded_shards,
    )
    ld = _make_data_loader(
      ds,
      batch_size=batch_size,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
      drop_last=True,
    )
    return ds, ld

  def _build_val_loader(selected_val_shards: List[Path], batch_size: int) -> Tuple[ClipDataset, DataLoader]:
    ds = ClipDataset(
      selected_val_shards,
      clip_samples=clip_samples,
      sample_rate=args.sample_rate,
      shuffle_shards=True,
      seed=args.seed + 999,
      repeat=True,
      live_data_dir=None,
      shards_glob=args.shards_glob,
      streams_glob=args.streams_glob,
      refresh_interval_sec=args.shard_refresh_sec,
      exclude_shards=None,
    )
    ld = _make_data_loader(
      ds,
      batch_size=batch_size,
      num_workers=max(1, args.num_workers // 2),
      pin_memory=(device.type == "cuda"),
      drop_last=True,
    )
    return ds, ld

  def _build_val_manifest_loader(manifest_path: Path, batch_size: int) -> Tuple[ManifestClipDataset, DataLoader]:
    ds = ManifestClipDataset(
      manifest_path,
      clip_samples=clip_samples,
      sample_rate=args.sample_rate,
      shuffle_shards=True,
      seed=args.seed + 999,
      repeat=True,
      refresh_interval_sec=(args.val_shard_refresh_sec if args.val_live_refresh else 10**9),
    )
    ld = _make_data_loader(
      ds,
      batch_size=batch_size,
      num_workers=max(1, args.num_workers // 2),
      pin_memory=(device.type == "cuda"),
      drop_last=True,
    )
    return ds, ld

  dataset, loader = _build_train_loader(val_shards, current_train_batch)
  val_dataset = None
  val_loader = None
  if orchestrated_val_manifest is not None:
    val_dataset, val_loader = _build_val_manifest_loader(orchestrated_val_manifest, current_train_batch)
  elif val_shards:
    val_dataset, val_loader = _build_val_loader(val_shards, current_train_batch)

  # Training
  out_dir = args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  if args.lr_gns_adapt:
    print("Warning: --lr-gns-adapt is deprecated; use --lr-gns-mode sqrt.", flush=True)
  if auto_warmup_enabled and lr_gns_mode == "sqrt":
    print("Auto warmup enabled: LR GNS sqrt adaptation will stay disabled until warmup ends.", flush=True)
  if auto_warmup_enabled:
    print(
      "Auto warmup enabled: "
      f"init_lr={auto_warmup_current_lr:.3e} probe_batch={current_train_batch} "
      f"steps={args.auto_warmup_steps} metric_every={args.auto_warmup_metric_every} "
      f"lr_freeze_step={auto_warmup_lr_freeze_step} "
      f"lr_safety={args.auto_warmup_lr_safety_frac:.3f} "
      f"lr_ema_beta={args.auto_warmup_ema_beta:.3f} cbs_ema_beta={args.auto_warmup_cbs_ema_beta:.3f} "
      f"cbs_window={args.gns_batch_window} cbs_target_utility={args.gns_batch_target_utility:.3f} "
      f"batch_opt_mult={batch_opt_runtime_mult:.3f} oom_buffer={args.batch_opt_oom_buffer_frac:.3f}"
      + (f" max_lr={auto_warmup_max_lr:.3e}" if auto_warmup_max_lr is not None else "")
      + (f" max_batch={auto_warmup_max_batch_size}" if auto_warmup_max_batch_size is not None else ""),
      flush=True,
    )
  if lr_gns_mode == "sqrt":
    print(
      "LR GNS mode=sqrt: "
      f"ref_batch={lr_gns_ref_batch:.1f} beta={args.lr_gns_ema_beta} "
      f"min_samples={args.lr_gns_min_samples} update_every={args.lr_gns_update_every} "
      f"factor_range=[{args.lr_gns_min_factor}, {args.lr_gns_max_factor}] "
      f"cbs_window={args.gns_batch_window} cbs_target_utility={args.gns_batch_target_utility:.3f} "
      f"batch_opt_mult={batch_opt_runtime_mult:.3f} oom_buffer={args.batch_opt_oom_buffer_frac:.3f}",
      flush=True,
    )
  elif args.gns_every > 0:
    print("LR GNS mode=stable: GNS will be measured and logged, but LR stays on the base schedule.", flush=True)
  student_compile_status = "on" if student_compiled_active else (
    "deferred" if args.compile_student and auto_warmup_enabled and step < args.auto_warmup_steps else "off"
  )
  print(
    "Runtime backend: "
    f"amp={int(amp_enabled and device.type == 'cuda')} "
    f"amp_dtype={str(amp_dtype).removeprefix('torch.')} "
    f"fused_adamw={int(fused_adamw_active)} "
    f"compile_preprocess={'on' if preprocess_compiled_active else 'off'} "
    f"compile_teacher={'on' if teacher_compiled_active else 'off'} "
    f"compile_student={student_compile_status}",
    flush=True,
  )
  if perf_enabled:
    print(
      "Optimizer mode: "
      f"{optimizer_mode} log_every={args.optimizer_log_every} "
      f"teacher_batch_factor={teacher_superbatch_runtime_factor}"
      + (f" teacher_max_batch={args.teacher_max_batch}" if args.teacher_max_batch > 0 else ""),
      flush=True,
    )
  if args.wandb:
    try:
      import wandb  # type: ignore
    except Exception as exc:  # noqa: BLE001
      _die(f"wandb is required when --wandb is set: {exc}")
    wandb.init(
      project=args.wandb_project,
      entity=args.wandb_entity,
      name=args.wandb_run_name,
      tags=[t for t in (args.wandb_tags.split(",") if args.wandb_tags else []) if t],
      config=vars(args),
    )
  else:
    wandb = None  # type: ignore

  def _save_ckpt(tag: str) -> None:
    ckpt = {
      "student": student.state_dict(),
      "proj": proj.state_dict(),
      "optim": optim.state_dict(),
      "scaler": scaler.state_dict() if scaler.is_enabled() else None,
      "step": step,
      "lr_gns_state": {
        "mode": lr_gns_mode,
        "ema_opt_batch": lr_gns_ema_opt_batch,
        "selected_opt_batch": lr_gns_selected_opt_batch,
        "batch_opt_runtime_mult": float(batch_opt_runtime_mult),
        "recent_opt_batches": [float(v) for v in lr_gns_recent_opt_batches[-int(args.gns_batch_window) :]],
        "last_effective_opt_batch": last_effective_opt_batch,
        "last_batch_opt_step_speedup": last_batch_opt_step_speedup,
        "samples": int(lr_gns_samples),
        "factor": float(lr_gns_factor),
        "last_update_step": int(lr_gns_last_update_step),
        "ref_batch": float(lr_gns_ref_batch),
      },
      "optimizer_mode_state": {
        "mode": optimizer_mode,
        "teacher_superbatch_runtime_factor": int(teacher_superbatch_runtime_factor),
      },
      "auto_warmup_state": (
        {
          "current_lr": float(auto_warmup_current_lr),
          "current_batch_size": int(current_train_batch),
          "ema_crit_lr": auto_warmup_ema_crit_lr,
          "first_crit_lr": auto_warmup_first_crit_lr,
          "recent_crit_points": [[int(s), float(v)] for s, v in auto_warmup_recent_crit_points[-7:]],
          "last_crit_update_step": auto_warmup_last_crit_update_step,
          "last_crit_lr": auto_warmup_last_crit_lr,
          "last_crit_sharpness": auto_warmup_last_crit_sharpness,
          "last_lr_goal": auto_warmup_last_lr_goal,
          "last_batch_goal": auto_warmup_last_batch_goal,
          "handoff_lr": auto_warmup_handoff_lr,
          "handoff_batch_size": auto_warmup_handoff_batch_size,
          "gpu_batch_cap": auto_warmup_gpu_batch_cap,
          "last_safe_batch_size": int(auto_warmup_last_safe_batch_size),
        }
        if auto_warmup_enabled
        else None
      ),
      "args": vars(args),
    }
    torch.save(ckpt, out_dir / f"ckpt_{tag}.pt")
    if tag != "final" and args.max_checkpoints > 0:
      removed = _prune_old_step_checkpoints(out_dir, args.max_checkpoints)
      if removed > 0:
        print(f"ckpt_prune removed={removed} keep={args.max_checkpoints}", flush=True)

  data_iter = iter(loader)
  val_iter = iter(val_loader) if val_loader is not None else None
  teacher_cache: Deque[Tuple[torch.Tensor, torch.Tensor]] = deque()
  def _refresh_runtime_loaders(reason: str) -> None:
    nonlocal dataset, loader, data_iter, val_dataset, val_loader, val_iter, lr_gns_ref_batch, teacher_cache
    dataset, loader = _build_train_loader(val_shards, current_train_batch)
    data_iter = iter(loader)
    teacher_cache.clear()
    if val_shards:
      val_dataset, val_loader = _build_val_loader(val_shards, current_train_batch)
      val_iter = iter(val_loader)
    elif orchestrated_val_manifest is not None:
      val_dataset, val_loader = _build_val_manifest_loader(orchestrated_val_manifest, current_train_batch)
      val_iter = iter(val_loader)
    else:
      val_dataset = None
      val_loader = None
      val_iter = None
    lr_gns_ref_batch = _resolve_lr_gns_ref_batch(current_train_batch)
    print(f"runtime_batch_update step={step} batch={current_train_batch} reason={reason}", flush=True)

  def _teacher_superbatch_group_count() -> int:
    if not teacher_superbatch_enabled:
      return 1
    factor = max(1, int(teacher_superbatch_runtime_factor))
    if args.teacher_max_batch > 0:
      factor = min(factor, max(1, int(args.teacher_max_batch) // max(1, current_train_batch)))
    return max(1, factor)

  def _update_batch_opt_diagnostics() -> None:
    nonlocal last_batch_opt_step_speedup, last_effective_opt_batch
    if lr_gns_selected_opt_batch is None or (not math.isfinite(lr_gns_selected_opt_batch)) or lr_gns_selected_opt_batch <= 0.0:
      last_effective_opt_batch = None
      last_batch_opt_step_speedup = None
      return
    last_effective_opt_batch = float(lr_gns_selected_opt_batch) * max(1.0, float(batch_opt_runtime_mult))
    last_batch_opt_step_speedup = _expected_step_speedup_from_batches(
      crit_batches=lr_gns_recent_opt_batches,
      baseline_batch=float(lr_gns_selected_opt_batch),
      candidate_batch=float(last_effective_opt_batch),
    )

  def _current_batch_goal_from_selected() -> Optional[int]:
    batch_goal, effective_cbs = _resolve_batch_goal_from_cbs(
      lr_gns_selected_opt_batch,
      batch_opt_mult=batch_opt_runtime_mult,
      round_to=args.auto_warmup_batch_round_to,
      gpu_batch_cap=auto_warmup_gpu_batch_cap,
      max_batch_size=auto_warmup_max_batch_size,
    )
    if effective_cbs is not None:
      _update_batch_opt_diagnostics()
    return batch_goal

  def _backoff_batch_opt_multiplier(reason: str, *, failed_batch: Optional[int]) -> bool:
    nonlocal batch_opt_runtime_mult, current_train_batch, auto_warmup_gpu_batch_cap, auto_warmup_last_batch_goal
    if lr_gns_selected_opt_batch is None or (not math.isfinite(lr_gns_selected_opt_batch)) or lr_gns_selected_opt_batch <= 0.0:
      return False
    if batch_opt_runtime_mult <= 1.0 and (failed_batch is None or current_train_batch <= 1):
      return False
    buffer_frac = float(args.batch_opt_oom_buffer_frac)
    reference_batch = int(failed_batch) if failed_batch is not None else int(current_train_batch)
    if auto_warmup_active and current_train_batch > auto_warmup_last_safe_batch_size:
      reference_batch = min(reference_batch, int(auto_warmup_last_safe_batch_size))
    target_batch = max(1.0, float(reference_batch) * max(0.0, 1.0 - buffer_frac))
    new_mult = max(1.0, target_batch / max(float(lr_gns_selected_opt_batch), 1e-8))
    changed = bool(new_mult < (batch_opt_runtime_mult - 1e-6))
    batch_opt_runtime_mult = min(batch_opt_runtime_mult, new_mult)
    _update_batch_opt_diagnostics()
    batch_goal = _current_batch_goal_from_selected()
    if batch_goal is not None:
      auto_warmup_last_batch_goal = float(batch_goal)
      if batch_goal < current_train_batch:
        current_train_batch = int(batch_goal)
        if auto_warmup_gpu_batch_cap is None:
          auto_warmup_gpu_batch_cap = int(current_train_batch)
        else:
          auto_warmup_gpu_batch_cap = min(int(auto_warmup_gpu_batch_cap), int(current_train_batch))
        _refresh_runtime_loaders(reason)
        changed = True
    if changed:
      print(
        f"batch_opt backoff step={step} reason={reason} mult={batch_opt_runtime_mult:.3f} "
        f"batch={current_train_batch}"
        + (
          f" cbs_sel~={lr_gns_selected_opt_batch:.1f}"
          if lr_gns_selected_opt_batch is not None
          else ""
        )
        + (
          f" step_x~={last_batch_opt_step_speedup:.3f}"
          if last_batch_opt_step_speedup is not None
          else ""
        ),
        flush=True,
      )
    return changed

  def _maybe_backoff_near_oom() -> bool:
    nonlocal teacher_superbatch_runtime_factor, teacher_cache
    free_frac = _cuda_free_memory_fraction(device)
    if free_frac is None or free_frac >= float(args.batch_opt_oom_buffer_frac):
      return False
    effective_teacher_group = _teacher_superbatch_group_count()
    if teacher_superbatch_enabled and effective_teacher_group > 1:
      failed_factor = int(teacher_superbatch_runtime_factor)
      teacher_superbatch_runtime_factor = max(1, failed_factor // 2)
      teacher_cache.clear()
      if device.type == "cuda":
        torch.cuda.empty_cache()
      print(
        f"teacher_superbatch near_oom_backoff step={step} free_frac={free_frac:.3f} "
        f"failed_factor={failed_factor} new_factor={teacher_superbatch_runtime_factor}",
        flush=True,
      )
      return True
    return _backoff_batch_opt_multiplier("near_oom_backoff", failed_batch=current_train_batch)

  def _collect_teacher_group(group_count: int, step_timer: _StepTimer) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    nonlocal data_iter
    raw_batches: List[torch.Tensor] = []
    chunk_sizes: List[int] = []
    wait_start = time.perf_counter()
    for _ in range(max(1, int(group_count))):
      batch, data_iter = _next_batch(loader, data_iter)
      raw_batches.append(batch)
      chunk_sizes.append(int(batch.shape[0]))
    step_timer.add_ms("loader_wait_ms", (time.perf_counter() - wait_start) * 1000.0)

    merged_batch = raw_batches[0] if len(raw_batches) == 1 else torch.cat(raw_batches, dim=0)
    step_timer.start("h2d_ms")
    merged_batch = merged_batch.to(device, non_blocking=True)
    step_timer.stop("h2d_ms")
    step_timer.start("preprocess_ms")
    spec_all = preprocess_audio(merged_batch)
    step_timer.stop("preprocess_ms")
    step_timer.start("teacher_ms")
    target_all = _teacher_targets_from_spec(
      teacher,
      spec_all,
      teacher_autocast_ctx=_teacher_autocast,
    )
    step_timer.stop("teacher_ms")
    return list(spec_all.split(chunk_sizes, dim=0)), list(target_all.split(chunk_sizes, dim=0)), len(chunk_sizes)

  def _fill_teacher_cache(step_timer: _StepTimer) -> None:
    nonlocal teacher_cache
    if teacher_cache:
      return
    group_count = _teacher_superbatch_group_count()
    spec_chunks, target_chunks, _actual_group_count = _collect_teacher_group(group_count, step_timer)
    teacher_cache.extend(zip(spec_chunks, target_chunks))

  auto_warmup_active = bool(auto_warmup_enabled and step < args.auto_warmup_steps)
  if auto_warmup_enabled and not auto_warmup_active:
    if auto_warmup_handoff_lr is None:
      auto_warmup_handoff_lr = float(auto_warmup_current_lr)
    if auto_warmup_handoff_batch_size is None:
      auto_warmup_handoff_batch_size = int(current_train_batch)
  if auto_warmup_active and (not auto_warmup_restored or lr_gns_selected_opt_batch is None):
    bootstrap_metrics, data_iter = _estimate_gns(
      loader=loader,
      data_iter=data_iter,
      preprocess_audio=preprocess_audio,
      teacher=teacher,
      student=student,
      proj=proj,
      device=device,
      trainable_params=params,
      batch_size=current_train_batch,
      gns_param_sample=args.gns_param_sample,
      contrastive_temp=args.contrastive_temp,
      loss_mse_weight=args.loss_mse_weight,
      loss_contrastive_weight=args.loss_contrastive_weight,
      loss_relational_weight=args.loss_relational_weight,
      autocast_ctx=_autocast,
      teacher_autocast_ctx=_teacher_autocast,
    )
    if bootstrap_metrics is not None:
      bootstrap_opt_batch = float(bootstrap_metrics.get("gns_opt_batch", float("nan")))
      if math.isfinite(bootstrap_opt_batch) and bootstrap_opt_batch > 0:
        lr_gns_recent_opt_batches = _sanitize_positive_float_window(
          list(lr_gns_recent_opt_batches) + [bootstrap_opt_batch],
          max_len=int(args.gns_batch_window),
        )
        lr_gns_ema_opt_batch = bootstrap_opt_batch
        lr_gns_selected_opt_batch = _select_cbs_from_recent_window(
          lr_gns_recent_opt_batches,
          target_utility=float(args.gns_batch_target_utility),
        )
        lr_gns_samples = max(1, lr_gns_samples)
        _update_batch_opt_diagnostics()
        bootstrap_batch = _current_batch_goal_from_selected()
        if bootstrap_batch is None:
          bootstrap_batch = _round_batch_size_down(bootstrap_opt_batch, args.auto_warmup_batch_round_to)
          if auto_warmup_gpu_batch_cap is not None:
            bootstrap_batch = min(bootstrap_batch, auto_warmup_gpu_batch_cap)
          if auto_warmup_max_batch_size is not None:
            bootstrap_batch = min(bootstrap_batch, auto_warmup_max_batch_size)
        auto_warmup_last_batch_goal = float(bootstrap_batch)
        print(
          f"auto_warmup measurement_batch={current_train_batch} "
          f"post_warmup_batch_goal={bootstrap_batch} cbs~={bootstrap_opt_batch:.1f} "
          + (
            f" cbs_sel~={lr_gns_selected_opt_batch:.1f}"
            if lr_gns_selected_opt_batch is not None
            else ""
          )
          + (
            f" cbs_eff~={last_effective_opt_batch:.1f}"
            if last_effective_opt_batch is not None
            else ""
          )
          + (
            f" step_x~={last_batch_opt_step_speedup:.3f}"
            if last_batch_opt_step_speedup is not None
            else ""
          )
          + " "
          f"nsr={bootstrap_metrics['gns_nsr']:.4f}",
          flush=True,
        )

  while step < args.max_steps:
    if (
      deferred_val_setup
      and val_enabled
      and orchestrated_val_manifest is None
      and args.live_shard_refresh
      and step >= args.val_defer_start_steps
      and (step % args.val_defer_check_every == 0)
    ):
      fresh_shards = _discover_shards(data_dir, args.shards_glob, args.streams_glob)
      if len(fresh_shards) >= 2:
        candidate_val_shards, candidate_val_clips = _select_val_shards(
          fresh_shards,
          seed=args.seed,
          val_fraction=args.val_fraction,
          val_target_clips=args.val_target_clips,
          clip_count_cache=clip_count_cache,
        )
        target_ready = (
          bool(candidate_val_shards)
          and (args.val_target_clips <= 0 or candidate_val_clips >= args.val_target_clips)
        )
        if target_ready:
          val_shards = candidate_val_shards
          val_clip_total = candidate_val_clips
          dataset, loader = _build_train_loader(val_shards, current_train_batch)
          data_iter = iter(loader)
          val_dataset, val_loader = _build_val_loader(val_shards, current_train_batch)
          val_iter = iter(val_loader)
          deferred_val_setup = False
          try:
            _write_val_shards_file(
              val_shards_file,
              val_shards=val_shards,
              step=step,
              deferred=False,
            )
          except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to update val shard file {val_shards_file}: {exc}", flush=True)
          print(
            f"Validation enabled at step={step}: val_shards={len(val_shards)} val_clips~{val_clip_total}",
            flush=True,
          )
        else:
          print(
            f"Deferred validation: have val_clips~{candidate_val_clips}, waiting for target {args.val_target_clips}.",
            flush=True,
          )

    auto_warmup_active = bool(auto_warmup_enabled and step < args.auto_warmup_steps)
    lr_gns_controller_active = bool(lr_gns_mode == "sqrt" and not auto_warmup_active)
    if auto_warmup_active:
      base_lr = float(auto_warmup_current_lr)
    elif auto_warmup_enabled:
      if auto_warmup_handoff_lr is None:
        auto_warmup_handoff_lr = float(auto_warmup_current_lr)
      if auto_warmup_handoff_batch_size is None:
        auto_warmup_handoff_batch_size = int(current_train_batch)
      lr_gns_ref_batch = _resolve_lr_gns_ref_batch(current_train_batch)
      schedule_step = max(0, step - args.auto_warmup_steps)
      schedule_max_steps = max(1, args.max_steps - args.auto_warmup_steps)
      base_lr = float(auto_warmup_handoff_lr) * _lr_multiplier(
        step=schedule_step,
        max_steps=schedule_max_steps,
        schedule=args.lr_schedule,
        warmup_steps=0,
        min_ratio=args.lr_min_ratio,
      )
    else:
      lr_gns_ref_batch = _resolve_lr_gns_ref_batch(current_train_batch)
      schedule_start_step = max(0, int(args.lr_schedule_start_step))
      schedule_step = max(0, (step + 1) - schedule_start_step)
      schedule_max_steps = max(1, args.max_steps - schedule_start_step)
      base_lr = args.lr * _lr_multiplier(
        step=schedule_step,
        max_steps=schedule_max_steps,
        schedule=args.lr_schedule,
        warmup_steps=args.lr_warmup_steps,
        min_ratio=args.lr_min_ratio,
      )
    cur_lr = base_lr * (lr_gns_factor if lr_gns_controller_active else 1.0)
    _set_lr(optim, cur_lr)
    optim.zero_grad(set_to_none=True)
    total_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    mse_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    con_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    rel_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    cos_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    tnorm_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    snorm_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    mse0_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    ev_sum_t = torch.zeros((), device=device, dtype=torch.float32)
    warmup_metric_step = bool(auto_warmup_active and ((step + 1) % args.auto_warmup_metric_every == 0))
    step_wall_start = time.perf_counter()
    step_timer = _StepTimer(device=device, enabled=perf_enabled)
    try:
      accum_done = 0
      while accum_done < args.grad_accum:
        _fill_teacher_cache(step_timer)
        spec, target = teacher_cache.popleft()
        step_timer.start("student_fwd_ms")
        with _autocast():
          loss_total, loss_mse, loss_con, loss_rel, student_emb = student_loss_runner(
            spec,
            target,
          )
          loss = loss_total / args.grad_accum
        finite_loss = torch.isfinite(loss_total.detach()).all()
        if device.type == "cuda" and hasattr(torch, "_assert_async"):
          torch._assert_async(  # type: ignore[attr-defined]
            finite_loss,
            f"Non-finite training loss at step={step} lr={cur_lr:.3e} batch={current_train_batch}",
          )
        elif not bool(finite_loss):
          raise RuntimeError(
            "Non-finite training loss "
            f"at step={step} lr={cur_lr:.3e} batch={current_train_batch} "
            f"(mse={float(loss_mse.detach().float().cpu()):.6f} "
            f"con={float(loss_con.detach().float().cpu()):.6f} "
            f"rel={float(loss_rel.detach().float().cpu()):.6f})"
          )

        with torch.no_grad():
          t_f = target.float()
          s_f = student_emb.float()
          tnorm = t_f.norm(dim=-1).mean()
          snorm = s_f.norm(dim=-1).mean()
          cos = F.cosine_similarity(s_f, t_f, dim=-1).mean()
          mse0 = (t_f * t_f).mean()
          var = t_f.var(unbiased=False)
          ev = 1.0 - (loss_mse.float() / (var + 1e-8))
        step_timer.stop("student_fwd_ms")

        step_timer.start("backward_ms")
        scaler.scale(loss).backward()
        step_timer.stop("backward_ms")
        total_loss_t = total_loss_t + loss.detach().float()
        mse_sum_t = mse_sum_t + loss_mse.detach().float()
        con_sum_t = con_sum_t + loss_con.detach().float()
        rel_sum_t = rel_sum_t + loss_rel.detach().float()
        cos_sum_t = cos_sum_t + cos.detach().float()
        tnorm_sum_t = tnorm_sum_t + tnorm.detach().float()
        snorm_sum_t = snorm_sum_t + snorm.detach().float()
        mse0_sum_t = mse0_sum_t + mse0.detach().float()
        ev_sum_t = ev_sum_t + ev.detach().float()
        accum_done += 1

      if warmup_metric_step:
        aux_start = time.perf_counter()
        current_safe_lr: Optional[float] = None
        if step < auto_warmup_lr_freeze_step:
          if scaler.is_enabled():
            scaler.unscale_(optim)
          crit_lr_result, data_iter = _estimate_critical_lr(
            loader=loader,
            data_iter=data_iter,
            preprocess_audio=preprocess_audio,
            teacher=teacher,
            student=student,
            proj=proj,
            optim=optim,
            device=device,
            contrastive_temp=args.contrastive_temp,
            loss_mse_weight=args.loss_mse_weight,
            loss_contrastive_weight=args.loss_contrastive_weight,
            loss_relational_weight=args.loss_relational_weight,
            autocast_ctx=_autocast,
            teacher_autocast_ctx=_teacher_autocast,
            current_lr=cur_lr,
            prev_estimate=(auto_warmup_ema_crit_lr if auto_warmup_ema_crit_lr is not None else auto_warmup_last_crit_lr),
            max_lr_cap=auto_warmup_max_lr,
          )
          if (
            crit_lr_result is not None
            and math.isfinite(crit_lr_result.critical_lr)
            and crit_lr_result.critical_lr > 0
          ):
            beta = float(args.auto_warmup_ema_beta)
            filtered_crit_lr = float(crit_lr_result.critical_lr)
            if auto_warmup_first_crit_lr is None:
              auto_warmup_first_crit_lr = float(filtered_crit_lr)
            auto_warmup_last_crit_lr = float(filtered_crit_lr)
            auto_warmup_last_crit_sharpness = float(crit_lr_result.critical_sharpness)
            current_safe_lr = float(args.auto_warmup_lr_safety_frac) * max(float(filtered_crit_lr), 0.0)
            if auto_warmup_ema_crit_lr is None:
              auto_warmup_ema_crit_lr = float(filtered_crit_lr)
            else:
              auto_warmup_ema_crit_lr = (beta * auto_warmup_ema_crit_lr) + ((1.0 - beta) * float(filtered_crit_lr))
            auto_warmup_recent_crit_points.append((int(step), float(filtered_crit_lr)))
            if len(auto_warmup_recent_crit_points) > 7:
              auto_warmup_recent_crit_points = auto_warmup_recent_crit_points[-7:]
            auto_warmup_last_crit_update_step = int(step)
            forecast_crit_lr = _forecast_terminal_crit_lr(
              auto_warmup_ema_crit_lr,
              auto_warmup_recent_crit_points,
              step=step,
              warmup_steps=args.auto_warmup_steps,
            )
            if (
              forecast_crit_lr is not None
              and auto_warmup_first_crit_lr is not None
              and math.isfinite(auto_warmup_first_crit_lr)
              and auto_warmup_first_crit_lr > 0.0
            ):
              forecast_crit_lr = min(float(forecast_crit_lr), float(auto_warmup_first_crit_lr))
            forecast_safe_lr = float(args.auto_warmup_lr_safety_frac) * max(
              float(forecast_crit_lr if forecast_crit_lr is not None else auto_warmup_ema_crit_lr),
              0.0,
            )
            final_target_update = (step + max(1, int(args.auto_warmup_metric_every))) >= auto_warmup_lr_freeze_step
            if final_target_update and current_safe_lr is not None and math.isfinite(current_safe_lr) and current_safe_lr > 0.0:
              lr_goal = float(current_safe_lr)
            else:
              lr_goal = forecast_safe_lr
            if current_safe_lr is not None and math.isfinite(current_safe_lr) and current_safe_lr > 0.0:
              lr_goal = min(lr_goal, float(current_safe_lr))
            if auto_warmup_max_lr is not None:
              lr_goal = min(lr_goal, auto_warmup_max_lr)
            auto_warmup_last_lr_goal = lr_goal
        lr_goal = auto_warmup_last_lr_goal
        if lr_goal is not None and math.isfinite(lr_goal) and lr_goal > 0.0:
          next_lr = _ramp_warmup_lr(
            float(auto_warmup_current_lr),
            float(lr_goal),
            step=step,
            warmup_steps=args.auto_warmup_steps,
            metric_every=args.auto_warmup_metric_every,
          )
          next_lr = _cap_early_warmup_lr_increase(
            float(auto_warmup_current_lr),
            float(next_lr),
            step=step,
            warmup_steps=args.auto_warmup_steps,
          )
          if current_safe_lr is not None and math.isfinite(current_safe_lr) and current_safe_lr > 0.0:
            next_lr = min(next_lr, float(current_safe_lr))
          auto_warmup_current_lr = float(next_lr)
          cur_lr = float(next_lr)
          _set_lr(optim, cur_lr)
        step_timer.add_ms("aux_ms", (time.perf_counter() - aux_start) * 1000.0)

      step_timer.start("optim_ms")
      scaler.step(optim)
      scaler.update()
      step_timer.stop("optim_ms")
    except RuntimeError as exc:
      effective_teacher_group = _teacher_superbatch_group_count()
      if teacher_superbatch_enabled and _is_cuda_oom(exc) and effective_teacher_group > 1:
        failed_factor = int(teacher_superbatch_runtime_factor)
        teacher_superbatch_runtime_factor = max(1, failed_factor // 2)
        teacher_cache.clear()
        optim.zero_grad(set_to_none=True)
        student.zero_grad(set_to_none=True)
        proj.zero_grad(set_to_none=True)
        if device.type == "cuda":
          torch.cuda.empty_cache()
        print(
          f"teacher_superbatch oom_backoff step={step} failed_factor={failed_factor} "
          f"new_factor={teacher_superbatch_runtime_factor}",
          flush=True,
        )
        continue
      if _is_cuda_oom(exc) and _backoff_batch_opt_multiplier("oom_backoff", failed_batch=int(current_train_batch)):
        optim.zero_grad(set_to_none=True)
        student.zero_grad(set_to_none=True)
        proj.zero_grad(set_to_none=True)
        if device.type == "cuda":
          torch.cuda.empty_cache()
        continue
      raise
    step_perf = step_timer.finish()
    if perf_enabled and perf_window is not None:
      step_perf["step_ms"] = (time.perf_counter() - step_wall_start) * 1000.0
      perf_window.add(step_perf)
    if (step + 1) % args.optimizer_log_every == 0:
      _maybe_backoff_near_oom()
    step += 1
    auto_warmup_last_safe_batch_size = int(current_train_batch)

    if auto_warmup_enabled and step == auto_warmup_lr_freeze_step and auto_warmup_last_lr_goal is not None:
      print(
        f"auto_warmup lr_target_frozen step={step} goal_lr={auto_warmup_last_lr_goal:.3e}",
        flush=True,
      )

    if args.log_every and step % args.log_every == 0:
      elapsed = max(1e-9, time.perf_counter() - t0)
      rate = step / elapsed
      scale = 1.0 / max(1, args.grad_accum)
      total_loss = _scalar_to_float(total_loss_t)
      mse_avg = _scalar_to_float(mse_sum_t) * scale
      con_avg = _scalar_to_float(con_sum_t) * scale
      rel_avg = _scalar_to_float(rel_sum_t) * scale
      cos_avg = _scalar_to_float(cos_sum_t) * scale
      tnorm_avg = _scalar_to_float(tnorm_sum_t) * scale
      snorm_avg = _scalar_to_float(snorm_sum_t) * scale
      mse0_avg = _scalar_to_float(mse0_sum_t) * scale
      ev_avg = _scalar_to_float(ev_sum_t) * scale
      print(
        f"step={step} loss={total_loss:.6f} "
        f"mse={mse_avg:.6f} con={con_avg:.6f} rel={rel_avg:.6f} "
        f"cos={cos_avg:.4f} tnorm={tnorm_avg:.3f} snorm={snorm_avg:.3f} "
        f"mse0={mse0_avg:.6f} ev={ev_avg:.4f} "
        f"lr={cur_lr:.3e} batch={current_train_batch}"
        + (f" (base={base_lr:.3e} gns_fac={lr_gns_factor:.3f})" if lr_gns_controller_active else "")
        + (
          f" aw=1 goal_lr={auto_warmup_last_lr_goal:.3e} cbs~={lr_gns_selected_opt_batch:.1f}"
          + (
            f" mult={batch_opt_runtime_mult:.3f}"
            if batch_opt_runtime_mult > 1.0
            else ""
          )
          + (
            f" step_x~={last_batch_opt_step_speedup:.3f}"
            if last_batch_opt_step_speedup is not None
            else ""
          )
          if auto_warmup_active and auto_warmup_last_lr_goal is not None and lr_gns_selected_opt_batch is not None
          else (" aw=1" if auto_warmup_active else "")
        )
        + (
          f" teacher_fac={teacher_superbatch_runtime_factor}"
          if teacher_superbatch_enabled
          else ""
        )
        + f" steps/s={rate:.2f} elapsed={elapsed:.1f}s",
        flush=True,
      )
      if wandb is not None:
        wandb.log(
          {
            "train/step": step,
            "train/loss": total_loss,
            "train/mse": mse_avg,
            "train/contrastive": con_avg,
            "train/relational": rel_avg,
            "train/cosine": cos_avg,
            "train/teacher_norm": tnorm_avg,
            "train/student_norm": snorm_avg,
            "train/mse_zero": mse0_avg,
            "train/explained_variance": ev_avg,
            "train/lr": cur_lr,
            "train/lr_base": base_lr,
            "train/lr_gns_factor": lr_gns_factor if lr_gns_controller_active else 1.0,
            "train/lr_gns_ref_batch": float(lr_gns_ref_batch),
            "train/lr_gns_mode_sqrt": float(1.0 if lr_gns_mode == "sqrt" else 0.0),
            "train/lr_gns_active": float(1.0 if lr_gns_controller_active else 0.0),
            "train/gns_ema_opt_batch": (lr_gns_ema_opt_batch if lr_gns_ema_opt_batch is not None else float("nan")),
            "train/gns_selected_opt_batch": (
              lr_gns_selected_opt_batch if lr_gns_selected_opt_batch is not None else float("nan")
            ),
            "train/batch_opt_runtime_mult": float(batch_opt_runtime_mult),
            "train/batch_opt_step_speedup": (
              last_batch_opt_step_speedup if last_batch_opt_step_speedup is not None else float("nan")
            ),
            "train/effective_opt_batch": (
              last_effective_opt_batch if last_effective_opt_batch is not None else float("nan")
            ),
            "train/batch_size": float(current_train_batch),
            "train/effective_batch_size": float(current_train_batch * max(1, args.grad_accum)),
            "train/auto_warmup_active": float(1.0 if auto_warmup_active else 0.0),
            "train/auto_warmup_lr_goal": (auto_warmup_last_lr_goal if auto_warmup_last_lr_goal is not None else float("nan")),
            "train/auto_warmup_crit_lr": (auto_warmup_last_crit_lr if auto_warmup_last_crit_lr is not None else float("nan")),
            "train/auto_warmup_crit_sharpness": (
              auto_warmup_last_crit_sharpness if auto_warmup_last_crit_sharpness is not None else float("nan")
            ),
            "train/auto_warmup_batch": float(current_train_batch),
            "train/auto_warmup_cbs": (
              lr_gns_selected_opt_batch if lr_gns_selected_opt_batch is not None else float("nan")
            ),
            "train/auto_warmup_cbs_ema": (lr_gns_ema_opt_batch if lr_gns_ema_opt_batch is not None else float("nan")),
            "train/auto_warmup_handoff_lr": (auto_warmup_handoff_lr if auto_warmup_handoff_lr is not None else float("nan")),
            "train/auto_warmup_handoff_batch": (
              float(auto_warmup_handoff_batch_size)
              if auto_warmup_handoff_batch_size is not None
              else float("nan")
            ),
            "train/optimizer_mode_teacher_superbatch": float(1.0 if teacher_superbatch_enabled else 0.0),
            "train/teacher_batch_factor": float(teacher_superbatch_runtime_factor),
            "train/steps_per_s": rate,
          },
          step=step,
        )

    if perf_enabled and perf_window is not None and step % args.optimizer_log_every == 0:
      perf_means = perf_window.means()
      step_ms = float(perf_means.get("step_ms", float("nan")))
      loader_wait_ms = float(perf_means.get("loader_wait_ms", 0.0))
      h2d_ms = float(perf_means.get("h2d_ms", 0.0))
      preprocess_ms = float(perf_means.get("preprocess_ms", 0.0))
      teacher_ms = float(perf_means.get("teacher_ms", 0.0))
      student_fwd_ms = float(perf_means.get("student_fwd_ms", 0.0))
      backward_ms = float(perf_means.get("backward_ms", 0.0))
      optim_ms = float(perf_means.get("optim_ms", 0.0))
      aux_ms = float(perf_means.get("aux_ms", 0.0))
      input_ms = loader_wait_ms + h2d_ms + preprocess_ms
      student_ms = student_fwd_ms + backward_ms + optim_ms
      bottleneck = _classify_perf_bottleneck(perf_means)
      teacher_pct = (teacher_ms / step_ms) if step_ms > 0.0 else float("nan")
      input_pct = (input_ms / step_ms) if step_ms > 0.0 else float("nan")
      student_pct = (student_ms / step_ms) if step_ms > 0.0 else float("nan")
      print(
        f"perf@step={step} step_ms={step_ms:.2f} "
        f"loader_wait_ms={loader_wait_ms:.2f} h2d_ms={h2d_ms:.2f} preprocess_ms={preprocess_ms:.2f} "
        f"teacher_ms={teacher_ms:.2f} student_fwd_ms={student_fwd_ms:.2f} "
        f"backward_ms={backward_ms:.2f} optim_ms={optim_ms:.2f} aux_ms={aux_ms:.2f} "
        f"teacher_pct={teacher_pct:.3f} input_pct={input_pct:.3f} student_pct={student_pct:.3f} "
        f"bottleneck={bottleneck} distributed_comm=na",
        flush=True,
      )
      if wandb is not None:
        wandb.log(
          {
            "perf/step_ms": step_ms,
            "perf/loader_wait_ms": loader_wait_ms,
            "perf/h2d_ms": h2d_ms,
            "perf/preprocess_ms": preprocess_ms,
            "perf/teacher_ms": teacher_ms,
            "perf/student_fwd_ms": student_fwd_ms,
            "perf/backward_ms": backward_ms,
            "perf/optim_ms": optim_ms,
            "perf/aux_ms": aux_ms,
            "perf/input_ms": input_ms,
            "perf/student_ms": student_ms,
            "perf/teacher_pct": teacher_pct,
            "perf/input_pct": input_pct,
            "perf/student_pct": student_pct,
            "perf/bottleneck_teacher": float(1.0 if bottleneck == "teacher_bound" else 0.0),
            "perf/bottleneck_input": float(1.0 if bottleneck == "input_bound" else 0.0),
            "perf/bottleneck_student": float(1.0 if bottleneck == "student_bound" else 0.0),
            "perf/distributed_comm_applicable": 0.0,
            "perf/input_comm_limited": float(1.0 if bottleneck == "input_bound" else 0.0),
          },
          step=step,
        )
      perf_window.reset()

    if auto_warmup_active and step % args.auto_warmup_metric_every == 0:
      gns_metrics, data_iter = _estimate_gns(
        loader=loader,
        data_iter=data_iter,
        preprocess_audio=preprocess_audio,
        teacher=teacher,
        student=student,
        proj=proj,
        device=device,
        trainable_params=params,
        batch_size=current_train_batch,
        gns_param_sample=args.gns_param_sample,
        contrastive_temp=args.contrastive_temp,
        loss_mse_weight=args.loss_mse_weight,
      loss_contrastive_weight=args.loss_contrastive_weight,
      loss_relational_weight=args.loss_relational_weight,
      autocast_ctx=_autocast,
      teacher_autocast_ctx=_teacher_autocast,
    )
      if gns_metrics is not None:
        gns_opt_batch = float(gns_metrics.get("gns_opt_batch", float("nan")))
        if math.isfinite(gns_opt_batch) and gns_opt_batch > 0:
          lr_gns_recent_opt_batches = _sanitize_positive_float_window(
            list(lr_gns_recent_opt_batches) + [gns_opt_batch],
            max_len=int(args.gns_batch_window),
          )
          if lr_gns_ema_opt_batch is None:
            lr_gns_ema_opt_batch = gns_opt_batch
          else:
            beta = float(args.auto_warmup_cbs_ema_beta)
            lr_gns_ema_opt_batch = (beta * lr_gns_ema_opt_batch) + ((1.0 - beta) * gns_opt_batch)
          lr_gns_selected_opt_batch = _select_cbs_from_recent_window(
            lr_gns_recent_opt_batches,
            target_utility=float(args.gns_batch_target_utility),
          )
          _update_batch_opt_diagnostics()
          lr_gns_samples += 1
          if lr_gns_selected_opt_batch is not None and lr_gns_selected_opt_batch > 0:
            batch_goal = _current_batch_goal_from_selected()
            if batch_goal is None:
              batch_goal = _round_batch_size_down(lr_gns_selected_opt_batch, args.auto_warmup_batch_round_to)
              if auto_warmup_gpu_batch_cap is not None:
                batch_goal = min(batch_goal, int(auto_warmup_gpu_batch_cap))
              if auto_warmup_max_batch_size is not None:
                batch_goal = min(batch_goal, int(auto_warmup_max_batch_size))
              batch_goal = max(1, batch_goal)
            auto_warmup_last_batch_goal = float(batch_goal)
        print(
          f"auto_warmup@step={step} crit_lr~="
          f"{(auto_warmup_last_crit_lr if auto_warmup_last_crit_lr is not None else float('nan')):.3e} "
          f"sharpness~="
          f"{(auto_warmup_last_crit_sharpness if auto_warmup_last_crit_sharpness is not None else float('nan')):.3e} "
          f"lr_goal={((auto_warmup_last_lr_goal if auto_warmup_last_lr_goal is not None else float('nan'))):.3e} "
          f"lr={auto_warmup_current_lr:.3e} "
          f"cbs~={gns_metrics['gns_opt_batch']:.1f} batch={current_train_batch}"
          + (
            f" cbs_sel~={lr_gns_selected_opt_batch:.1f}"
            if lr_gns_selected_opt_batch is not None
            else ""
          )
          + (
            f" cbs_eff~={last_effective_opt_batch:.1f}"
            if last_effective_opt_batch is not None
            else ""
          )
          + (
            f" cbs_ema~={lr_gns_ema_opt_batch:.1f}"
            if lr_gns_ema_opt_batch is not None
            else ""
          )
          + (
            f" step_x~={last_batch_opt_step_speedup:.3f}"
            if last_batch_opt_step_speedup is not None
            else ""
          )
          + (
            f" batch_goal={auto_warmup_last_batch_goal:.1f}"
            if auto_warmup_last_batch_goal is not None
            else ""
          )
          + f" nsr={gns_metrics['gns_nsr']:.4f}",
          flush=True,
        )
        if wandb is not None:
          wandb.log(
            {
              "train/auto_warmup_cbs_raw": gns_metrics["gns_opt_batch"],
              "train/auto_warmup_cbs": (
                lr_gns_selected_opt_batch if lr_gns_selected_opt_batch is not None else float("nan")
              ),
              "train/auto_warmup_cbs_selected": (
                lr_gns_selected_opt_batch if lr_gns_selected_opt_batch is not None else float("nan")
              ),
              "train/auto_warmup_cbs_effective": (
                last_effective_opt_batch if last_effective_opt_batch is not None else float("nan")
              ),
              "train/auto_warmup_cbs_ema": (lr_gns_ema_opt_batch if lr_gns_ema_opt_batch is not None else float("nan")),
              "train/auto_warmup_batch_opt_mult": float(batch_opt_runtime_mult),
              "train/auto_warmup_step_speedup": (
                last_batch_opt_step_speedup if last_batch_opt_step_speedup is not None else float("nan")
              ),
              "train/auto_warmup_batch": float(current_train_batch),
              "train/auto_warmup_batch_goal": (
                auto_warmup_last_batch_goal if auto_warmup_last_batch_goal is not None else float("nan")
              ),
              "train/auto_warmup_crit_lr": (auto_warmup_last_crit_lr if auto_warmup_last_crit_lr is not None else float("nan")),
              "train/auto_warmup_crit_sharpness": (
                auto_warmup_last_crit_sharpness if auto_warmup_last_crit_sharpness is not None else float("nan")
              ),
              "train/auto_warmup_nsr": gns_metrics["gns_nsr"],
            },
            step=step,
          )
    elif args.gns_every and step % args.gns_every == 0:
      gns_metrics, data_iter = _estimate_gns(
        loader=loader,
        data_iter=data_iter,
        preprocess_audio=preprocess_audio,
        teacher=teacher,
        student=student,
        proj=proj,
        device=device,
        trainable_params=params,
        batch_size=current_train_batch,
        gns_param_sample=args.gns_param_sample,
        contrastive_temp=args.contrastive_temp,
        loss_mse_weight=args.loss_mse_weight,
        loss_contrastive_weight=args.loss_contrastive_weight,
        loss_relational_weight=args.loss_relational_weight,
        autocast_ctx=_autocast,
        teacher_autocast_ctx=_teacher_autocast,
      )
      if gns_metrics is not None:
        gns_opt_batch = float(gns_metrics.get("gns_opt_batch", float("nan")))
        gns_adapted = False
        gns_raw_factor = float("nan")
        if math.isfinite(gns_opt_batch) and gns_opt_batch > 0:
          lr_gns_recent_opt_batches = _sanitize_positive_float_window(
            list(lr_gns_recent_opt_batches) + [gns_opt_batch],
            max_len=int(args.gns_batch_window),
          )
          if lr_gns_ema_opt_batch is None:
            lr_gns_ema_opt_batch = gns_opt_batch
          else:
            beta = float(args.lr_gns_ema_beta)
            lr_gns_ema_opt_batch = (beta * lr_gns_ema_opt_batch) + ((1.0 - beta) * gns_opt_batch)
          lr_gns_selected_opt_batch = _select_cbs_from_recent_window(
            lr_gns_recent_opt_batches,
            target_utility=float(args.gns_batch_target_utility),
          )
          _update_batch_opt_diagnostics()
          lr_gns_samples += 1

          if (
            lr_gns_controller_active
            and lr_gns_samples >= args.lr_gns_min_samples
            and (step - lr_gns_last_update_step) >= args.lr_gns_update_every
            and lr_gns_selected_opt_batch is not None
            and lr_gns_selected_opt_batch > 0
          ):
            gns_raw_factor = math.sqrt(lr_gns_ref_batch / max(lr_gns_selected_opt_batch, 1e-8))
            if math.isfinite(gns_raw_factor):
              lr_gns_factor = float(
                min(
                  float(args.lr_gns_max_factor),
                  max(float(args.lr_gns_min_factor), gns_raw_factor),
                )
              )
              lr_gns_last_update_step = step
              gns_adapted = True

        print(
          f"gns@step={step} opt_batch~={gns_metrics['gns_opt_batch']:.1f} "
          f"nsr={gns_metrics['gns_nsr']:.4f} "
          f"noise={gns_metrics['gns_noise_batch']:.3e} signal={gns_metrics['gns_signal']:.3e}"
          + (
            f" sel={lr_gns_selected_opt_batch:.1f} "
            + (
              f"eff={last_effective_opt_batch:.1f} "
              if last_effective_opt_batch is not None
              else ""
            )
            + (f"ema={lr_gns_ema_opt_batch:.1f} " if lr_gns_ema_opt_batch is not None else "")
            + f"samples={lr_gns_samples} mult={batch_opt_runtime_mult:.3f} "
            f"lr_fac={lr_gns_factor:.3f}"
            + (
              f" step_x~={last_batch_opt_step_speedup:.3f}"
              if last_batch_opt_step_speedup is not None
              else ""
            )
            + (
              f" raw={gns_raw_factor:.3f} updated={int(gns_adapted)}"
              if lr_gns_controller_active and lr_gns_samples >= args.lr_gns_min_samples
              else ""
            )
            if lr_gns_selected_opt_batch is not None
            else ""
          ),
          flush=True,
        )
        if wandb is not None:
          wandb.log(
            {
              "train/gns_opt_batch": gns_metrics["gns_opt_batch"],
              "train/gns_nsr": gns_metrics["gns_nsr"],
              "train/gns_noise_batch": gns_metrics["gns_noise_batch"],
              "train/gns_signal": gns_metrics["gns_signal"],
              "train/gns_loss": gns_metrics["gns_loss"],
              "train/gns_ema_opt_batch": (lr_gns_ema_opt_batch if lr_gns_ema_opt_batch is not None else float("nan")),
              "train/gns_selected_opt_batch": (
                lr_gns_selected_opt_batch if lr_gns_selected_opt_batch is not None else float("nan")
              ),
              "train/gns_effective_opt_batch": (
                last_effective_opt_batch if last_effective_opt_batch is not None else float("nan")
              ),
              "train/batch_opt_runtime_mult": float(batch_opt_runtime_mult),
              "train/batch_opt_step_speedup": (
                last_batch_opt_step_speedup if last_batch_opt_step_speedup is not None else float("nan")
              ),
              "train/gns_samples": float(lr_gns_samples),
              "train/lr_gns_factor": lr_gns_factor if lr_gns_controller_active else 1.0,
              "train/lr_gns_ref_batch": float(lr_gns_ref_batch),
              "train/lr_gns_mode_sqrt": float(1.0 if lr_gns_mode == "sqrt" else 0.0),
              "train/lr_gns_active": float(1.0 if lr_gns_controller_active else 0.0),
              "train/lr_gns_updated": float(1.0 if gns_adapted else 0.0),
            },
            step=step,
          )

    if auto_warmup_enabled and step == args.auto_warmup_steps:
      auto_warmup_handoff_lr = float(auto_warmup_current_lr)
      handoff_batch_goal = _current_batch_goal_from_selected()
      if handoff_batch_goal is not None and handoff_batch_goal != current_train_batch:
        auto_warmup_last_safe_batch_size = int(current_train_batch)
        current_train_batch = int(handoff_batch_goal)
        _refresh_runtime_loaders("auto_warmup_handoff")
      auto_warmup_handoff_batch_size = int(current_train_batch)
      _enable_student_compile("post_warmup")
      print(
        f"auto_warmup handoff step={step} lr={auto_warmup_handoff_lr:.3e} "
        f"batch={auto_warmup_handoff_batch_size}",
        flush=True,
      )

    if val_loader is not None and args.val_every and step % args.val_every == 0:
      student.eval()
      proj.eval()
      teacher.eval()
      with torch.inference_mode():
        v_loss = 0.0
        v_mse = 0.0
        v_con = 0.0
        v_rel = 0.0
        v_cos = 0.0
        v_tnorm = 0.0
        v_snorm = 0.0
        v_mse0 = 0.0
        v_ev = 0.0
        v_batches = 0
        for _ in range(args.val_batches):
          if val_iter is None:
            break
          try:
            v_batch = next(val_iter)
          except StopIteration:
            val_iter = iter(val_loader)
            v_batch = next(val_iter)
          v_batch = v_batch.to(device, non_blocking=True)
          v_spec = preprocess_audio(v_batch)
          v_target = _teacher_targets_from_spec(
            teacher,
            v_spec,
            teacher_autocast_ctx=_teacher_autocast,
          )
          with _autocast():
            v_total, v_loss_mse, v_loss_con, v_loss_rel, v_student = student_loss_runner(v_spec, v_target)
          t_f = v_target.float()
          s_f = v_student.float()
          v_cos += float(F.cosine_similarity(s_f, t_f, dim=-1).mean().detach().cpu())
          v_tnorm += float(t_f.norm(dim=-1).mean().detach().cpu())
          v_snorm += float(s_f.norm(dim=-1).mean().detach().cpu())
          v_mse0 += float((t_f * t_f).mean().detach().cpu())
          v_var = t_f.var(unbiased=False)
          v_ev += float((1.0 - (v_loss_mse.float() / (v_var + 1e-8))).detach().cpu())
          v_loss += float(v_total.detach().cpu())
          v_mse += float(v_loss_mse.detach().cpu())
          v_con += float(v_loss_con.detach().cpu())
          v_rel += float(v_loss_rel.detach().cpu())
          v_batches += 1
        if v_batches > 0:
          v_loss /= v_batches
          v_mse /= v_batches
          v_con /= v_batches
          v_rel /= v_batches
          v_cos /= v_batches
          v_tnorm /= v_batches
          v_snorm /= v_batches
          v_mse0 /= v_batches
          v_ev /= v_batches
          print(
            f"val@step={step} loss={v_loss:.6f} mse={v_mse:.6f} con={v_con:.6f} rel={v_rel:.6f} "
            f"cos={v_cos:.4f} tnorm={v_tnorm:.3f} snorm={v_snorm:.3f} "
            f"mse0={v_mse0:.6f} ev={v_ev:.4f}",
            flush=True,
          )
          if wandb is not None:
            wandb.log(
              {
                "val/step": step,
                "val/loss": v_loss,
                "val/mse": v_mse,
                "val/contrastive": v_con,
                "val/relational": v_rel,
                "val/cosine": v_cos,
                "val/teacher_norm": v_tnorm,
                "val/student_norm": v_snorm,
                "val/mse_zero": v_mse0,
                "val/explained_variance": v_ev,
              },
              step=step,
            )
      student.train()
      proj.train()

    if args.save_every and step % args.save_every == 0:
      _save_ckpt(f"{step:06d}")

  _save_ckpt("final")
  if wandb is not None:
    wandb.finish()


if __name__ == "__main__":
  main()
