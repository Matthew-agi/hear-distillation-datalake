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
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _die(msg: str) -> "None":
  raise SystemExit(msg)


def _import_preprocess_audio(repo_root: Path):
  import sys

  import importlib

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
        f = tf.extractfile(member)
        if f is None:
          continue
        try:
          data = f.read()
        except Exception:
          # Most commonly truncated tar shards; skip remaining entries in this shard.
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


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Distill HeAR into a ViT-S student.")
  ap.add_argument("--data-dir", type=Path, default=Path("data/laion_audio_2s"), help="Directory with shard-*.tar files.")
  ap.add_argument("--shards-glob", type=str, default="shard-*.tar", help="Glob pattern for shards.")
  ap.add_argument("--streams-glob", type=str, default="stream-*", help="Glob for stream subfolders inside data-dir.")
  ap.add_argument("--out", type=Path, default=Path("checkpoints/hear_vit_s"), help="Output/checkpoint directory.")
  ap.add_argument("--max-steps", type=int, default=20000, help="Number of training steps.")
  ap.add_argument("--batch-size", type=int, default=64, help="Batch size.")
  ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
  ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
  ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device.")
  ap.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
  ap.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"], help="LR schedule.")
  ap.add_argument("--lr-warmup-steps", type=int, default=0, help="Linear warmup steps for LR.")
  ap.add_argument("--lr-min-ratio", type=float, default=0.1, help="Final LR ratio for cosine schedule.")
  ap.add_argument("--lr-gns-adapt", action="store_true", help="Adapt LR using GNS-estimated optimal batch size.")
  ap.add_argument("--lr-gns-ema-beta", type=float, default=0.99, help="EMA beta for GNS optimal batch smoothing.")
  ap.add_argument("--lr-gns-min-samples", type=int, default=20, help="Minimum GNS samples before LR adaptation starts.")
  ap.add_argument("--lr-gns-update-every", type=int, default=50, help="LR adaptation cadence in steps.")
  ap.add_argument("--lr-gns-min-factor", type=float, default=0.1, help="Minimum LR factor from GNS adaptation.")
  ap.add_argument("--lr-gns-max-factor", type=float, default=1.0, help="Maximum LR factor from GNS adaptation.")
  ap.add_argument("--lr-gns-ref-batch", type=float, default=0.0, help="Reference batch for LR adaptation (<=0 uses batch_size*grad_accum).")
  ap.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay.")
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
  ap.add_argument("--gns-every", type=int, default=5, help="Estimate gradient noise scale every N steps (0 disables).")
  ap.add_argument("--gns-param-sample", type=int, default=200000, help="Max gradient elements to sample for GNS estimate.")
  ap.add_argument("--teacher-id", type=str, default="google/hear-pytorch", help="Teacher model id.")
  ap.add_argument("--clip-seconds", type=float, default=2.0, help="Clip length in seconds.")
  ap.add_argument("--sample-rate", type=int, default=16000, help="Sample rate.")
  ap.add_argument("--shuffle-shards", action="store_true", help="Shuffle shard order per epoch.")
  ap.add_argument("--repeat", action="store_true", help="Repeat over shards indefinitely (recommended).")
  ap.add_argument("--live-shard-refresh", action="store_true", help="Refresh shard list while training to ingest newly written shards.")
  ap.add_argument("--shard-refresh-sec", type=float, default=30.0, help="Seconds between shard list refreshes when --live-shard-refresh is enabled.")
  ap.add_argument("--amp", action="store_true", help="Use AMP (cuda only).")
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
  try:
    import timm
  except Exception as exc:  # noqa: BLE001
    _die(f"timm is required for ViT-S student: {exc}")

  # ViT-S backbone, 1-channel input, 192x128 spectrograms.
  try:
    model = timm.create_model(
      "vit_small_patch16_224",
      img_size=(192, 128),
      in_chans=1,
      num_classes=0,
      global_pool="avg",
    )
  except Exception:
    # Fallback: use default size; the caller can resize if needed.
    model = timm.create_model(
      "vit_small_patch16_224",
      in_chans=1,
      num_classes=0,
      global_pool="avg",
    )

  if use_canon:
    dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    if dim is None:
      _die("Could not determine student embed_dim for Canon layers.")
    if hasattr(model, "blocks"):
      try:
        grid_size = None
        expect_cls = None
        if canon_2d:
          patch_embed = getattr(model, "patch_embed", None)
          grid_size = getattr(patch_embed, "grid_size", None) if patch_embed is not None else None
          if grid_size is None:
            print("Warning: --canon-2d requested but model has no patch_embed.grid_size; falling back to 1D Canon.", flush=True)
          num_prefix = getattr(model, "num_prefix_tokens", None)
          if num_prefix is not None:
            try:
              num_prefix = int(num_prefix)
            except Exception:
              num_prefix = None
            if num_prefix in (0, 1):
              expect_cls = bool(num_prefix)
            elif num_prefix is not None:
              print(
                f"Warning: Canon2D only supports 0/1 prefix tokens but model reports {num_prefix}; "
                "falling back to 1D Canon.",
                flush=True,
              )
          if expect_cls is None:
            expect_cls = getattr(model, "cls_token", None) is not None
        for i in range(len(model.blocks)):
          model.blocks[i] = CanonBlockWrapper(
            model.blocks[i],
            int(dim),
            kernel_size=canon_kernel,
            canon_a=canon_a,
            canon_b=canon_b,
            canon_b_qkv=canon_b_qkv,
            canon_c=canon_c,
            canon_d=canon_d,
            causal=canon_causal,
            use_2d=canon_2d,
            grid_size=grid_size,
            expect_cls=expect_cls,
          )
      except Exception as exc:  # noqa: BLE001
        _die(f"Failed to insert Canon layers: {exc}")
    else:
      _die("Student model has no `.blocks` attribute; cannot insert Canon layers.")

  if canon_no_pos_enc:
    if use_canon:
      _disable_positional_embeddings(model)
    else:
      print("Warning: --canon-no-pos-enc set but Canon is disabled; ignoring.", flush=True)

  return model


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
  return 1.0


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
    with torch.no_grad():
      spec = preprocess_audio(batch)
      target = teacher(spec, return_dict=True).pooler_output.detach()
    student.zero_grad(set_to_none=True)
    proj.zero_grad(set_to_none=True)
    with autocast_ctx():
      student_feats = _student_features(student, spec)
      student_emb = proj(student_feats)
      loss_mse = F.mse_loss(student_emb, target)
      loss_con = _contrastive_loss(student_emb, target, temperature=contrastive_temp)
      loss_rel = _relational_loss(student_emb, target)
      loss_total = (
        loss_mse_weight * loss_mse
        + loss_contrastive_weight * loss_con
        + loss_relational_weight * loss_rel
      )
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
    grads.append(g)
    losses.append(float(loss_total.detach().cpu()))

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
  nsr = noise_batch / max(signal, 1e-12)
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
  if args.lr_warmup_steps < 0:
    _die("--lr-warmup-steps must be >= 0.")
  if not (0.0 <= args.lr_min_ratio <= 1.0):
    _die("--lr-min-ratio must be in [0, 1].")
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
  if args.max_checkpoints < 0:
    _die("--max-checkpoints must be >= 0.")
  if args.gns_every < 0:
    _die("--gns-every must be >= 0.")
  if args.gns_param_sample <= 0:
    _die("--gns-param-sample must be > 0.")
  if args.lr_gns_adapt and args.gns_every <= 0:
    _die("--lr-gns-adapt requires --gns-every > 0.")
  if args.resume_from is not None and args.resume_latest:
    _die("Use only one of --resume-from or --resume-latest.")

  data_dir = args.data_dir
  shards = _discover_shards(data_dir, args.shards_glob, args.streams_glob)
  if not shards:
    _die(f"No shards found in {data_dir} matching {args.shards_glob}")
  val_enabled = bool(args.val_fraction > 0.0 or args.val_target_clips > 0)
  if len(shards) < 2 and val_enabled:
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

  repo_root = Path(__file__).resolve().parent
  preprocess_audio = _import_preprocess_audio(repo_root)

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
  print("=== Teacher Model ===", flush=True)
  print(teacher, flush=True)
  print("=== Student Model ===", flush=True)
  if use_canon:
    canon_b_mode = "qkv" if getattr(args, "canon_b_qkv", False) else "post-attn"
    pos_enc_mode = "off" if getattr(args, "canon_no_pos_enc", False) else "on"
    print(
      f"(Canon enabled: A={canon_a}, B={canon_b}({canon_b_mode}), C={canon_c}, D={canon_d}, "
      f"kernel={args.canon_kernel}, causal={args.canon_causal}, 2d={bool(getattr(args, 'canon_2d', False))}, "
      f"pos_enc={pos_enc_mode})",
      flush=True,
    )
  print(student, flush=True)
  print("=== Student Projection ===", flush=True)
  print(proj, flush=True)

  # Optimizer
  params = list(student.parameters()) + list(proj.parameters())
  optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
  if args.amp and device.type == "cuda":
    try:
      scaler = torch.amp.GradScaler("cuda")
      def _autocast():
        return torch.amp.autocast(device_type="cuda")
    except Exception:
      scaler = torch.cuda.amp.GradScaler()
      def _autocast():
        return torch.cuda.amp.autocast()
  else:
    scaler = _NoopScaler()
    def _autocast():
      return contextlib.nullcontext()

  resume_ckpt_path: Optional[Path] = None
  if args.resume_from is not None:
    resume_ckpt_path = args.resume_from
  elif args.resume_latest:
    resume_ckpt_path = _find_latest_checkpoint(args.out)
    if resume_ckpt_path is None:
      _die(f"--resume-latest requested, but no checkpoint found in {args.out}.")

  lr_gns_ref_batch = float(args.lr_gns_ref_batch) if args.lr_gns_ref_batch > 0 else float(args.batch_size * max(1, args.grad_accum))
  lr_gns_ema_opt_batch: Optional[float] = None
  lr_gns_samples = 0
  lr_gns_factor = 1.0
  lr_gns_last_update_step = 0
  step = 0

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
    print(f"Resumed training from {Path(resume_ckpt_path)} at step={step}.", flush=True)

  # Data
  rng = random.Random(args.seed)
  shard_list = list(shards)
  rng.shuffle(shard_list)
  clip_count_cache: Dict[Path, int] = {}
  val_shards: List[Path] = []
  val_clip_total = 0
  deferred_val_setup = False
  if val_enabled:
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
  if val_enabled and not deferred_val_setup and not train_shards:
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
    + (f" (val_clips~{val_clip_total})" if val_shards else (" (deferred)" if deferred_val_setup else "")),
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
  def _build_train_loader(excluded_shards: List[Path]) -> Tuple[ClipDataset, DataLoader]:
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
    ld = DataLoader(
      ds,
      batch_size=args.batch_size,
      num_workers=args.num_workers,
      pin_memory=(device.type == "cuda"),
      drop_last=True,
      prefetch_factor=2,
    )
    return ds, ld

  def _build_val_loader(selected_val_shards: List[Path]) -> Tuple[ClipDataset, DataLoader]:
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
    ld = DataLoader(
      ds,
      batch_size=args.batch_size,
      num_workers=max(1, args.num_workers // 2),
      pin_memory=(device.type == "cuda"),
      drop_last=True,
      prefetch_factor=2,
    )
    return ds, ld

  dataset, loader = _build_train_loader(val_shards)
  val_dataset = None
  val_loader = None
  if val_shards:
    val_dataset, val_loader = _build_val_loader(val_shards)

  # Training
  out_dir = args.out
  out_dir.mkdir(parents=True, exist_ok=True)
  t0 = time.perf_counter()
  if args.lr_gns_adapt:
    print(
      "LR GNS adapt enabled: "
      f"ref_batch={lr_gns_ref_batch:.1f} beta={args.lr_gns_ema_beta} "
      f"min_samples={args.lr_gns_min_samples} update_every={args.lr_gns_update_every} "
      f"factor_range=[{args.lr_gns_min_factor}, {args.lr_gns_max_factor}]",
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
        "ema_opt_batch": lr_gns_ema_opt_batch,
        "samples": int(lr_gns_samples),
        "factor": float(lr_gns_factor),
        "last_update_step": int(lr_gns_last_update_step),
        "ref_batch": float(lr_gns_ref_batch),
      },
      "args": vars(args),
    }
    torch.save(ckpt, out_dir / f"ckpt_{tag}.pt")
    if tag != "final" and args.max_checkpoints > 0:
      removed = _prune_old_step_checkpoints(out_dir, args.max_checkpoints)
      if removed > 0:
        print(f"ckpt_prune removed={removed} keep={args.max_checkpoints}", flush=True)

  data_iter = iter(loader)
  val_iter = iter(val_loader) if val_loader is not None else None
  while step < args.max_steps:
    if (
      deferred_val_setup
      and val_enabled
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
          dataset, loader = _build_train_loader(val_shards)
          data_iter = iter(loader)
          val_dataset, val_loader = _build_val_loader(val_shards)
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

    base_lr = args.lr * _lr_multiplier(
      step=step + 1,
      max_steps=args.max_steps,
      schedule=args.lr_schedule,
      warmup_steps=args.lr_warmup_steps,
      min_ratio=args.lr_min_ratio,
    )
    cur_lr = base_lr * (lr_gns_factor if args.lr_gns_adapt else 1.0)
    _set_lr(optim, cur_lr)
    optim.zero_grad(set_to_none=True)
    total_loss = 0.0
    mse_sum = 0.0
    con_sum = 0.0
    rel_sum = 0.0
    cos_sum = 0.0
    tnorm_sum = 0.0
    snorm_sum = 0.0
    mse0_sum = 0.0
    ev_sum = 0.0

    for _ in range(args.grad_accum):
      batch, data_iter = _next_batch(loader, data_iter)

      batch = batch.to(device, non_blocking=True)
      with torch.no_grad():
        spec = preprocess_audio(batch)
        teacher_out = teacher(spec, return_dict=True)
        target = teacher_out.pooler_output.detach()

      with _autocast():
        student_feats = _student_features(student, spec)
        student_emb = proj(student_feats)
        loss_mse = torch.nn.functional.mse_loss(student_emb, target)
        loss_con = _contrastive_loss(student_emb, target, temperature=args.contrastive_temp)
        loss_rel = _relational_loss(student_emb, target)
        loss = (
          args.loss_mse_weight * loss_mse
          + args.loss_contrastive_weight * loss_con
          + args.loss_relational_weight * loss_rel
        ) / args.grad_accum

      with torch.no_grad():
        t_f = target.float()
        s_f = student_emb.float()
        tnorm = t_f.norm(dim=-1).mean()
        snorm = s_f.norm(dim=-1).mean()
        cos = F.cosine_similarity(s_f, t_f, dim=-1).mean()
        mse0 = (t_f * t_f).mean()
        var = t_f.var(unbiased=False)
        ev = 1.0 - (loss_mse.float() / (var + 1e-8))

      scaler.scale(loss).backward()
      total_loss += float(loss.detach().cpu())
      mse_sum += float(loss_mse.detach().cpu())
      con_sum += float(loss_con.detach().cpu())
      rel_sum += float(loss_rel.detach().cpu())
      cos_sum += float(cos.detach().cpu())
      tnorm_sum += float(tnorm.detach().cpu())
      snorm_sum += float(snorm.detach().cpu())
      mse0_sum += float(mse0.detach().cpu())
      ev_sum += float(ev.detach().cpu())

    scaler.step(optim)
    scaler.update()
    step += 1

    if args.log_every and step % args.log_every == 0:
      elapsed = max(1e-9, time.perf_counter() - t0)
      rate = step / elapsed
      scale = 1.0 / max(1, args.grad_accum)
      mse_avg = mse_sum * scale
      con_avg = con_sum * scale
      rel_avg = rel_sum * scale
      cos_avg = cos_sum * scale
      tnorm_avg = tnorm_sum * scale
      snorm_avg = snorm_sum * scale
      mse0_avg = mse0_sum * scale
      ev_avg = ev_sum * scale
      print(
        f"step={step} loss={total_loss:.6f} "
        f"mse={mse_avg:.6f} con={con_avg:.6f} rel={rel_avg:.6f} "
        f"cos={cos_avg:.4f} tnorm={tnorm_avg:.3f} snorm={snorm_avg:.3f} "
        f"mse0={mse0_avg:.6f} ev={ev_avg:.4f} "
        f"lr={cur_lr:.3e}"
        + (f" (base={base_lr:.3e} gns_fac={lr_gns_factor:.3f})" if args.lr_gns_adapt else "")
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
            "train/lr_gns_factor": lr_gns_factor if args.lr_gns_adapt else 1.0,
            "train/steps_per_s": rate,
          },
          step=step,
        )

    if args.gns_every and step % args.gns_every == 0:
      gns_metrics, data_iter = _estimate_gns(
        loader=loader,
        data_iter=data_iter,
        preprocess_audio=preprocess_audio,
        teacher=teacher,
        student=student,
        proj=proj,
        device=device,
        trainable_params=params,
        batch_size=args.batch_size,
        gns_param_sample=args.gns_param_sample,
        contrastive_temp=args.contrastive_temp,
        loss_mse_weight=args.loss_mse_weight,
        loss_contrastive_weight=args.loss_contrastive_weight,
        loss_relational_weight=args.loss_relational_weight,
        autocast_ctx=_autocast,
      )
      if gns_metrics is not None:
        gns_opt_batch = float(gns_metrics.get("gns_opt_batch", float("nan")))
        gns_adapted = False
        gns_raw_factor = float("nan")
        if math.isfinite(gns_opt_batch) and gns_opt_batch > 0:
          if lr_gns_ema_opt_batch is None:
            lr_gns_ema_opt_batch = gns_opt_batch
          else:
            beta = float(args.lr_gns_ema_beta)
            lr_gns_ema_opt_batch = (beta * lr_gns_ema_opt_batch) + ((1.0 - beta) * gns_opt_batch)
          lr_gns_samples += 1

          if (
            args.lr_gns_adapt
            and lr_gns_samples >= args.lr_gns_min_samples
            and (step - lr_gns_last_update_step) >= args.lr_gns_update_every
            and lr_gns_ema_opt_batch is not None
            and lr_gns_ema_opt_batch > 0
          ):
            gns_raw_factor = lr_gns_ref_batch / max(lr_gns_ema_opt_batch, 1e-8)
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
            f" ema={lr_gns_ema_opt_batch:.1f} samples={lr_gns_samples} "
            f"lr_fac={lr_gns_factor:.3f}"
            + (
              f" raw={gns_raw_factor:.3f} updated={int(gns_adapted)}"
              if args.lr_gns_adapt and lr_gns_samples >= args.lr_gns_min_samples
              else ""
            )
            if lr_gns_ema_opt_batch is not None
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
              "train/gns_samples": float(lr_gns_samples),
              "train/lr_gns_factor": lr_gns_factor if args.lr_gns_adapt else 1.0,
              "train/lr_gns_updated": float(1.0 if gns_adapted else 0.0),
            },
            step=step,
          )

    if val_loader is not None and args.val_every and step % args.val_every == 0:
      student.eval()
      proj.eval()
      teacher.eval()
      with torch.no_grad():
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
          v_target = teacher(v_spec, return_dict=True).pooler_output
          v_student = proj(_student_features(student, v_spec))
          v_loss_mse = F.mse_loss(v_student, v_target)
          v_loss_con = _contrastive_loss(v_student, v_target, temperature=args.contrastive_temp)
          v_loss_rel = _relational_loss(v_student, v_target)
          v_total = (
            args.loss_mse_weight * v_loss_mse
            + args.loss_contrastive_weight * v_loss_con
            + args.loss_relational_weight * v_loss_rel
          )
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
