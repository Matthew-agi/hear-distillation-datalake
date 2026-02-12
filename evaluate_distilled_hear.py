#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate a distilled HeAR student on HF datasets used in the HeAR paper.

Default datasets:
  - FSD50K: CLAPv2/FSD50K (labels inferred from text field)
  - FluSense: vtsouval/flusense
  - Coswara: szzs1693/coswara-data
"""

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _die(msg: str) -> "None":
  raise SystemExit(msg)


def _set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)


def _import_audio_utils(repo_root: Path):
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
      return importlib.import_module("hear.python.data_processing.audio_utils")
    except Exception as exc:  # noqa: BLE001
      last_exc = exc
      continue

  roots_txt = ", ".join(str(p / "hear") for p in checked)
  _die(
    "Failed to import `hear.python.data_processing.audio_utils`.\n"
    "Expected a cloned `hear` repo at one of: "
    f"{roots_txt}\n"
    "Install dependencies (torch/scipy/numpy) and retry.\n"
    f"Original error: {last_exc}"
  )


def _import_preprocess_audio(repo_root: Path):
  audio_utils = _import_audio_utils(repo_root)
  if not hasattr(audio_utils, "preprocess_audio"):
    _die("`audio_utils` import succeeded but has no `preprocess_audio` attribute.")
  return audio_utils.preprocess_audio


def _import_preprocess_audio_full_clip(repo_root: Path):
  audio_utils = _import_audio_utils(repo_root)
  mel_pcen = getattr(audio_utils, "_mel_pcen", None)
  resize = getattr(audio_utils, "_torch_resize_bilinear_tf_compat", None)
  if mel_pcen is None or resize is None:
    _die(
      "Could not load full-clip preprocessing helpers from "
      "`hear.python.data_processing.audio_utils`."
    )

  def _preprocess_audio_full_clip(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim != 2:
      raise ValueError(f"Input audio must have rank 2, got rank {audio.ndim}")
    if audio.shape[1] <= 0:
      audio = F.pad(audio, (0, 1))
    spectrogram = mel_pcen(audio.float())
    spectrogram = torch.unsqueeze(spectrogram, dim=1)
    return resize(spectrogram, size=(192, 128))

  return _preprocess_audio_full_clip


def _pick_device(requested: str) -> Tuple[str, torch.device]:
  if requested == "cpu":
    return "cpu", torch.device("cpu")
  if requested == "cuda":
    if not torch.cuda.is_available():
      _die("Requested CUDA, but torch.cuda.is_available() is False.")
    return "cuda", torch.device("cuda")
  if requested == "mps":
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
      _die("Requested MPS, but torch.backends.mps.is_available() is False.")
    return "mps", torch.device("mps")
  if torch.cuda.is_available():
    return "cuda", torch.device("cuda")
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return "mps", torch.device("mps")
  return "cpu", torch.device("cpu")


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


class CanonLayer(nn.Module):
  def __init__(self, dim: int, kernel_size: int = 4, causal: bool = False) -> None:
    super().__init__()
    self.kernel_size = int(kernel_size)
    self.causal = bool(causal)
    self.conv = nn.Conv1d(dim, dim, kernel_size=self.kernel_size, groups=dim, bias=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = x.transpose(1, 2)
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
    self.conv = nn.Conv2d(dim, dim, kernel_size=(self.kernel_h, self.kernel_w), groups=dim, bias=True)
    self.grid_size: Optional[Tuple[int, int]] = None
    self.expect_cls: Optional[bool] = None
    self._warned = False
    self._fallback = CanonLayer(dim, kernel_size=self.kernel_h, causal=self.causal_time)

  def _warn_once(self, msg: str) -> None:
    if not self._warned:
      print(msg, flush=True)
      self._warned = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def _student_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
  if hasattr(model, "forward_features"):
    feats = model.forward_features(x)
  else:
    feats = model(x)
  if isinstance(feats, (list, tuple)):
    feats = feats[-1]
  if feats.ndim == 3:
    feats = feats[:, 0, :]
  elif feats.ndim == 4:
    feats = feats.mean(dim=(-2, -1))
  return feats


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
  use_canon = bool(d.get("canon", False) or d.get("canon_abcd", False) or canon_a or canon_b or canon_c or canon_d or legacy_pre or legacy_post)
  return use_canon, canon_a, canon_b, canon_c, canon_d


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

  try:
    model = timm.create_model(
      "vit_small_patch16_224",
      img_size=(192, 128),
      in_chans=1,
      num_classes=0,
      global_pool="avg",
    )
  except Exception:
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


def _load_ckpt(path: Path, *, allow_unsafe: bool) -> dict:
  try:
    from torch.serialization import safe_globals  # type: ignore[attr-defined]
  except Exception:
    safe_globals = None

  if safe_globals is not None:
    import pathlib

    with safe_globals([pathlib.PosixPath]):
      try:
        return torch.load(path, map_location="cpu", weights_only=True)
      except TypeError:
        return torch.load(path, map_location="cpu")
      except Exception as exc:  # noqa: BLE001
        if not allow_unsafe:
          _die(
            "Safe checkpoint load failed. If you trust this checkpoint, retry with --unsafe-load.\n"
            f"Original error: {exc}"
          )
  if allow_unsafe:
    try:
      return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
      return torch.load(path, map_location="cpu")
  return torch.load(path, map_location="cpu")


def _infer_canon_layout_from_state(student_state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
  keys = list(student_state.keys())
  has_block_wrapper = any(".block." in k for k in keys)
  if not has_block_wrapper:
    return {}
  canon_a = any(".attn.module." in k for k in keys)
  canon_b_qkv = any((".attn.qkv.qkv." in k) or (".attn.module.qkv.qkv." in k) for k in keys)
  canon_b_post = any((".attn.proj.1.conv." in k) or (".attn.module.proj.1.conv." in k) for k in keys)
  canon_c = any(".mlp.module." in k for k in keys)
  canon_d = any((".mlp.fc1.fc1." in k) or (".mlp.module.fc1.fc1." in k) for k in keys)

  canon_2d: Optional[bool] = None
  for k, v in student_state.items():
    if ".conv.weight" not in k:
      continue
    if (".canon.conv.weight" in k) or (".proj.1.conv.weight" in k):
      if v.ndim == 4:
        canon_2d = True
        break
      if v.ndim == 3:
        canon_2d = False
        break
  return {
    "use_canon": True,
    "canon_a": canon_a,
    "canon_b": bool(canon_b_qkv or canon_b_post),
    "canon_b_qkv": canon_b_qkv,
    "canon_c": canon_c,
    "canon_d": canon_d,
    "canon_2d": canon_2d,
  }


def _load_student_and_proj(
  ckpt_path: Path,
  device: torch.device,
  allow_unsafe: bool,
  *,
  embedding_head: str,
) -> Tuple[nn.Module, nn.Module, dict, int]:
  ckpt = _load_ckpt(ckpt_path, allow_unsafe=allow_unsafe)
  if "student" not in ckpt:
    _die("Checkpoint missing required key: 'student'.")
  ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
  use_canon, canon_a, canon_b, canon_c, canon_d = _resolve_canon_flags_from_dict(ckpt_args)
  canon_kernel = int(ckpt_args.get("canon_kernel", 4))
  canon_causal = bool(ckpt_args.get("canon_causal", False))
  canon_2d = bool(ckpt_args.get("canon_2d", False))
  canon_no_pos_enc = bool(ckpt_args.get("canon_no_pos_enc", False))
  canon_b_qkv = bool(ckpt_args.get("canon_b_qkv", False))

  inferred = _infer_canon_layout_from_state(ckpt["student"])
  if inferred.get("use_canon", False):
    use_canon = True
    canon_a = bool(inferred.get("canon_a", canon_a))
    canon_b = bool(inferred.get("canon_b", canon_b))
    canon_c = bool(inferred.get("canon_c", canon_c))
    canon_d = bool(inferred.get("canon_d", canon_d))
    canon_b_qkv = bool(inferred.get("canon_b_qkv", canon_b_qkv))
    inferred_2d = inferred.get("canon_2d")
    if inferred_2d is not None:
      canon_2d = bool(inferred_2d)

  student = _build_student(
    use_canon=use_canon,
    canon_2d=canon_2d,
    canon_no_pos_enc=canon_no_pos_enc,
    canon_kernel=canon_kernel,
    canon_a=canon_a,
    canon_b=canon_b,
    canon_b_qkv=canon_b_qkv,
    canon_c=canon_c,
    canon_d=canon_d,
    canon_causal=canon_causal,
  ).to(device)
  student.load_state_dict(ckpt["student"], strict=True)
  student.eval()

  proj_state = ckpt.get("proj")
  student_dim = getattr(student, "num_features", None) or getattr(student, "embed_dim", None)

  if embedding_head == "proj":
    if proj_state is None or "weight" not in proj_state:
      _die("Projection head requested but checkpoint projection state is missing.")
    out_features, in_features = proj_state["weight"].shape
    proj = nn.Linear(in_features, out_features).to(device)
    proj.load_state_dict(proj_state, strict=True)
    proj.eval()
    emb_dim = int(out_features)
    return student, proj, ckpt_args, emb_dim

  if embedding_head == "student":
    proj = nn.Identity().to(device)
    proj.eval()
    if proj_state is not None and "weight" in proj_state:
      _, in_features = proj_state["weight"].shape
      emb_dim = int(in_features)
    elif student_dim is not None:
      emb_dim = int(student_dim)
    else:
      _die("Could not determine student embedding dimension for --embedding-head student.")
    return student, proj, ckpt_args, emb_dim

  _die(f"Unknown embedding head: {embedding_head}")


def _load_hf_hear_embedder(
  *,
  model_id: str,
  preprocess_audio: Callable[[torch.Tensor], torch.Tensor],
  device: torch.device,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
  try:
    from transformers import AutoModel
  except Exception as exc:  # noqa: BLE001
    _die(f"transformers is required for --embedding-model hear-hf: {exc}")

  os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
  try:
    model = AutoModel.from_pretrained(model_id)
  except Exception as exc:  # noqa: BLE001
    _die(
      f"Failed to load Hugging Face model '{model_id}'.\n"
      "If the repo is gated/private, run `huggingface-cli login` or set `HF_TOKEN`.\n"
      f"Original error: {exc}"
    )
  model.eval().to(device)
  for p in model.parameters():
    p.requires_grad = False

  def _embed_batch(batch: torch.Tensor) -> torch.Tensor:
    spec = preprocess_audio(batch)
    out = model(spec, return_dict=True)
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
      _die("Hugging Face HeAR output has no `pooler_output` field.")
    return pooled

  with torch.inference_mode():
    dummy = torch.zeros((1, 32000), dtype=torch.float32, device=device)
    emb = _embed_batch(dummy)
  return _embed_batch, int(emb.shape[-1])


def _ckpt_fingerprint(path: Path) -> str:
  st = path.stat()
  return f"{st.st_size}-{int(st.st_mtime)}"


def _hf_model_fingerprint(model_id: str) -> str:
  safe = model_id.replace("/", "_")
  digest = hashlib.sha1(model_id.encode("utf-8")).hexdigest()[:12]
  return f"hf-{safe}-{digest}"


def _cache_key(
  dataset_id: str,
  split: str,
  *,
  model_fp: str,
  target_sr: int,
  clip_seconds: float,
  crop: str,
  full_clip: bool,
  max_items: Optional[int],
) -> str:
  safe_id = dataset_id.replace("/", "_")
  max_tag = f"max{max_items}" if max_items is not None else "full"
  if full_clip:
    return f"{safe_id}-{split}-sr{target_sr}-fullclip-{max_tag}-{model_fp}"
  return f"{safe_id}-{split}-sr{target_sr}-cs{clip_seconds:.2f}-crop-{crop}-{max_tag}-{model_fp}"


def _load_cache(cache_dir: Path, key: str) -> Optional[Tuple[torch.Tensor, List[dict]]]:
  emb_path = cache_dir / key / "embeddings.npy"
  meta_path = cache_dir / key / "meta.jsonl"
  if not emb_path.exists() or not meta_path.exists():
    return None
  emb = np.load(emb_path)
  meta: List[dict] = []
  with meta_path.open("r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      meta.append(json.loads(line))
  return torch.from_numpy(emb), meta


def _save_cache(cache_dir: Path, key: str, X: torch.Tensor, meta: List[dict], info: dict) -> None:
  out_dir = cache_dir / key
  out_dir.mkdir(parents=True, exist_ok=True)
  emb_path = out_dir / "embeddings.npy"
  meta_path = out_dir / "meta.jsonl"
  info_path = out_dir / "info.json"
  np.save(emb_path, X.cpu().numpy())
  with meta_path.open("w", encoding="utf-8") as f:
    for row in meta:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
  info_path.write_text(json.dumps(info, indent=2))


def _embed_split_to_cache(
  ds: Iterable[dict],
  *,
  dataset_id: str,
  split: str,
  cache_dir: Path,
  cache_key: str,
  embed_batch_fn: Callable[[torch.Tensor], torch.Tensor],
  device: torch.device,
  target_sr: int,
  clip_samples: int,
  crop: str,
  full_clip: bool,
  batch_size: int,
  max_items: Optional[int],
  embedding_dim: int,
  log_every: int,
  log_decode_errors: bool,
  decode_log_max: int,
  meta_fn: Callable[[dict], Optional[dict]],
) -> Tuple[torch.Tensor, List[dict]]:
  pending_audio: List[torch.Tensor] = []
  pending_owner: List[int] = []
  pending_order: List[int] = []
  pending_meta: Dict[int, dict] = {}
  pending_needed: Dict[int, int] = {}
  pending_seen: Dict[int, int] = {}
  pending_sum: Dict[int, torch.Tensor] = {}
  next_owner_id = 0
  embeddings: List[torch.Tensor] = []
  meta_out: List[dict] = []
  kept = 0
  seg_total = 0
  skipped_meta = 0
  skipped_audio = 0
  decode_errors: Dict[str, int] = {}
  decode_logged = 0
  start_t = time.perf_counter()

  def _finalize_ready() -> None:
    while pending_order:
      owner = pending_order[0]
      seen = pending_seen.get(owner, 0)
      needed = pending_needed.get(owner, 0)
      if seen < needed:
        break
      pending_order.pop(0)
      emb_sum = pending_sum.pop(owner, None)
      meta = pending_meta.pop(owner, None)
      pending_needed.pop(owner, None)
      pending_seen.pop(owner, None)
      if emb_sum is None or meta is None or needed <= 0:
        continue
      embeddings.append((emb_sum / float(needed)).detach().cpu())
      meta_out.append(meta)

  def _flush(*, force: bool) -> None:
    nonlocal pending_audio, pending_owner
    if not pending_audio:
      return
    while pending_audio and (force or len(pending_audio) >= batch_size):
      n_take = len(pending_audio) if force else batch_size
      audio_items = pending_audio[:n_take]
      owners = pending_owner[:n_take]
      del pending_audio[:n_take]
      del pending_owner[:n_take]
      if full_clip:
        # In full-clip mode lengths can differ across items, so embed one clip per forward.
        for audio_t, owner in zip(audio_items, owners):
          batch = audio_t.unsqueeze(0)
          for emb in _embed_batches_with_fn(embed_batch_fn, [batch], device):
            row = emb[0]
            if owner not in pending_sum:
              pending_sum[owner] = row.clone()
            else:
              pending_sum[owner] = pending_sum[owner] + row
            pending_seen[owner] = pending_seen.get(owner, 0) + 1
      else:
        batch = torch.stack(audio_items, dim=0)
        for emb in _embed_batches_with_fn(embed_batch_fn, [batch], device):
          for row, owner in zip(emb, owners):
            if owner not in pending_sum:
              pending_sum[owner] = row.clone()
            else:
              pending_sum[owner] = pending_sum[owner] + row
            pending_seen[owner] = pending_seen.get(owner, 0) + 1
      _finalize_ready()

  if log_every:
    mode = "full-no-chunk" if full_clip else f"crop-{crop}"
    print(f"{dataset_id}/{split} CACHE start batch_size={batch_size} mode={mode} max_items={max_items}", flush=True)

  for ex in ds:
    if max_items is not None and kept >= max_items:
      break
    meta = meta_fn(ex)
    if meta is None:
      skipped_meta += 1
      continue
    audio = ex.get("audio")
    arr, sr, err = _decode_audio_dict(audio, ex=ex, default_sr=target_sr)
    if arr is None or sr is None:
      skipped_audio += 1
      if err:
        decode_errors[err] = decode_errors.get(err, 0) + 1
        if log_decode_errors and decode_logged < max(0, decode_log_max):
          decode_logged += 1
          keys = list(audio.keys()) if isinstance(audio, dict) else None
          has_bytes = isinstance(audio, dict) and audio.get("bytes") is not None
          has_path = isinstance(audio, dict) and audio.get("path") is not None
          a_type = type(audio).__name__
          a_mod = type(audio).__module__
          a_repr = repr(audio)
          if len(a_repr) > 160:
            a_repr = a_repr[:160] + "..."
          attr_flags = {}
          for attr in ("array", "sampling_rate", "sample_rate", "path", "bytes", "sr"):
            try:
              attr_flags[attr] = hasattr(audio, attr)
            except Exception:
              attr_flags[attr] = False
          print(
            f"{dataset_id}/{split} DECODE_FAIL reason={err} type={a_mod}.{a_type} "
            f"keys={keys} bytes={has_bytes} path={has_path} attrs={attr_flags} repr={a_repr}",
            flush=True,
          )
      continue
    clips = _prep_audio_clips(
      arr,
      sr,
      target_sr=target_sr,
      clip_samples=clip_samples,
      crop=crop,
      full_clip=full_clip,
    )
    n_clips = int(clips.shape[0])
    owner = next_owner_id
    next_owner_id += 1
    pending_order.append(owner)
    pending_meta[owner] = meta
    pending_needed[owner] = n_clips
    pending_seen[owner] = 0
    pending_audio.extend([clips[i] for i in range(n_clips)])
    pending_owner.extend([owner] * n_clips)
    kept += 1
    seg_total += n_clips
    if len(pending_audio) >= batch_size or (max_items is not None and kept >= max_items):
      _flush(force=False)
    if log_every and kept % log_every == 0:
      elapsed = time.perf_counter() - start_t
      rate = kept / elapsed if elapsed > 0 else 0.0
      eta = ""
      if max_items is not None and rate > 0:
        remaining = max_items - kept
        eta = f" eta={remaining / rate / 60:.1f}m"
      print(
        f"{dataset_id}/{split} CACHE kept={kept} skip_meta={skipped_meta} skip_audio={skipped_audio} "
        f"segments={seg_total} rate={rate:.2f}/s elapsed={elapsed/60:.1f}m{eta}",
        flush=True,
      )

  _flush(force=True)
  _finalize_ready()
  if decode_errors:
    top = sorted(decode_errors.items(), key=lambda kv: kv[1], reverse=True)
    summary = ", ".join([f"{k}={v}" for k, v in top[:6]])
    print(f"{dataset_id}/{split} DECODE_ERRORS {summary}", flush=True)
  if embeddings:
    X = torch.stack(embeddings, dim=0)
  else:
    X = torch.empty((0, int(embedding_dim)))
  info = {
    "dataset_id": dataset_id,
    "split": split,
    "count": int(X.shape[0]),
    "skipped_meta": skipped_meta,
    "skipped_audio": skipped_audio,
    "decode_errors": decode_errors,
    "target_sr": target_sr,
    "clip_samples": (None if full_clip else clip_samples),
    "crop": crop,
    "full_clip": full_clip,
    "segments_total": seg_total,
    "max_items": max_items,
  }
  _save_cache(cache_dir, cache_key, X, meta_out, info)
  return X, meta_out


def _get_cached_split(
  *,
  dataset_id: str,
  split: str,
  ds: Iterable[dict],
  cache_dir: Path,
  model_fp: str,
  target_sr: int,
  clip_seconds: float,
  crop: str,
  full_clip: bool,
  batch_size: int,
  max_items: Optional[int],
  embedding_dim: int,
  log_every: int,
  log_decode_errors: bool,
  decode_log_max: int,
  embed_batch_fn: Callable[[torch.Tensor], torch.Tensor],
  device: torch.device,
  meta_fn: Callable[[dict], Optional[dict]],
  refresh: bool,
) -> Tuple[torch.Tensor, List[dict]]:
  key = _cache_key(
    dataset_id,
    split,
    model_fp=model_fp,
    target_sr=target_sr,
    clip_seconds=clip_seconds,
    crop=crop,
    full_clip=full_clip,
    max_items=max_items,
  )
  if not refresh:
    cached = _load_cache(cache_dir, key)
    if cached is not None:
      print(f"{dataset_id}/{split} CACHE hit ({cached[0].shape[0]} embeddings)", flush=True)
      return cached
  return _embed_split_to_cache(
    ds,
    dataset_id=dataset_id,
    split=split,
    cache_dir=cache_dir,
    cache_key=key,
    embed_batch_fn=embed_batch_fn,
    device=device,
    target_sr=target_sr,
    clip_samples=int(round(clip_seconds * target_sr)),
    crop=crop,
    full_clip=full_clip,
    batch_size=batch_size,
    max_items=max_items,
    embedding_dim=embedding_dim,
    log_every=log_every,
    log_decode_errors=log_decode_errors,
    decode_log_max=decode_log_max,
    meta_fn=meta_fn,
  )


def _apply_label_fn(
  X: torch.Tensor,
  meta: Sequence[dict],
  label_fn: Callable[[dict], Optional[Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
  indices: List[int] = []
  labels: List[Any] = []
  for i, m in enumerate(meta):
    y = label_fn(m)
    if y is None:
      continue
    indices.append(i)
    labels.append(y)
  if not indices:
    return torch.empty((0, X.shape[1])), torch.empty((0,))
  X_sel = X[indices]
  y_tensor = torch.tensor(labels)
  return X_sel, y_tensor


def _apply_label_fn_with_groups(
  X: torch.Tensor,
  meta: Sequence[dict],
  label_fn: Callable[[dict], Optional[Any]],
  group_keys: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor, List[Any]]:
  indices: List[int] = []
  labels: List[Any] = []
  groups: List[Any] = []
  for i, m in enumerate(meta):
    y = label_fn(m)
    if y is None:
      continue
    indices.append(i)
    labels.append(y)
    gid_parts = []
    for key in group_keys:
      val = m.get(key)
      if val is None:
        continue
      gid_parts.append(str(val))
    if gid_parts:
      groups.append("|".join(gid_parts))
    else:
      groups.append(f"__idx_{i}")
  if not indices:
    return torch.empty((0, X.shape[1])), torch.empty((0,)), []
  X_sel = X[indices]
  y_tensor = torch.tensor(labels)
  return X_sel, y_tensor, groups


def _group_by_participant(
  X: torch.Tensor,
  meta: Sequence[dict],
  label_fn: Callable[[dict], Optional[Any]],
) -> Tuple[torch.Tensor, torch.Tensor]:
  groups: Dict[str, Tuple[torch.Tensor, int, Any]] = {}
  conflicts: set[str] = set()
  for i, m in enumerate(meta):
    pid = m.get("participant_id")
    if not pid:
      continue
    y = label_fn(m)
    if y is None:
      continue
    if pid in conflicts:
      continue
    if pid in groups:
      cur_e, cur_n, cur_y = groups[pid]
      if cur_y != y:
        conflicts.add(pid)
        groups.pop(pid, None)
        continue
      groups[pid] = (cur_e + X[i], cur_n + 1, cur_y)
    else:
      groups[pid] = (X[i].clone(), 1, y)
  if not groups:
    return torch.empty((0, X.shape[1])), torch.empty((0,))
  embs: List[torch.Tensor] = []
  labels: List[Any] = []
  for e, n, y in groups.values():
    embs.append(e / float(n))
    labels.append(y)
  return torch.stack(embs, dim=0), torch.tensor(labels)

def _resample(audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
  if src_sr == dst_sr:
    return audio
  try:
    from scipy import signal
  except Exception as exc:  # noqa: BLE001
    _die(f"Resampling requires scipy: {exc}")
  new_len = int(round(audio.shape[0] * (dst_sr / src_sr)))
  res = signal.resample(audio.numpy(), new_len)
  return torch.from_numpy(res).float()


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
  if audio.ndim == 2:
    # soundfile returns [T, C]; torchaudio often returns [C, T]
    if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
      return audio.mean(dim=0)
    return audio.mean(dim=1)
  return audio


def _crop_peak(audio: torch.Tensor, clip_samples: int, sr: int) -> torch.Tensor:
  if audio.numel() <= clip_samples:
    if audio.numel() < clip_samples:
      return F.pad(audio, (0, clip_samples - audio.numel()))
    return audio
  win = max(1, int(sr * 0.05))
  hop = max(1, int(sr * 0.01))
  if audio.numel() < win:
    start = 0
  else:
    # Use pooling instead of unfold to avoid materializing a huge [frames, win]
    # view for long clips.
    energy = F.avg_pool1d((audio * audio).view(1, 1, -1), kernel_size=win, stride=hop).view(-1)
    idx = int(torch.argmax(energy)) if energy.numel() else 0
    center = idx * hop + win // 2
    start = max(0, min(int(center - clip_samples // 2), audio.numel() - clip_samples))
  return audio[start : start + clip_samples]


def _crop_center(audio: torch.Tensor, clip_samples: int) -> torch.Tensor:
  if audio.numel() <= clip_samples:
    if audio.numel() < clip_samples:
      return F.pad(audio, (0, clip_samples - audio.numel()))
    return audio
  start = (audio.numel() - clip_samples) // 2
  return audio[start : start + clip_samples]


def _prep_audio_resampled(
  audio_arr: Any,
  sr: int,
  *,
  target_sr: int,
) -> torch.Tensor:
  audio = torch.as_tensor(audio_arr).float()
  audio = _to_mono(audio)
  return _resample(audio, sr, target_sr)


def _prep_audio_clips(
  audio_arr: Any,
  sr: int,
  *,
  target_sr: int,
  clip_samples: int,
  crop: str,
  full_clip: bool,
) -> torch.Tensor:
  # For fixed-window evaluation (full_clip=False), crop in the source sample
  # rate first and then resample only the selected window. This avoids expensive
  # resampling of long clips when we only need a short segment.
  audio = torch.as_tensor(audio_arr).float()
  audio = _to_mono(audio)
  if full_clip:
    audio = _resample(audio, sr, target_sr)
    if audio.numel() <= 0:
      audio = F.pad(audio, (0, 1))
    return audio.unsqueeze(0)

  # Equivalent window length at the source sample rate.
  clip_src = int(round(float(clip_samples) * (float(sr) / float(target_sr))))
  clip_src = max(1, clip_src)
  if crop == "peak":
    seg = _crop_peak(audio, clip_src, sr)
  else:
    seg = _crop_center(audio, clip_src)
  seg = _resample(seg, sr, target_sr)
  # Enforce exact window length after resampling (rounding can drift by a few samples).
  if seg.numel() < clip_samples:
    seg = F.pad(seg, (0, clip_samples - seg.numel()))
  elif seg.numel() > clip_samples:
    seg = seg[:clip_samples]
  return seg.unsqueeze(0)


def _prep_audio(
  audio_arr: Any,
  sr: int,
  *,
  target_sr: int,
  clip_samples: int,
  crop: str,
) -> torch.Tensor:
  return _prep_audio_clips(
    audio_arr,
    sr,
    target_sr=target_sr,
    clip_samples=clip_samples,
    crop=crop,
    full_clip=False,
  ).squeeze(0)


def _decode_audio_dict(
  audio: Any,
  ex: Optional[dict] = None,
  *,
  default_sr: Optional[int] = None,
) -> Tuple[Optional[Any], Optional[int], Optional[str]]:
  if audio is None:
    return None, None, "audio_none"

  decode_err: Optional[str] = None

  # pyarrow.StructScalar or similar -> dict
  if not isinstance(audio, dict) and hasattr(audio, "as_py"):
    try:
      audio = audio.as_py()
    except Exception as exc:  # noqa: BLE001
      decode_err = f"audio_as_py:{type(exc).__name__}"

  # some datasets return lazy decoders; try explicit decode
  if not isinstance(audio, dict) and hasattr(audio, "decode"):
    try:
      decoded = audio.decode()
      if isinstance(decoded, dict):
        audio = decoded
      elif isinstance(decoded, (tuple, list)) and len(decoded) == 2:
        arr, sr = decoded
        if sr is None:
          sr = default_sr
        if sr is not None:
          return arr, int(sr), None
      elif isinstance(decoded, (np.ndarray, torch.Tensor)):
        sr = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
        if ex is not None and sr is None:
          sr = ex.get("sampling_rate") or ex.get("sample_rate") or ex.get("sr") or ex.get("audio_sampling_rate")
        if sr is None:
          sr = default_sr
        if sr is not None:
          return decoded, int(sr), None
    except Exception as exc:  # noqa: BLE001
      decode_err = f"audio_decode_method:{type(exc).__name__}"

  if not isinstance(audio, dict):
    arr = getattr(audio, "array", None)
    sr = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
    if arr is not None and sr is not None:
      return arr, int(sr), None
    path_attr = getattr(audio, "path", None)
    bytes_attr = getattr(audio, "bytes", None)
    if any(v is not None for v in (arr, sr, path_attr, bytes_attr)):
      audio = {
        "array": arr,
        "sampling_rate": sr,
        "bytes": bytes_attr,
        "path": path_attr,
      }

  # mapping-like
  if not isinstance(audio, dict) and hasattr(audio, "keys") and hasattr(audio, "__getitem__"):
    try:
      audio = {k: audio[k] for k in audio.keys()}
    except Exception as exc:  # noqa: BLE001
      decode_err = f"audio_mapping:{type(exc).__name__}"

  # object with attributes
  if not isinstance(audio, dict) and hasattr(audio, "__dict__"):
    try:
      d = {
        "array": getattr(audio, "array", None),
        "sampling_rate": getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None),
        "bytes": getattr(audio, "bytes", None),
        "path": getattr(audio, "path", None),
      }
      if any(v is not None for v in d.values()):
        audio = d
    except Exception as exc:  # noqa: BLE001
      decode_err = f"audio_attrs:{type(exc).__name__}"

  if isinstance(audio, dict):
    arr = audio.get("array")
    sr = audio.get("sampling_rate")
    if arr is not None and sr is not None:
      return arr, int(sr), None
    if arr is not None and sr is None and default_sr is not None:
      return arr, int(default_sr), None
    data_bytes = audio.get("bytes")
    if data_bytes is not None:
      err = None
      try:
        import soundfile as sf
      except Exception:
        sf = None
      if sf is not None:
        try:
          with io.BytesIO(data_bytes) as bio:
            data, sr = sf.read(bio, dtype="float32", always_2d=False)
          return data, int(sr), None
        except Exception:
          err = "soundfile_bytes"
      try:
        import torchaudio
        with io.BytesIO(data_bytes) as bio:
          data, sr = torchaudio.load(bio)
        if data.ndim == 2:
          data = data.mean(dim=0)
        return data.numpy(), int(sr), None
      except Exception:
        return None, None, (err + "+torchaudio_bytes" if err else "torchaudio_bytes")
    path = audio.get("path")
    if path:
      err = None
      try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        return data, int(sr), None
      except Exception:
        err = "soundfile_path"
      try:
        import torchaudio
        data, sr = torchaudio.load(path)
        if data.ndim == 2:
          data = data.mean(dim=0)
        return data.numpy(), int(sr), None
      except Exception:
        return None, None, (err + "+torchaudio_path" if err else "torchaudio_path")
    if "array" in audio or "sampling_rate" in audio:
      return None, None, "audio_array_missing"
    if data_bytes is None and path is None:
      return None, None, "audio_missing_bytes_path"
    return None, None, "audio_decode_failed"

  # tuple/list (array, sr)
  if isinstance(audio, (tuple, list)) and len(audio) == 2:
    arr, sr = audio
    if sr is not None:
      return arr, int(sr), None

  # raw bytes
  if isinstance(audio, memoryview):
    audio = audio.tobytes()
  if isinstance(audio, (bytes, bytearray)):
    try:
      import soundfile as sf
      with io.BytesIO(audio) as bio:
        data, sr = sf.read(bio, dtype="float32", always_2d=False)
      return data, int(sr), None
    except Exception:
      return None, None, "soundfile_bytes"

  # numpy/torch array without sampling rate
  if isinstance(audio, (np.ndarray, torch.Tensor)):
    sr = None
    if ex is not None:
      sr = ex.get("sampling_rate") or ex.get("sample_rate") or ex.get("sr") or ex.get("audio_sampling_rate")
    if sr is None:
      sr = default_sr
    if sr is not None:
      return audio, int(sr), None
  if ex is not None:
    sr = ex.get("sampling_rate") or ex.get("sample_rate") or ex.get("sr") or ex.get("audio_sampling_rate")
    if sr is not None:
      return audio, int(sr), None

  # path-like
  if isinstance(audio, (str, Path)):
    path = str(audio)
    err = None
    try:
      import soundfile as sf
      data, sr = sf.read(path, dtype="float32", always_2d=False)
      return data, int(sr), None
    except Exception:
      err = "soundfile_path"
    try:
      import torchaudio
      data, sr = torchaudio.load(path)
      if data.ndim == 2:
        data = data.mean(dim=0)
      return data.numpy(), int(sr), None
    except Exception:
      return None, None, (err + "+torchaudio_path" if err else "torchaudio_path")

  # final fallback: ask HF Audio to decode if available
  try:
    from datasets import Audio as HFAudio
  except Exception:
    HFAudio = None  # type: ignore
  if HFAudio is not None:
    try:
      decoded = HFAudio(sampling_rate=default_sr).decode_example(audio)
      if isinstance(decoded, dict):
        arr = decoded.get("array")
        sr = decoded.get("sampling_rate") or default_sr
        if arr is not None and sr is not None:
          return arr, int(sr), None
    except Exception:
      pass

  return None, None, (decode_err or "audio_not_dict")


def _embed_batches(
  student: nn.Module,
  proj: nn.Module,
  preprocess_audio: Callable[[torch.Tensor], torch.Tensor],
  batches: Iterable[torch.Tensor],
  device: torch.device,
) -> Iterable[torch.Tensor]:
  for batch in batches:
    batch = batch.to(device)
    with torch.inference_mode():
      spec = preprocess_audio(batch)
      feats = _student_features(student, spec)
      emb = proj(feats)
    yield emb.detach().cpu()


def _embed_batches_with_fn(
  embed_batch_fn: Callable[[torch.Tensor], torch.Tensor],
  batches: Iterable[torch.Tensor],
  device: torch.device,
) -> Iterable[torch.Tensor]:
  for batch in batches:
    batch = batch.to(device)
    with torch.inference_mode():
      emb = embed_batch_fn(batch)
    yield emb.detach().cpu()


def _roc_auc_score(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
  y_true = y_true.flatten().detach().cpu()
  y_score = y_score.flatten().detach().cpu()
  pos = int((y_true == 1).sum())
  neg = int((y_true == 0).sum())
  if pos == 0 or neg == 0:
    return float("nan")
  order = torch.argsort(y_score, descending=True)
  y = y_true[order]
  tps = torch.cumsum(y == 1, dim=0).float()
  fps = torch.cumsum(y == 0, dim=0).float()
  tpr = tps / pos
  fpr = fps / neg
  tpr = torch.cat([torch.tensor([0.0]), tpr, torch.tensor([1.0])])
  fpr = torch.cat([torch.tensor([0.0]), fpr, torch.tensor([1.0])])
  return float(torch.trapz(tpr, fpr))


def _average_precision(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
  y_true = y_true.flatten().detach().cpu()
  y_score = y_score.flatten().detach().cpu()
  pos = int((y_true == 1).sum())
  if pos == 0:
    return float("nan")
  order = torch.argsort(y_score, descending=True)
  y = y_true[order]
  tp = torch.cumsum(y == 1, dim=0).float()
  fp = torch.cumsum(y == 0, dim=0).float()
  precision = tp / torch.clamp(tp + fp, min=1.0)
  return float((precision * (y == 1).float()).sum() / pos)


def _bootstrap_ap_cis_shared(
  tasks: Sequence[Tuple[str, np.ndarray, np.ndarray]],
  *,
  n_bootstrap: int,
  seed: int,
  alpha: float,
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float], int]:
  """
  Bootstrap AP confidence intervals for multiple tasks that share the same test examples.

  Each task is a tuple (name, y_true, y_score) with y_true in {0,1}.

  Returns:
    - per_task_ci[name] = (lo, hi)
    - map_ci = (lo, hi) for mean(AP) across tasks
    - n_valid = number of bootstrap draws used (draws with >=1 positive for all tasks)
  """
  if n_bootstrap <= 0:
    return {}, (float("nan"), float("nan")), 0
  if not (0.0 < alpha < 1.0):
    raise ValueError("--bootstrap-alpha must be in (0,1).")
  if not tasks:
    return {}, (float("nan"), float("nan")), 0

  n = int(tasks[0][1].shape[0])
  if n <= 0:
    return {}, (float("nan"), float("nan")), 0
  for name, y, s in tasks:
    if int(y.shape[0]) != n or int(s.shape[0]) != n:
      raise ValueError(f"Bootstrap requires shared test set size; task '{name}' differs.")

  rng = np.random.default_rng(int(seed))
  p = np.full((n,), 1.0 / float(n), dtype=np.float64)
  denom = np.arange(1, n + 1, dtype=np.float64)

  task_names: List[str] = []
  orders: List[np.ndarray] = []
  y_sorted_list: List[np.ndarray] = []
  for name, y, s in tasks:
    task_names.append(str(name))
    # Stable sort for deterministic tie-breaking.
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    orders.append(order)
    y_sorted = np.asarray(y, dtype=np.int8)[order]
    y_sorted_list.append(y_sorted)

  boot_matrix: List[np.ndarray] = []
  for _ in range(int(n_bootstrap)):
    weights = rng.multinomial(n, p).astype(np.int32, copy=False)
    aps = np.empty((len(tasks),), dtype=np.float64)
    ok = True
    for j in range(len(tasks)):
      w_sorted = weights[orders[j]]
      # Expanded bootstrap sample in score-sorted order (length == n).
      y_rep = np.repeat(y_sorted_list[j], w_sorted)
      pos = int(y_rep.sum())
      if pos <= 0:
        ok = False
        break
      tp = np.cumsum(y_rep, dtype=np.int64)
      precision = tp / denom
      aps[j] = float(precision[y_rep == 1].sum() / float(pos))
    if not ok:
      continue
    boot_matrix.append(aps)

  if not boot_matrix:
    per_task = {name: (float("nan"), float("nan")) for name in task_names}
    return per_task, (float("nan"), float("nan")), 0

  boot = np.stack(boot_matrix, axis=0)
  lo_q = 100.0 * (alpha / 2.0)
  hi_q = 100.0 * (1.0 - alpha / 2.0)
  per_task_ci: Dict[str, Tuple[float, float]] = {}
  for j, name in enumerate(task_names):
    per_task_ci[name] = (
      float(np.percentile(boot[:, j], lo_q)),
      float(np.percentile(boot[:, j], hi_q)),
    )
  boot_map = boot.mean(axis=1)
  map_ci = (
    float(np.percentile(boot_map, lo_q)),
    float(np.percentile(boot_map, hi_q)),
  )
  return per_task_ci, map_ci, int(boot.shape[0])


def _format_ap_ci(ap: float, lo: float, hi: float) -> str:
  if math.isnan(ap) or math.isnan(lo) or math.isnan(hi):
    return f"{ap:.4f}"
  return f"{ap:.4f} [{lo:.4f}, {hi:.4f}]"


def _accuracy(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
  preds = (y_score >= 0).long()
  return float((preds.flatten() == y_true.flatten().long()).float().mean())


def _r2_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  y_true = y_true.float()
  y_pred = y_pred.float()
  ss_res = torch.sum((y_true - y_pred) ** 2)
  ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
  if ss_tot == 0:
    return float("nan")
  return float(1.0 - (ss_res / ss_tot))


def _mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
  return float(torch.mean(torch.abs(y_true.float() - y_pred.float())))


def _format_reg(res: Dict[str, float]) -> str:
  c_val = res.get("C", float("nan"))
  if not math.isnan(c_val):
    return f"C={c_val:.4g}"
  a_val = res.get("alpha", float("nan"))
  if not math.isnan(a_val):
    return f"alpha={a_val:.4g}"
  wd_val = res.get("weight_decay", float("nan"))
  if not math.isnan(wd_val):
    return f"wd={wd_val:.4g}"
  return "reg=nan"


def _parse_int_list(value: str) -> List[int]:
  if value is None:
    return []
  raw = [v.strip() for v in value.split(",")]
  items = [v for v in raw if v]
  if not items:
    return []
  if len(items) == 1 and items[0].lower() in {"none", "null"}:
    return []
  out: List[int] = []
  for item in items:
    try:
      out.append(int(item))
    except ValueError as exc:
      _die(f"Invalid integer in list: {item} ({exc})")
  return out


def _parse_float_list(value: str) -> List[float]:
  if value is None:
    return []
  raw = [v.strip() for v in value.split(",")]
  items = [v for v in raw if v]
  if not items:
    return []
  if len(items) == 1 and items[0].lower() in {"none", "null"}:
    return []
  out: List[float] = []
  for item in items:
    try:
      out.append(float(item))
    except ValueError as exc:
      _die(f"Invalid float in list: {item} ({exc})")
  return out


def _parse_str_list(value: Optional[str]) -> List[str]:
  if value is None:
    return []
  raw = [v.strip() for v in value.split(",")]
  items = [v for v in raw if v]
  if not items:
    return []
  if len(items) == 1 and items[0].lower() in {"none", "null"}:
    return []
  return items


def _normalize_task_name(name: str) -> str:
  s = str(name).strip().lower()
  s = s.replace("_", "-").replace(" ", "-")
  while "--" in s:
    s = s.replace("--", "-")
  return s


def _make_probe(
  in_dim: int,
  out_dim: int,
  *,
  probe_type: str,
  mlp_hidden: Sequence[int],
  mlp_dropout: float,
) -> nn.Module:
  if probe_type == "linear":
    return nn.Linear(in_dim, out_dim)
  if not mlp_hidden:
    _die("MLP probe requires --probe-mlp-hidden with at least one size.")
  layers: List[nn.Module] = []
  prev = in_dim
  for h in mlp_hidden:
    layers.append(nn.Linear(prev, h))
    layers.append(nn.ReLU())
    if mlp_dropout > 0:
      layers.append(nn.Dropout(p=mlp_dropout))
    prev = h
  layers.append(nn.Linear(prev, out_dim))
  return nn.Sequential(*layers)


def _as_numpy(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy()


def _sklearn_decision(model: Any, X: torch.Tensor) -> np.ndarray:
  X_np = _as_numpy(X)
  if hasattr(model, "decision_function"):
    scores = model.decision_function(X_np)
  elif hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_np)
    scores = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
  else:
    scores = model.predict(X_np)
  return np.asarray(scores).reshape(-1)


def _make_cv_splitter(
  y: np.ndarray,
  *,
  cv: int,
  seed: int,
  groups: Optional[Sequence[Any]],
  is_classification: bool,
):
  if cv < 2:
    return None
  n_samples = int(y.shape[0])
  if n_samples < 2:
    return None
  if groups is not None:
    from sklearn.model_selection import GroupKFold

    unique_groups = len(set(groups))
    if unique_groups < 2:
      return None
    n_splits = min(cv, unique_groups)
    return GroupKFold(n_splits=n_splits)
  if is_classification:
    from sklearn.model_selection import StratifiedKFold

    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts.min()) if counts.size else 0
    n_splits = min(cv, min_class, n_samples)
    if n_splits < 2:
      return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
  from sklearn.model_selection import KFold

  n_splits = min(cv, n_samples)
  if n_splits < 2:
    return None
  return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _train_binary_probe_sklearn(
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  X_val: torch.Tensor,
  y_val: torch.Tensor,
  *,
  c_grid: Sequence[float],
  cv: int,
  seed: int,
  groups: Optional[Sequence[Any]],
  solver: str,
  max_iter: int,
  tol: float,
) -> Tuple[Any, Dict[str, float]]:
  try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
  except Exception as exc:  # noqa: BLE001
    _die(f"scikit-learn is required for sklearn probes: {exc}")

  X_np = _as_numpy(X_train)
  y_np = _as_numpy(y_train).astype(int)
  groups_arr = None if groups is None else np.asarray(groups)
  c_vals = list(c_grid) if c_grid else [1.0]
  splitter = _make_cv_splitter(y_np, cv=cv, seed=seed, groups=groups_arr, is_classification=True)

  if splitter is None or len(c_vals) == 1:
    best_c = float(c_vals[0])
    model = LogisticRegression(solver=solver, max_iter=max_iter, tol=tol, C=best_c)
    model.fit(X_np, y_np)
  else:
    grid = GridSearchCV(
      LogisticRegression(solver=solver, max_iter=max_iter, tol=tol),
      {"C": c_vals},
      scoring="roc_auc",
      cv=splitter,
      refit=True,
    )
    grid.fit(X_np, y_np, groups=groups_arr)
    model = grid.best_estimator_
    best_c = float(grid.best_params_.get("C", 1.0))

  val_scores = _sklearn_decision(model, X_val)
  val_auc = _roc_auc_score(y_val, torch.from_numpy(val_scores))
  return model, {"val_auc": val_auc, "C": best_c}


def _train_regression_probe_sklearn(
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  X_val: torch.Tensor,
  y_val: torch.Tensor,
  *,
  alpha_grid: Sequence[float],
  cv: int,
  seed: int,
  groups: Optional[Sequence[Any]],
) -> Tuple[Any, Dict[str, float]]:
  try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
  except Exception as exc:  # noqa: BLE001
    _die(f"scikit-learn is required for sklearn probes: {exc}")

  X_np = _as_numpy(X_train)
  y_np = _as_numpy(y_train).astype(float)
  groups_arr = None if groups is None else np.asarray(groups)
  a_vals = list(alpha_grid) if alpha_grid else [1.0]
  splitter = _make_cv_splitter(y_np, cv=cv, seed=seed, groups=groups_arr, is_classification=False)

  if splitter is None or len(a_vals) == 1:
    best_alpha = float(a_vals[0])
    model = Ridge(alpha=best_alpha)
    model.fit(X_np, y_np)
  else:
    grid = GridSearchCV(
      Ridge(),
      {"alpha": a_vals},
      scoring="r2",
      cv=splitter,
      refit=True,
    )
    grid.fit(X_np, y_np, groups=groups_arr)
    model = grid.best_estimator_
    best_alpha = float(grid.best_params_.get("alpha", 1.0))

  val_pred = model.predict(_as_numpy(X_val))
  val_r2 = _r2_score(y_val, torch.from_numpy(np.asarray(val_pred)))
  return model, {"val_r2": val_r2, "alpha": best_alpha}


def _train_binary_probe(
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  X_val: torch.Tensor,
  y_val: torch.Tensor,
  *,
  weight_decays: Sequence[float],
  lr: float,
  epochs: int,
  batch_size: int,
  device: torch.device,
  probe_type: str,
  mlp_hidden: Sequence[int],
  mlp_dropout: float,
) -> Tuple[nn.Module, Dict[str, float]]:
  best_auc = -math.inf
  best_state = None
  best_metrics: Dict[str, float] = {}
  for wd in weight_decays:
    model = _make_probe(
      X_train.shape[1],
      1,
      probe_type=probe_type,
      mlp_hidden=mlp_hidden,
      mlp_dropout=mlp_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    if pos > 0 and neg > 0:
      pos_weight = torch.tensor([neg / pos], device=device)
      loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
      loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
      idx = torch.randperm(X_train.shape[0])
      for start in range(0, X_train.shape[0], batch_size):
        sel = idx[start : start + batch_size]
        xb = X_train[sel].to(device)
        yb = y_train[sel].float().to(device).unsqueeze(1)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
      val_logits = model(X_val.to(device)).squeeze(1).cpu()
    auc = _roc_auc_score(y_val, val_logits)
    if auc > best_auc:
      best_auc = auc
      best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
      best_metrics = {"val_auc": auc, "weight_decay": wd}
  if best_state is None:
    _die("Failed to train binary probe (no valid models).")
  best_model = _make_probe(
    X_train.shape[1],
    1,
    probe_type=probe_type,
    mlp_hidden=mlp_hidden,
    mlp_dropout=mlp_dropout,
  )
  best_model.load_state_dict(best_state)
  return best_model, best_metrics


def _train_regression_probe(
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  X_val: torch.Tensor,
  y_val: torch.Tensor,
  *,
  weight_decays: Sequence[float],
  lr: float,
  epochs: int,
  batch_size: int,
  device: torch.device,
  probe_type: str,
  mlp_hidden: Sequence[int],
  mlp_dropout: float,
) -> Tuple[nn.Module, Dict[str, float]]:
  best_r2 = -math.inf
  best_state = None
  best_metrics: Dict[str, float] = {}
  for wd in weight_decays:
    model = _make_probe(
      X_train.shape[1],
      1,
      probe_type=probe_type,
      mlp_hidden=mlp_hidden,
      mlp_dropout=mlp_dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
      idx = torch.randperm(X_train.shape[0])
      for start in range(0, X_train.shape[0], batch_size):
        sel = idx[start : start + batch_size]
        xb = X_train[sel].to(device)
        yb = y_train[sel].float().to(device).unsqueeze(1)
        opt.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
      val_pred = model(X_val.to(device)).squeeze(1).cpu()
    r2 = _r2_score(y_val, val_pred)
    if r2 > best_r2:
      best_r2 = r2
      best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
      best_metrics = {"val_r2": r2, "weight_decay": wd}
  if best_state is None:
    _die("Failed to train regression probe (no valid models).")
  best_model = _make_probe(
    X_train.shape[1],
    1,
    probe_type=probe_type,
    mlp_hidden=mlp_hidden,
    mlp_dropout=mlp_dropout,
  )
  best_model.load_state_dict(best_state)
  return best_model, best_metrics


def _collect_embeddings(
  ds: Iterable[dict],
  *,
  audio_key: str,
  label_fn: Callable[[dict], Optional[Any]],
  preprocess_audio: Callable[[torch.Tensor], torch.Tensor],
  student: nn.Module,
  proj: nn.Module,
  device: torch.device,
  target_sr: int,
  clip_samples: int,
  crop: str,
  batch_size: int,
  max_items: Optional[int],
  log_every: int = 0,
  log_prefix: str = "",
  key_fn: Optional[Callable[[dict], Optional[str]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  embeddings: List[torch.Tensor] = []
  labels: List[Any] = []
  pending_audio: List[torch.Tensor] = []
  pending_labels: List[Any] = []
  pending_keys: List[Optional[str]] = []
  grouped: Dict[str, Tuple[torch.Tensor, int, Any]] = {}

  def _flush() -> None:
    nonlocal pending_audio, pending_labels, pending_keys
    if not pending_audio:
      return
    batch = torch.stack(pending_audio, dim=0)
    for emb in _embed_batches(student, proj, preprocess_audio, [batch], device):
      if key_fn is None:
        embeddings.extend(list(emb))
        labels.extend(pending_labels)
      else:
        for e, k, y in zip(emb, pending_keys, pending_labels):
          if k is None:
            continue
          if k not in grouped:
            grouped[k] = (e.clone(), 1, y)
          else:
            cur_e, cur_n, cur_y = grouped[k]
            grouped[k] = (cur_e + e, cur_n + 1, cur_y)
    pending_audio = []
    pending_labels = []
    pending_keys = []

  count = 0
  kept = 0
  skipped_label = 0
  skipped_audio = 0
  start_t = time.perf_counter()
  last_log = start_t
  if log_every:
    prefix = f"{log_prefix} " if log_prefix else ""
    max_tag = f" max_items={max_items}" if max_items is not None else ""
    print(f"{prefix}COLLECT start batch_size={batch_size}{max_tag}", flush=True)
  for ex in ds:
    if max_items is not None and count >= max_items:
      break
    y = label_fn(ex)
    if y is None:
      skipped_label += 1
      continue
    audio = ex.get(audio_key)
    arr, sr, err = _decode_audio_dict(audio, ex=ex, default_sr=target_sr)
    if arr is None or sr is None:
      skipped_audio += 1
      continue
    audio_t = _prep_audio(arr, sr, target_sr=target_sr, clip_samples=clip_samples, crop=crop)
    pending_audio.append(audio_t)
    pending_labels.append(y)
    pending_keys.append(key_fn(ex) if key_fn is not None else None)
    count += 1
    kept += 1
    if len(pending_audio) >= batch_size:
      _flush()
    if log_every and kept % log_every == 0:
      now = time.perf_counter()
      elapsed = now - start_t
      rate = kept / elapsed if elapsed > 0 else 0.0
      eta = ""
      if max_items is not None and rate > 0:
        remaining = max_items - kept
        eta_s = max(0.0, remaining / rate)
        eta = f" eta={eta_s/60:.1f}m"
      prefix = f"{log_prefix} " if log_prefix else ""
      print(
        f"{prefix}COLLECT kept={kept} skip_label={skipped_label} skip_audio={skipped_audio} "
        f"rate={rate:.2f}/s elapsed={elapsed/60:.1f}m{eta}",
        flush=True,
      )
      last_log = now
  _flush()

  if key_fn is not None:
    for e, n, y in grouped.values():
      embeddings.append(e / float(n))
      labels.append(y)

  if not embeddings:
    return torch.empty((0, 1)), torch.empty((0,))
  X = torch.stack(embeddings, dim=0)
  y_tensor = torch.tensor(labels)
  return X, y_tensor


def _split_fallback(ds, seed: int, test_size: float = 0.2, val_size: float = 0.1):
  from datasets import Dataset

  if not isinstance(ds, Dataset):
    ds = Dataset.from_list(list(ds))
  tmp = ds.train_test_split(test_size=test_size, seed=seed)
  val_frac = val_size / (1 - test_size)
  tmp2 = tmp["train"].train_test_split(test_size=val_frac, seed=seed)
  return tmp2["train"], tmp2["test"], tmp["test"]


def _evaluate_binary_task(
  name: str,
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  X_val: torch.Tensor,
  y_val: torch.Tensor,
  X_test: torch.Tensor,
  y_test: torch.Tensor,
  *,
  weight_decays: Sequence[float],
  lr: float,
  epochs: int,
  batch_size: int,
  device: torch.device,
  probe_type: str,
  mlp_hidden: Sequence[int],
  mlp_dropout: float,
  probe_backend: str,
  c_grid: Sequence[float],
  cv: int,
  seed: int,
  groups: Optional[Sequence[Any]] = None,
  logreg_solver: str = "lbfgs",
  logreg_max_iter: int = 1000,
  logreg_tol: float = 1e-4,
) -> Dict[str, float]:
  if probe_backend == "sklearn":
    if probe_type != "linear":
      _die("MLP probes require --probe-backend torch.")
    model, best = _train_binary_probe_sklearn(
      X_train,
      y_train,
      X_val,
      y_val,
      c_grid=c_grid,
      cv=cv,
      seed=seed,
      groups=groups,
      solver=logreg_solver,
      max_iter=logreg_max_iter,
      tol=logreg_tol,
    )
    test_logits = torch.from_numpy(_sklearn_decision(model, X_test))
  else:
    model, best = _train_binary_probe(
      X_train,
      y_train,
      X_val,
      y_val,
      weight_decays=weight_decays,
      lr=lr,
      epochs=epochs,
      batch_size=batch_size,
      device=device,
      probe_type=probe_type,
      mlp_hidden=mlp_hidden,
      mlp_dropout=mlp_dropout,
    )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
      test_logits = model(X_test.to(device)).squeeze(1).cpu()
  return {
    "task": name,
    "auc": _roc_auc_score(y_test, test_logits),
    "ap": _average_precision(y_test, test_logits),
    "acc": _accuracy(y_test, test_logits),
    "y_test": y_test.detach().cpu(),
    "test_logits": test_logits.detach().cpu(),
    "val_auc": best.get("val_auc", float("nan")),
    "weight_decay": best.get("weight_decay", float("nan")),
    "C": best.get("C", float("nan")),
  }


def _evaluate_regression_task(
  name: str,
  X_train: torch.Tensor,
  y_train: torch.Tensor,
  X_val: torch.Tensor,
  y_val: torch.Tensor,
  X_test: torch.Tensor,
  y_test: torch.Tensor,
  *,
  weight_decays: Sequence[float],
  lr: float,
  epochs: int,
  batch_size: int,
  device: torch.device,
  probe_type: str,
  mlp_hidden: Sequence[int],
  mlp_dropout: float,
  probe_backend: str,
  alpha_grid: Sequence[float],
  cv: int,
  seed: int,
  groups: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
  if probe_backend == "sklearn":
    if probe_type != "linear":
      _die("MLP probes require --probe-backend torch.")
    model, best = _train_regression_probe_sklearn(
      X_train,
      y_train,
      X_val,
      y_val,
      alpha_grid=alpha_grid,
      cv=cv,
      seed=seed,
      groups=groups,
    )
    test_pred = torch.from_numpy(np.asarray(model.predict(_as_numpy(X_test))))
  else:
    model, best = _train_regression_probe(
      X_train,
      y_train,
      X_val,
      y_val,
      weight_decays=weight_decays,
      lr=lr,
      epochs=epochs,
      batch_size=batch_size,
      device=device,
      probe_type=probe_type,
      mlp_hidden=mlp_hidden,
      mlp_dropout=mlp_dropout,
    )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
      test_pred = model(X_test.to(device)).squeeze(1).cpu()
  return {
    "task": name,
    "r2": _r2_score(y_test, test_pred),
    "mae": _mae(y_test, test_pred),
    "val_r2": best.get("val_r2", float("nan")),
    "weight_decay": best.get("weight_decay", float("nan")),
    "alpha": best.get("alpha", float("nan")),
  }


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Evaluate distilled or full HeAR embeddings on HF datasets.")
  ap.add_argument(
    "--embedding-model",
    choices=["distilled", "hear-hf"],
    default="distilled",
    help="Embedding model source: distilled checkpoint (`distilled`) or full HeAR from Hugging Face (`hear-hf`).",
  )
  ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/hear_vit_s/ckpt_final.pt"))
  ap.add_argument("--hf-model-id", type=str, default="google/hear-pytorch", help="HF model id for --embedding-model hear-hf.")
  ap.add_argument(
    "--embedding-head",
    choices=["proj", "student"],
    default="proj",
    help="Distilled-only: projection head output (`proj`) or raw student features before projection (`student`).",
  )
  ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
  ap.add_argument("--unsafe-load", action="store_true", help="Allow unsafe torch.load if needed.")
  ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
  ap.add_argument("--probe-batch-size", type=int, default=256, help="Probe batch size.")
  ap.add_argument("--probe-epochs", type=int, default=15, help="Probe epochs.")
  ap.add_argument("--probe-lr", type=float, default=1e-2, help="Probe learning rate.")
  ap.add_argument("--probe-weight-decays", type=str, default="0,1e-4,1e-3,1e-2", help="Comma-separated weight decays (torch backend).")
  ap.add_argument("--probe-type", choices=["linear", "mlp"], default="linear", help="Probe head type.")
  ap.add_argument("--probe-backend", choices=["sklearn", "torch"], default="sklearn", help="Probe implementation backend.")
  ap.add_argument("--probe-c-grid", type=str, default="1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000", help="Comma-separated C grid for sklearn LogisticRegression.")
  ap.add_argument("--probe-alpha-grid", type=str, default="1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000", help="Comma-separated alpha grid for sklearn Ridge.")
  ap.add_argument("--probe-cv", type=int, default=5, help="Cross-validation folds for sklearn probes.")
  ap.add_argument("--probe-logreg-solver", type=str, default="lbfgs", help="LogisticRegression solver (sklearn).")
  ap.add_argument("--probe-logreg-max-iter", type=int, default=5000, help="LogisticRegression max_iter (sklearn).")
  ap.add_argument("--probe-logreg-tol", type=float, default=1e-4, help="LogisticRegression tol (sklearn).")
  ap.add_argument("--probe-mlp-hidden", type=str, default="256", help="Comma-separated MLP hidden sizes.")
  ap.add_argument("--probe-mlp-dropout", type=float, default=0.0, help="MLP dropout.")
  ap.add_argument("--seed", type=int, default=1337)
  ap.add_argument("--log-every", type=int, default=1000, help="Log progress every N kept clips (0 disables).")
  ap.add_argument("--log-decode-errors", action="store_true", help="Log audio decode failures (limited).")
  ap.add_argument("--decode-log-max", type=int, default=10, help="Max decode errors to log per split.")
  ap.add_argument(
    "--force-decode-false",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Force Audio(decode=False) to avoid lazy decoders.",
  )
  ap.add_argument("--cache-dir", type=Path, default=Path("cache/eval_embeddings"), help="Cache directory for embeddings.")
  ap.add_argument("--cache-refresh", action="store_true", help="Recompute cached embeddings.")
  ap.add_argument("--max-train", type=int, default=None)
  ap.add_argument("--max-val", type=int, default=None)
  ap.add_argument("--max-test", type=int, default=None)
  ap.add_argument(
    "--bootstrap-samples",
    type=int,
    default=1000,
    help="Number of bootstrap resamples for AP confidence intervals (0 disables).",
  )
  ap.add_argument(
    "--bootstrap-alpha",
    type=float,
    default=0.05,
    help="Alpha for a (1-alpha) CI (e.g., 0.05 -> 95%% CI).",
  )
  ap.add_argument(
    "--bootstrap-seed",
    type=int,
    default=None,
    help="Seed for bootstrap resampling (defaults to --seed).",
  )
  ap.add_argument("--target-sr", type=int, default=16000)
  ap.add_argument("--clip-seconds", type=float, default=2.0)
  ap.add_argument(
    "--full-clip",
    action="store_true",
    help="Run inference on the full decoded clip as a single input (no 2s crop/chunking).",
  )
  ap.add_argument("--run-fsd50k", action="store_true")
  ap.add_argument("--run-flusense", action="store_true")
  ap.add_argument("--run-coswara", action="store_true")
  ap.add_argument("--fsd50k-id", type=str, default="CLAPv2/FSD50K")
  ap.add_argument("--flusense-id", type=str, default="vtsouval/flusense")
  ap.add_argument("--coswara-id", type=str, default="szzs1693/coswara-data")
  ap.add_argument(
    "--fsd50k-tasks",
    type=str,
    default="cough,breathing,throat-clearing,laughter,speech,sneeze",
  )
  ap.add_argument("--fsd50k-group-keys", type=str, default="index,datasetname", help="Comma-separated meta keys for CV grouping.")
  ap.add_argument("--flusense-group-keys", type=str, default="", help="Comma-separated meta keys for CV grouping.")
  ap.add_argument("--coswara-group-keys", type=str, default="participant_id", help="Comma-separated meta keys for CV grouping.")
  ap.add_argument("--flusense-labels", type=str, default="breathe,cough,gasp,sneeze,sniffle,speech,throat-clearing")
  ap.add_argument("--coswara-audio-types", type=str, default="cough-heavy,cough-shallow")
  ap.add_argument("--coswara-min-quality", type=int, default=0)
  ap.add_argument("--coswara-use-test-status", action="store_true")
  ap.add_argument("--group-by-participant", action="store_true", default=False)
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  _set_seed(args.seed)
  if args.clip_seconds <= 0:
    _die("--clip-seconds must be > 0.")
  device_type, device = _pick_device(args.device)
  repo_root = Path(__file__).resolve().parent
  preprocess_audio = (
    _import_preprocess_audio_full_clip(repo_root)
    if args.full_clip
    else _import_preprocess_audio(repo_root)
  )
  clip_samples = int(round(args.clip_seconds * args.target_sr))
  if clip_samples <= 0:
    _die("--clip-seconds * --target-sr must be at least 1 sample.")
  if (not args.full_clip) and clip_samples > 32000:
    _die(
      "Current HeAR preprocessing only supports up to 32000 samples per inference window.\n"
      "Lower --clip-seconds (or --target-sr) so that clip_seconds * target_sr <= 32000."
    )
  if args.embedding_model == "hear-hf":
    if args.embedding_head != "proj":
      print("Warning: --embedding-head applies only to --embedding-model distilled; ignoring.", flush=True)
    if args.unsafe_load:
      print("Warning: --unsafe-load applies only to --embedding-model distilled; ignoring.", flush=True)

  if args.embedding_model == "distilled":
    if not args.ckpt.exists():
      _die(f"Checkpoint not found: {args.ckpt}")
    student, proj, _ckpt_args, embedding_dim = _load_student_and_proj(
      args.ckpt,
      device,
      allow_unsafe=args.unsafe_load,
      embedding_head=args.embedding_head,
    )
    model_fp = f"distilled-{_ckpt_fingerprint(args.ckpt)}-head{args.embedding_head}"

    def embed_batch_fn(batch: torch.Tensor) -> torch.Tensor:
      spec = preprocess_audio(batch)
      feats = _student_features(student, spec)
      return proj(feats)

  else:
    embed_batch_fn, embedding_dim = _load_hf_hear_embedder(
      model_id=args.hf_model_id,
      preprocess_audio=preprocess_audio,
      device=device,
    )
    model_fp = _hf_model_fingerprint(args.hf_model_id)

  weight_decays = [float(x) for x in args.probe_weight_decays.split(",") if x.strip() != ""]
  mlp_hidden = _parse_int_list(args.probe_mlp_hidden)
  c_grid = _parse_float_list(args.probe_c_grid)
  alpha_grid = _parse_float_list(args.probe_alpha_grid)
  fsd_group_keys = _parse_str_list(args.fsd50k_group_keys)
  flu_group_keys = _parse_str_list(args.flusense_group_keys)
  cos_group_keys = _parse_str_list(args.coswara_group_keys)
  if args.probe_type == "mlp" and not mlp_hidden:
    _die("MLP probe selected but --probe-mlp-hidden is empty.")
  if args.probe_backend == "sklearn" and args.probe_type != "linear":
    _die("Sklearn backend supports only linear probes. Use --probe-backend torch for MLP.")

  print(f"Device: {device_type}")
  if args.embedding_model == "distilled":
    print(f"Embedding model: distilled ({args.ckpt})")
    print(f"Embedding head: {args.embedding_head} (dim={embedding_dim})")
  else:
    print(f"Embedding model: hear-hf ({args.hf_model_id}, dim={embedding_dim})")
  if args.full_clip:
    print(f"Clip mode: full clip (no chunking, whole resampled clip per forward)")
  else:
    print(f"Clip mode: crop ({args.clip_seconds:.2f}s @ {args.target_sr}Hz ({clip_samples} samples))")

  if not (args.run_fsd50k or args.run_flusense or args.run_coswara):
    args.run_fsd50k = args.run_flusense = args.run_coswara = True

  from datasets import load_dataset
  try:
    from datasets import Audio
  except Exception:
    Audio = None  # type: ignore

  def _maybe_cast_audio(ds):
    if not args.force_decode_false:
      return ds
    if Audio is None:
      return ds
    try:
      if "audio" in ds.column_names:
        return ds.cast_column("audio", Audio(decode=False))
    except Exception:
      return ds
    return ds

  if args.run_fsd50k:
    print("\n== FSD50K ==")
    ds_train = _maybe_cast_audio(load_dataset(args.fsd50k_id, split="train"))
    ds_val = _maybe_cast_audio(load_dataset(args.fsd50k_id, split="validation"))
    ds_test = _maybe_cast_audio(load_dataset(args.fsd50k_id, split="test"))

    def _fsd_meta(ex):
      text = ex.get("text")
      if text is None:
        return None
      return {
        "text": text,
        "raw_text": ex.get("raw_text"),
        "index": ex.get("index"),
        "datasetname": ex.get("datasetname"),
      }

    X_train_all, meta_train = _get_cached_split(
      dataset_id=args.fsd50k_id,
      split="train",
      ds=ds_train,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="peak",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_train,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_fsd_meta,
      refresh=args.cache_refresh,
    )
    X_val_all, meta_val = _get_cached_split(
      dataset_id=args.fsd50k_id,
      split="validation",
      ds=ds_val,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="peak",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_val,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_fsd_meta,
      refresh=args.cache_refresh,
    )
    X_test_all, meta_test = _get_cached_split(
      dataset_id=args.fsd50k_id,
      split="test",
      ds=ds_test,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="peak",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_test,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_fsd_meta,
      refresh=args.cache_refresh,
    )
    task_names = [_normalize_task_name(t) for t in args.fsd50k_tasks.split(",") if t.strip()]
    keyword_map = {
      "cough": ["cough"],
      "breathing": ["breathing", "breath"],
      "throat-clearing": ["throat clearing", "throat-clearing"],
      "laughter": ["laughter", "laughing", "giggle"],
      "speech": ["speech", "human voice"],
      "sneeze": ["sneeze", "sneezing"],
      "snoring": ["snoring", "snore"],
    }

    def _fsd_has_any(m: dict, kws: Sequence[str]) -> bool:
      text = str(m.get("text", "")).lower()
      raw = m.get("raw_text") or []
      raw_texts = [str(r).lower() for r in raw if r is not None]
      for k in kws:
        kk = str(k).lower()
        if kk in text:
          return True
        if any(kk in r for r in raw_texts):
          return True
      return False

    results: List[dict] = []
    for task in task_names:
      keywords = keyword_map.get(task, [task])

      def _label_fn_meta(m, kws=keywords):
        return 1 if _fsd_has_any(m, kws) else 0

      groups_train = None
      if args.probe_backend == "sklearn" and fsd_group_keys:
        X_train, y_train, groups_train = _apply_label_fn_with_groups(
          X_train_all,
          meta_train,
          _label_fn_meta,
          fsd_group_keys,
        )
      else:
        X_train, y_train = _apply_label_fn(X_train_all, meta_train, _label_fn_meta)
      X_val, y_val = _apply_label_fn(X_val_all, meta_val, _label_fn_meta)
      X_test, y_test = _apply_label_fn(X_test_all, meta_test, _label_fn_meta)
      if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping {task} (no samples).")
        continue
      res = _evaluate_binary_task(
        task,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        weight_decays=weight_decays,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        device=device,
        probe_type=args.probe_type,
        mlp_hidden=mlp_hidden,
        mlp_dropout=args.probe_mlp_dropout,
        probe_backend=args.probe_backend,
        c_grid=c_grid,
        cv=args.probe_cv,
        seed=args.seed,
        groups=groups_train,
        logreg_solver=args.probe_logreg_solver,
        logreg_max_iter=args.probe_logreg_max_iter,
        logreg_tol=args.probe_logreg_tol,
      )
      results.append(res)

    if results:
      map_val = float(np.mean([float(r["ap"]) for r in results]))
      per_task_ci: Dict[str, Tuple[float, float]] = {}
      map_ci = (float("nan"), float("nan"))
      n_valid = 0
      if args.bootstrap_samples and int(args.bootstrap_samples) > 0:
        boot_seed = int(args.seed if args.bootstrap_seed is None else args.bootstrap_seed)
        boot_alpha = float(args.bootstrap_alpha)
        task_items = []
        for r in results:
          y = np.asarray(r["y_test"].flatten(), dtype=np.int8)
          s = np.asarray(r["test_logits"].flatten(), dtype=np.float64)
          task_items.append((str(r["task"]), y, s))
        try:
          per_task_ci, map_ci, n_valid = _bootstrap_ap_cis_shared(
            task_items,
            n_bootstrap=int(args.bootstrap_samples),
            seed=boot_seed,
            alpha=boot_alpha,
          )
        except Exception as exc:  # noqa: BLE001
          print(f"Warning: bootstrap CI failed for FSD50K: {exc}", flush=True)
          per_task_ci, map_ci, n_valid = {}, (float("nan"), float("nan")), 0

      if per_task_ci and not (math.isnan(map_ci[0]) or math.isnan(map_ci[1])):
        ap_str = _format_ap_ci(map_val, map_ci[0], map_ci[1])
        print(f"mAP: ap={ap_str} (bootstrap n={n_valid}/{int(args.bootstrap_samples)})")
      else:
        print(f"mAP: ap={map_val:.4f}")

      for res in results:
        task = str(res["task"])
        lo, hi = per_task_ci.get(task, (float("nan"), float("nan")))
        ap_str = _format_ap_ci(float(res["ap"]), float(lo), float(hi))
        reg = _format_reg(res)
        print(
          f"{task}: auc={res['auc']:.4f} ap={ap_str} acc={res['acc']:.4f} "
          f"(val_auc={res['val_auc']:.4f}, {reg})"
        )

  if args.run_flusense:
    print("\n== FluSense ==")
    ds = _maybe_cast_audio(load_dataset(args.flusense_id, split="train"))
    target_labels = [t.strip() for t in args.flusense_labels.split(",") if t.strip()]
    try:
      split = ds.train_test_split(test_size=0.2, seed=args.seed, stratify_by_column="label")
      split2 = split["train"].train_test_split(test_size=0.125, seed=args.seed, stratify_by_column="label")
      ds_train, ds_val, ds_test = split2["train"], split2["test"], split["test"]
    except Exception:
      ds_train, ds_val, ds_test = _split_fallback(ds, args.seed, test_size=0.2, val_size=0.1)

    def _flu_meta(ex):
      lab = ex.get("label")
      if lab is None:
        return None
      return {
        "label": lab,
        "site": ex.get("site") or ex.get("location") or ex.get("device_id"),
        "clip_id": ex.get("clip_id") or ex.get("file_id") or ex.get("id"),
      }

    X_train_all, meta_train = _get_cached_split(
      dataset_id=args.flusense_id,
      split="train",
      ds=ds_train,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="center",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_train,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_flu_meta,
      refresh=args.cache_refresh,
    )
    X_val_all, meta_val = _get_cached_split(
      dataset_id=args.flusense_id,
      split="val",
      ds=ds_val,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="center",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_val,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_flu_meta,
      refresh=args.cache_refresh,
    )
    X_test_all, meta_test = _get_cached_split(
      dataset_id=args.flusense_id,
      split="test",
      ds=ds_test,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="center",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_test,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_flu_meta,
      refresh=args.cache_refresh,
    )

    results: List[dict] = []
    for label in target_labels:
      def _label_fn_meta(m, lbl=label):
        lab = m.get("label")
        if lab is None:
          return None
        return 1 if lab == lbl else 0

      groups_train = None
      if args.probe_backend == "sklearn" and flu_group_keys:
        X_train, y_train, groups_train = _apply_label_fn_with_groups(
          X_train_all,
          meta_train,
          _label_fn_meta,
          flu_group_keys,
        )
      else:
        X_train, y_train = _apply_label_fn(X_train_all, meta_train, _label_fn_meta)
      X_val, y_val = _apply_label_fn(X_val_all, meta_val, _label_fn_meta)
      X_test, y_test = _apply_label_fn(X_test_all, meta_test, _label_fn_meta)
      if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping {label} (no samples).")
        continue
      res = _evaluate_binary_task(
        label,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        weight_decays=weight_decays,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        device=device,
        probe_type=args.probe_type,
        mlp_hidden=mlp_hidden,
        mlp_dropout=args.probe_mlp_dropout,
        probe_backend=args.probe_backend,
        c_grid=c_grid,
        cv=args.probe_cv,
        seed=args.seed,
        groups=groups_train,
        logreg_solver=args.probe_logreg_solver,
        logreg_max_iter=args.probe_logreg_max_iter,
        logreg_tol=args.probe_logreg_tol,
      )
      results.append(res)
    if results:
      map_val = float(np.mean([float(r["ap"]) for r in results]))
      per_task_ci: Dict[str, Tuple[float, float]] = {}
      map_ci = (float("nan"), float("nan"))
      n_valid = 0
      if args.bootstrap_samples and int(args.bootstrap_samples) > 0:
        boot_seed = int(args.seed if args.bootstrap_seed is None else args.bootstrap_seed)
        boot_alpha = float(args.bootstrap_alpha)
        task_items = []
        for r in results:
          y = np.asarray(r["y_test"].flatten(), dtype=np.int8)
          s = np.asarray(r["test_logits"].flatten(), dtype=np.float64)
          task_items.append((str(r["task"]), y, s))
        try:
          per_task_ci, map_ci, n_valid = _bootstrap_ap_cis_shared(
            task_items,
            n_bootstrap=int(args.bootstrap_samples),
            seed=boot_seed,
            alpha=boot_alpha,
          )
        except Exception as exc:  # noqa: BLE001
          print(f"Warning: bootstrap CI failed for FluSense: {exc}", flush=True)
          per_task_ci, map_ci, n_valid = {}, (float("nan"), float("nan")), 0

      if per_task_ci and not (math.isnan(map_ci[0]) or math.isnan(map_ci[1])):
        ap_str = _format_ap_ci(map_val, map_ci[0], map_ci[1])
        print(f"mAP: ap={ap_str} (bootstrap n={n_valid}/{int(args.bootstrap_samples)})")
      else:
        print(f"mAP: ap={map_val:.4f}")

      for res in results:
        task = str(res["task"])
        lo, hi = per_task_ci.get(task, (float("nan"), float("nan")))
        ap_str = _format_ap_ci(float(res["ap"]), float(lo), float(hi))
        reg = _format_reg(res)
        print(
          f"{task}: auc={res['auc']:.4f} ap={ap_str} acc={res['acc']:.4f} "
          f"(val_auc={res['val_auc']:.4f}, {reg})"
        )

  if args.run_coswara:
    print("\n== Coswara ==")
    ds_train = _maybe_cast_audio(load_dataset(args.coswara_id, "audio", split="train"))
    ds_val = _maybe_cast_audio(load_dataset(args.coswara_id, "audio", split="val"))
    ds_test = _maybe_cast_audio(load_dataset(args.coswara_id, "audio", split="test"))
    allowed_types = [t.strip() for t in args.coswara_audio_types.split(",") if t.strip()]

    def _cos_meta(ex):
      return {
        "participant_id": ex.get("participant_id"),
        "audio_type": ex.get("audio_type"),
        "quality_score": ex.get("quality_score"),
        "covid_status": ex.get("covid_status"),
        "test_status": ex.get("test_status"),
        "gender": ex.get("gender"),
        "age": ex.get("age"),
        "smoking_status": ex.get("smoking_status", ex.get("smoking", ex.get("smoker", ex.get("tobacco")))),
      }

    X_train_all, meta_train = _get_cached_split(
      dataset_id=args.coswara_id,
      split="train",
      ds=ds_train,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="center",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_train,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_cos_meta,
      refresh=args.cache_refresh,
    )
    X_val_all, meta_val = _get_cached_split(
      dataset_id=args.coswara_id,
      split="val",
      ds=ds_val,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="center",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_val,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_cos_meta,
      refresh=args.cache_refresh,
    )
    X_test_all, meta_test = _get_cached_split(
      dataset_id=args.coswara_id,
      split="test",
      ds=ds_test,
      cache_dir=args.cache_dir,
      model_fp=model_fp,
      target_sr=args.target_sr,
      clip_seconds=args.clip_seconds,
      crop="center",
      full_clip=args.full_clip,
      batch_size=args.batch_size,
      max_items=args.max_test,
      embedding_dim=embedding_dim,
      log_every=args.log_every,
      log_decode_errors=args.log_decode_errors,
      decode_log_max=args.decode_log_max,
      embed_batch_fn=embed_batch_fn,
      device=device,
      meta_fn=_cos_meta,
      refresh=args.cache_refresh,
    )

    def _allowed(meta):
      if meta.get("audio_type") not in allowed_types:
        return False
      if args.coswara_min_quality > 0:
        q = meta.get("quality_score")
        if q is None or q < args.coswara_min_quality:
          return False
      return True

    def _covid_label(meta):
      if not _allowed(meta):
        return None
      status = meta.get("test_status") if args.coswara_use_test_status else meta.get("covid_status")
      if status is None:
        return None
      s = str(status).lower()
      if "positive" in s:
        return 1
      if "negative" in s or "healthy" in s:
        return 0
      return None

    def _gender_label(meta):
      if not _allowed(meta):
        return None
      g = meta.get("gender")
      if g is None:
        return None
      s = str(g).lower()
      if s.startswith("m"):
        return 1
      if s.startswith("f"):
        return 0
      return None

    def _smoking_label(meta):
      if not _allowed(meta):
        return None
      v = meta.get("smoking_status")
      if v is None:
        return None
      s = str(v).lower()
      if s in {"y", "yes", "true", "1", "current", "smoker"}:
        return 1
      if s in {"n", "no", "false", "0", "never", "non-smoker", "nonsmoker"}:
        return 0
      return None

    def _age_label(meta):
      if not _allowed(meta):
        return None
      v = meta.get("age")
      if v is None:
        return None
      try:
        return float(v)
      except Exception:
        return None

    tasks: List[Tuple[str, Callable[[dict], Optional[Any]], str]] = [
      ("covid", _covid_label, "binary"),
      ("sex", _gender_label, "binary"),
      ("smoking", _smoking_label, "binary"),
      ("age", _age_label, "regression"),
    ]
    for name, label_fn, kind in tasks:
      groups_train = None
      if args.group_by_participant:
        X_train, y_train = _group_by_participant(X_train_all, meta_train, label_fn)
        X_val, y_val = _group_by_participant(X_val_all, meta_val, label_fn)
        X_test, y_test = _group_by_participant(X_test_all, meta_test, label_fn)
      else:
        X_train, y_train, groups_train = _apply_label_fn_with_groups(
          X_train_all,
          meta_train,
          label_fn,
          group_keys=cos_group_keys,
        )
        X_val, y_val = _apply_label_fn(X_val_all, meta_val, label_fn)
        X_test, y_test = _apply_label_fn(X_test_all, meta_test, label_fn)
      if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"Skipping {name} (no samples).")
        continue
      if kind == "binary":
        res = _evaluate_binary_task(
          name,
          X_train,
          y_train,
          X_val,
          y_val,
          X_test,
          y_test,
          weight_decays=weight_decays,
          lr=args.probe_lr,
          epochs=args.probe_epochs,
          batch_size=args.probe_batch_size,
          device=device,
          probe_type=args.probe_type,
          mlp_hidden=mlp_hidden,
          mlp_dropout=args.probe_mlp_dropout,
          probe_backend=args.probe_backend,
          c_grid=c_grid,
          cv=args.probe_cv,
          seed=args.seed,
          groups=groups_train,
          logreg_solver=args.probe_logreg_solver,
          logreg_max_iter=args.probe_logreg_max_iter,
          logreg_tol=args.probe_logreg_tol,
        )
        reg = _format_reg(res)
        print(
          f"{name}: auc={res['auc']:.4f} ap={res['ap']:.4f} acc={res['acc']:.4f} "
          f"(val_auc={res['val_auc']:.4f}, {reg})"
        )
      else:
        res = _evaluate_regression_task(
          name,
          X_train,
          y_train,
          X_val,
          y_val,
          X_test,
          y_test,
          weight_decays=weight_decays,
          lr=args.probe_lr,
          epochs=args.probe_epochs,
          batch_size=args.probe_batch_size,
          device=device,
          probe_type=args.probe_type,
          mlp_hidden=mlp_hidden,
          mlp_dropout=args.probe_mlp_dropout,
          probe_backend=args.probe_backend,
          alpha_grid=alpha_grid,
          cv=args.probe_cv,
          seed=args.seed,
          groups=groups_train,
        )
        reg = _format_reg(res)
        print(f"{name}: r2={res['r2']:.4f} mae={res['mae']:.4f} (val_r2={res['val_r2']:.4f}, {reg})")


if __name__ == "__main__":
  main()
