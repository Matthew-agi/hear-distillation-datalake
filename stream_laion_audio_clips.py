#!/usr/bin/env python3
from __future__ import annotations

"""
Stream LAION-Audio-300M from Hugging Face and write fixed-length audio clips.

Default output format is sharded tar files:
  out_dir/shard-000000.tar, shard-000001.tar, ...
Each shard contains pairs:
  <clip_id>.wav
  <clip_id>.json

Example:
  python3 stream_laion_audio_clips.py \
    --out data/laion_audio_2s \
    --num-clips 500000 \
    --shuffle-buffer 2000 \
    --shard-size 1000

Notes:
  - Uses datasets streaming (does not download the whole dataset).
  - Decodes MP3 bytes using ffmpeg (recommended in cloud environments).
  - If a source clip is longer than the target duration, multiple 2s clips
    are produced (non-overlapping), plus one final overlapped clip if the
    remaining tail is >= 1s.
"""

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import threading


def _die(msg: str) -> "None":
  raise SystemExit(msg)


def _which(cmd: str) -> Optional[str]:
  from shutil import which

  return which(cmd)


def _run_ffmpeg_decode_mp3_to_s16le(
  mp3_bytes: bytes, *, sample_rate: int, channels: int, timeout_s: int
) -> Tuple[bool, bytes, str]:
  """
  Decode MP3 bytes -> raw signed 16-bit PCM bytes via ffmpeg.

  The shard payload is PCM16 WAV, so decoding straight to s16le avoids a 4x
  float32 intermediate and a second float-to-int conversion for every clip.
  Returns (ok, pcm_bytes, err_msg).
  """
  ffmpeg = _which("ffmpeg") or "ffmpeg"
  cmd = [
    ffmpeg,
    "-hide_banner",
    "-loglevel",
    "error",
    "-nostdin",
    "-i",
    "pipe:0",
    "-f",
    "s16le",
    "-acodec",
    "pcm_s16le",
    "-ac",
    str(channels),
    "-ar",
    str(sample_rate),
    "pipe:1",
  ]
  try:
    p = subprocess.run(
      cmd,
      input=mp3_bytes,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      timeout=timeout_s,
      check=False,
    )
  except subprocess.TimeoutExpired:
    return False, b"", "ffmpeg_timeout"
  if p.returncode != 0:
    return False, b"", (p.stderr.decode("utf-8", "replace") or "ffmpeg_failed")[:2000]
  return True, p.stdout, ""


def _wav_bytes_from_pcm16_mono(pcm16le: bytes, *, sample_rate: int) -> bytes:
  """Wrap little-endian mono PCM16 samples in a standard WAV container."""
  import wave

  buf = io.BytesIO()
  with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # int16
    wf.setframerate(int(sample_rate))
    wf.writeframes(pcm16le)
  return buf.getvalue()


@dataclass(frozen=True)
class ClipResult:
  clip_id: str
  wav_bytes: bytes
  meta: Dict[str, Any]


def _iter_laion_stream(
  *,
  split: str,
  shuffle_buffer: int,
  seed: int,
  skip: int = 0,
  num_shards: int = 1,
  shard_index: int = 0,
  strict_partition: bool = True,
) -> Iterator[Dict[str, Any]]:
  from datasets import Features, Value, load_dataset

  ds = load_dataset("laion/LAION-Audio-300M", split=split, streaming=True)
  if num_shards > 1:
    if not hasattr(ds, "shard"):
      if strict_partition:
        _die("Dataset streaming iterator does not support .shard(); cannot guarantee non-overlap across streams.")
      print("WARN: .shard() unavailable; multi-stream overlap may occur.", file=sys.stderr)
    else:
      try:
        try:
          ds = ds.shard(num_shards=num_shards, index=shard_index)
        except TypeError:
          ds = ds.shard(num_shards, shard_index)
      except Exception as exc:  # noqa: BLE001
        if strict_partition:
          _die(f"Failed to apply stream partitioning via .shard(): {exc}")
        print(f"WARN: failed to shard stream partition ({exc}); overlap may occur.", file=sys.stderr)
  # Keep the embedded MP3 payload as a raw Arrow struct. Casting to
  # Audio(decode=False) still enters torchcodec's encoder on some datasets 5.x
  # streaming paths; plain Features avoids both decoding and that dependency.
  features = ds.features.copy()
  features["audio.mp3"] = {"bytes": Value("binary"), "path": Value("string")}
  ds = ds.cast(Features(features))
  if shuffle_buffer and shuffle_buffer > 0:
    ds = ds.shuffle(buffer_size=int(shuffle_buffer), seed=int(seed))
  if skip and skip > 0 and hasattr(ds, "skip"):
    try:
      ds = ds.skip(int(skip))
    except Exception:
      pass
  return iter(ds)


def _segment_starts(
  total_samples: int, *, clip_samples: int, overlap_threshold_samples: int
) -> List[int]:
  if total_samples < clip_samples:
    return []
  n_full = total_samples // clip_samples
  starts = [i * clip_samples for i in range(n_full)]
  remainder = total_samples - n_full * clip_samples
  if remainder >= overlap_threshold_samples:
    last_start = total_samples - clip_samples
    if not starts or last_start != starts[-1]:
      starts.append(last_start)
  return starts


def _make_clips_from_example(
  ex: Dict[str, Any],
  *,
  clip_samples: int,
  sample_rate: int,
  timeout_s: int,
  overlap_threshold_samples: int,
) -> Optional[List[ClipResult]]:
  audio = ex.get("audio.mp3") or {}
  mp3_bytes = audio.get("bytes")
  if not isinstance(mp3_bytes, (bytes, bytearray)) or len(mp3_bytes) == 0:
    return None

  ok, pcm_bytes, err = _run_ffmpeg_decode_mp3_to_s16le(
    bytes(mp3_bytes),
    sample_rate=sample_rate,
    channels=1,
    timeout_s=timeout_s,
  )
  if not ok or not pcm_bytes:
    return None

  total_samples = len(pcm_bytes) // 2
  starts = _segment_starts(
    total_samples,
    clip_samples=clip_samples,
    overlap_threshold_samples=overlap_threshold_samples,
  )
  if not starts:
    return None

  # Build a stable-ish id using dataset __key__ plus local offset.
  key = str(ex.get("__key__", ""))
  url = str(ex.get("__url__", ""))
  results: List[ClipResult] = []
  total_clips = len(starts)
  for idx, start in enumerate(starts):
    byte_start = int(start) * 2
    byte_end = byte_start + (int(clip_samples) * 2)
    wav_bytes = _wav_bytes_from_pcm16_mono(
      pcm_bytes[byte_start:byte_end],
      sample_rate=sample_rate,
    )
    clip_id = f"{key}-{start}"
    meta = {
      "clip_id": clip_id,
      "sample_rate": sample_rate,
      "clip_samples": clip_samples,
      "clip_start_sample": int(start),
      "clip_end_sample": int(start + clip_samples),
      "clip_index": int(idx),
      "num_clips_from_source": int(total_clips),
      "source_num_samples": int(total_samples),
      "source_duration_s": float(total_samples) / float(sample_rate),
      "source_key": key,
      "source_url": url,
      "metadata": ex.get("metadata.json", {}),
    }
    results.append(ClipResult(clip_id=clip_id, wav_bytes=wav_bytes, meta=meta))

  return results


def _write_tar_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
  info = tarfile.TarInfo(name=name)
  info.size = len(data)
  info.mtime = int(time.time())
  tar.addfile(info, io.BytesIO(data))


def _open_shard(out_dir: Path, shard_idx: int) -> Tuple[tarfile.TarFile, Path, Path]:
  out_dir.mkdir(parents=True, exist_ok=True)
  shard_path = out_dir / f"shard-{shard_idx:06d}.tar"
  tmp_path = out_dir / f"shard-{shard_idx:06d}.tar.tmp"
  if tmp_path.exists():
    try:
      tmp_path.unlink()
    except Exception:
      pass
  return tarfile.open(tmp_path, mode="w"), tmp_path, shard_path


def _finalize_shard(
  tar: Optional[tarfile.TarFile],
  tmp_path: Optional[Path],
  final_path: Optional[Path],
  *,
  has_data: bool,
) -> None:
  if tar is None:
    return
  try:
    tar.close()
  finally:
    if tmp_path is None:
      return
    if has_data and final_path is not None:
      os.replace(tmp_path, final_path)
    else:
      try:
        tmp_path.unlink()
      except Exception:
        pass


def _count_existing_shard_stats(out_dir: Path) -> Tuple[int, int, int]:
  if not out_dir.exists():
    return 0, -1, 0
  shard_paths = sorted(out_dir.glob("shard-*.tar"))
  total = 0
  max_idx = -1
  total_bytes = 0
  for p in shard_paths:
    try:
      stem = p.stem  # shard-000123
      idx = int(stem.split("-")[-1])
      max_idx = max(max_idx, idx)
    except Exception:
      pass
    try:
      total_bytes += int(p.stat().st_size)
    except Exception:
      pass
    try:
      with tarfile.open(p, mode="r") as tf:
        for m in tf:
          if m.isfile() and m.name.endswith(".wav"):
            total += 1
    except Exception:
      continue
  return total, max_idx, total_bytes


def _resume_state_path(out_dir: Path) -> Path:
  return out_dir / "resume_state.json"


def _load_resume_state(out_dir: Path) -> Dict[str, Any]:
  p = _resume_state_path(out_dir)
  if not p.exists():
    return {}
  try:
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}
  except Exception:
    return {}


def _save_resume_state(out_dir: Path, payload: Dict[str, Any]) -> None:
  p = _resume_state_path(out_dir)
  tmp = p.with_suffix(p.suffix + ".tmp")
  tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
  os.replace(tmp, p)


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Stream LAION-Audio-300M and write N random fixed-length clips.")
  ap.add_argument("--out", type=Path, required=True, help="Output directory for shards.")
  ap.add_argument("--num-clips", type=int, default=500_000, help="Number of 2s clips to write (default: 500000, 0 disables clip target).")
  ap.add_argument("--target-bytes", type=int, default=0, help="Optional total output byte target (0 disables byte target).")
  ap.add_argument("--clip-seconds", type=float, default=2.0, help="Clip duration in seconds (default: 2.0).")
  ap.add_argument("--sample-rate", type=int, default=16_000, help="Output sample rate (default: 16000).")
  ap.add_argument("--split", type=str, default="train", help='Dataset split (default: "train").')
  ap.add_argument("--shuffle-buffer", type=int, default=2_000, help="Shuffle buffer size for streaming (default: 2000).")
  ap.add_argument("--seed", type=int, default=1337, help="RNG seed (default: 1337).")
  ap.add_argument("--shard-size", type=int, default=1000, help="Clips per tar shard (default: 1000).")
  ap.add_argument("--progress-every", type=int, default=500, help="Log progress every N written clips (default: 500).")
  ap.add_argument("--ffmpeg-timeout", type=int, default=60, help="Per-example ffmpeg decode timeout seconds (default: 60).")
  ap.add_argument("--decode-workers", type=int, default=2, help="Concurrent ffmpeg decoders per stream worker.")
  ap.add_argument("--decode-prefetch", type=int, default=4, help="Maximum ordered decode tasks buffered per stream worker.")
  ap.add_argument("--min-overlap-tail-sec", type=float, default=1.0, help="Only add overlapped clip if tail >= this many seconds (default: 1.0).")
  ap.add_argument("--max-stream-errors", type=int, default=100, help="Max streaming errors before abort (default: 100).")
  ap.add_argument("--max-examples", type=int, default=0, help="Optional cap on streamed examples (0 = no cap).")
  ap.add_argument("--resume", action="store_true", help="Resume by counting existing shards and appending new shards.")
  ap.add_argument(
    "--allow-unsafe-resume",
    action="store_true",
    help="Allow resume without persisted stream offset state (can produce duplicates).",
  )
  ap.add_argument("--overwrite", action="store_true", help="Allow overwriting existing shards when not resuming.")
  ap.add_argument("--num-streams", type=int, default=1, help="Number of parallel stream workers (default: 1).")
  ap.add_argument("--stream-index", type=int, default=0, help="Index of this stream worker [0..num-streams-1].")
  ap.add_argument(
    "--strict-partition",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Fail if stream sharding cannot be applied when --num-streams > 1.",
  )
  ap.add_argument("--spawn-streams", action="store_true", help="Spawn --num-streams worker processes from this one.")
  ap.add_argument("--hf-transfer", action="store_true", help="Enable hf_transfer for faster downloads.")
  ap.add_argument("--hf-token", type=str, default=None, help="HF token (or set HF_TOKEN env).")
  ap.add_argument("--hf-cache", type=Path, default=None, help="Path for HF hub cache (HF_HUB_CACHE).")
  ap.add_argument("--datasets-cache", type=Path, default=None, help="Path for datasets cache (HF_DATASETS_CACHE).")
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  if args.num_clips < 0:
    _die("--num-clips must be >= 0.")
  if args.target_bytes < 0:
    _die("--target-bytes must be >= 0.")
  if args.num_clips <= 0 and args.target_bytes <= 0:
    _die("Set at least one stopping target: --num-clips > 0 or --target-bytes > 0.")
  if args.clip_seconds <= 0:
    _die("--clip-seconds must be > 0.")
  if args.sample_rate <= 0:
    _die("--sample-rate must be > 0.")
  if args.shard_size <= 0:
    _die("--shard-size must be > 0.")
  if args.decode_workers <= 0:
    _die("--decode-workers must be > 0.")
  if args.decode_prefetch < args.decode_workers:
    _die("--decode-prefetch must be >= --decode-workers.")
  if args.min_overlap_tail_sec < 0:
    _die("--min-overlap-tail-sec must be >= 0.")
  if args.max_stream_errors <= 0:
    _die("--max-stream-errors must be > 0.")
  if args.num_streams <= 0:
    _die("--num-streams must be > 0.")
  if not (0 <= args.stream_index < args.num_streams):
    _die("--stream-index must be in [0, num-streams).")

  if args.spawn_streams and args.num_streams > 1:
    # Parent process: spawn N workers and exit.
    argv = list(sys.argv[1:])
    filtered: List[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
      if skip_next:
        skip_next = False
        continue
      if tok == "--spawn-streams":
        continue
      if tok == "--stream-index":
        skip_next = True
        continue
      filtered.append(tok)

    if "--num-streams" not in filtered:
      filtered.extend(["--num-streams", str(args.num_streams)])

    progress_re = re.compile(
      r"^PROGRESS written=(\d+)/(\d+) written_bytes=(\d+)/(\d+) seen=(\d+) skipped=(\d+) errors=(\d+) "
      r"rate=([0-9.]+) clips/s byte_rate=([0-9.]+) MiB/s elapsed=([0-9.]+)s"
    )
    done_re = re.compile(
      r"^DONE written=(\d+) written_bytes=(\d+) seen=(\d+) skipped=(\d+) elapsed=([0-9.]+)s "
      r"rate=([0-9.]+) clips/s byte_rate=([0-9.]+) MiB/s"
    )

    stats = {
      i: {"written": 0, "written_bytes": 0, "seen": 0, "skipped": 0, "errors": 0, "done": False}
      for i in range(args.num_streams)
    }
    lock = threading.Lock()
    last_agg_time = 0.0
    last_agg_written = 0
    start_time = time.time()

    def _maybe_agg(force: bool = False) -> None:
      nonlocal last_agg_time, last_agg_written
      now = time.time()
      total_written = sum(s["written"] for s in stats.values())
      total_written_bytes = sum(s["written_bytes"] for s in stats.values())
      total_seen = sum(s["seen"] for s in stats.values())
      total_skipped = sum(s["skipped"] for s in stats.values())
      total_errors = sum(s["errors"] for s in stats.values())
      if not force:
        if now - last_agg_time < 5 and (total_written - last_agg_written) < 500:
          return
      elapsed = max(1e-9, now - start_time)
      rate = total_written / elapsed
      byte_rate_mib = (total_written_bytes / (1024.0 ** 2)) / elapsed
      total_target = args.num_clips * args.num_streams
      print(
        f"AGG written={total_written}/{total_target} seen={total_seen} "
        f"skipped={total_skipped} errors={total_errors} "
        f"rate={rate:.2f} clips/s byte_rate={byte_rate_mib:.2f} MiB/s elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
      )
      last_agg_time = now
      last_agg_written = total_written

    def _handle_line(idx: int, line: str) -> None:
      line = line.rstrip("\n")
      m = progress_re.match(line)
      if m:
        with lock:
          stats[idx]["written"] = int(m.group(1))
          stats[idx]["written_bytes"] = int(m.group(3))
          stats[idx]["seen"] = int(m.group(5))
          stats[idx]["skipped"] = int(m.group(6))
          stats[idx]["errors"] = int(m.group(7))
          _maybe_agg()
        return
      m = done_re.match(line)
      if m:
        with lock:
          stats[idx]["written"] = int(m.group(1))
          stats[idx]["written_bytes"] = int(m.group(2))
          stats[idx]["seen"] = int(m.group(3))
          stats[idx]["skipped"] = int(m.group(4))
          stats[idx]["done"] = True
          _maybe_agg(force=True)
        return
      if line:
        with lock:
          print(f"[stream {idx}] {line}", file=sys.stderr)

    def _reader(idx: int, pipe) -> None:
      for line in iter(pipe.readline, ""):
        _handle_line(idx, line)
      try:
        pipe.close()
      except Exception:
        pass

    procs: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []
    for idx in range(args.num_streams):
      cmd = [sys.executable, sys.argv[0]] + filtered + ["--stream-index", str(idx)]
      p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
      )
      procs.append(p)
      t = threading.Thread(target=_reader, args=(idx, p.stdout), daemon=True)
      t.start()
      threads.append(t)

    exit_code = 0
    for p in procs:
      rc = p.wait()
      if rc != 0:
        exit_code = rc

    for t in threads:
      t.join(timeout=1.0)

    with lock:
      _maybe_agg(force=True)
    raise SystemExit(exit_code)

  # Cloud assumption: ffmpeg exists; still fail fast if missing.
  if not _which("ffmpeg"):
    _die("ffmpeg not found on PATH.")

  clip_samples = int(round(float(args.clip_seconds) * int(args.sample_rate)))
  overlap_threshold_samples = int(round(args.min_overlap_tail_sec * args.sample_rate))

  if args.hf_transfer:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
  if args.hf_token:
    os.environ["HF_TOKEN"] = args.hf_token
  if args.hf_cache:
    os.environ["HF_HUB_CACHE"] = str(args.hf_cache)
  if args.datasets_cache:
    os.environ["HF_DATASETS_CACHE"] = str(args.datasets_cache)

  out_dir = args.out
  if args.num_streams > 1:
    out_dir = out_dir / f"stream-{args.stream_index:03d}"
  out_dir.mkdir(parents=True, exist_ok=True)

  if not args.resume and not args.overwrite:
    existing = list(out_dir.glob("shard-*.tar"))
    if existing:
      _die(
        f"Found {len(existing)} existing shards in {out_dir}. "
        "Refusing to overwrite without --resume or --overwrite."
      )

  print(f"Streaming: laion/LAION-Audio-300M split={args.split}", file=sys.stderr)
  print(f"Stream worker: {args.stream_index}/{args.num_streams}", file=sys.stderr)
  print(
    f"Target: clips={args.num_clips} bytes={args.target_bytes} "
    f"of {args.clip_seconds:.3f}s ({clip_samples} samples) @ {args.sample_rate}Hz",
    file=sys.stderr,
  )
  print(f"Output: {out_dir} (shard_size={args.shard_size})", file=sys.stderr)
  print(
    f"Decode pipeline: workers={args.decode_workers} prefetch={args.decode_prefetch}",
    file=sys.stderr,
  )
  print(f"HF transfer: {bool(os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1')}", file=sys.stderr)
  print(f"HF token set: {bool(os.environ.get('HF_TOKEN'))}", file=sys.stderr)
  if os.environ.get("HF_HUB_CACHE"):
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}", file=sys.stderr)
  if os.environ.get("HF_DATASETS_CACHE"):
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}", file=sys.stderr)

  resume_written = 0
  resume_written_bytes = 0
  resume_seen = 0
  resume_state_written: Optional[int] = None
  resume_state_written_bytes: Optional[int] = None
  resume_state_next_shard_idx: Optional[int] = None
  shard_idx = 0
  if args.resume:
    manifest_path = out_dir / "manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
      try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
      except Exception:
        manifest = {}
      prev_num_streams = manifest.get("num_streams")
      prev_stream_index = manifest.get("stream_index")
      if prev_num_streams is not None and int(prev_num_streams) != int(args.num_streams):
        _die(
          f"Refusing resume: existing stream manifest uses num_streams={prev_num_streams}, "
          f"requested num_streams={args.num_streams}. Changing num_streams can break non-overlap."
        )
      if prev_stream_index is not None and int(prev_stream_index) != int(args.stream_index):
        _die(
          f"Refusing resume: existing stream manifest uses stream_index={prev_stream_index}, "
          f"requested stream_index={args.stream_index}."
        )
      prev_split = manifest.get("split")
      if prev_split is not None and str(prev_split) != str(args.split):
        _die(f"Refusing resume: existing split={prev_split}, requested split={args.split}.")
      prev_seed = manifest.get("seed")
      if prev_seed is not None and int(prev_seed) != int(args.seed):
        _die(f"Refusing resume: existing seed={prev_seed}, requested seed={args.seed}.")
      prev_shuffle_buffer = manifest.get("shuffle_buffer")
      if prev_shuffle_buffer is not None and int(prev_shuffle_buffer) != int(args.shuffle_buffer):
        _die(
          "Refusing resume: existing shuffle_buffer="
          f"{prev_shuffle_buffer}, requested shuffle_buffer={args.shuffle_buffer}."
        )
      prev_seen = manifest.get("seen")
      if prev_seen is not None:
        try:
          resume_seen = max(resume_seen, int(prev_seen))
        except Exception:
          pass
      prev_num_bytes = manifest.get("num_bytes")
      if prev_num_bytes is not None:
        try:
          resume_written_bytes = max(resume_written_bytes, int(prev_num_bytes))
        except Exception:
          pass
      prev_next_shard_idx = manifest.get("next_shard_idx")
      if prev_next_shard_idx is not None:
        try:
          resume_state_next_shard_idx = int(prev_next_shard_idx)
        except Exception:
          resume_state_next_shard_idx = resume_state_next_shard_idx
    state = _load_resume_state(out_dir)
    state_seen = state.get("seen")
    state_written = state.get("written")
    state_written_bytes = state.get("written_bytes")
    state_next_shard_idx = state.get("next_shard_idx")
    if state_written is not None:
      try:
        resume_state_written = int(state_written)
      except Exception:
        resume_state_written = None
    if state_written_bytes is not None:
      try:
        resume_state_written_bytes = int(state_written_bytes)
      except Exception:
        resume_state_written_bytes = None
    if state_next_shard_idx is not None:
      try:
        resume_state_next_shard_idx = int(state_next_shard_idx)
      except Exception:
        resume_state_next_shard_idx = resume_state_next_shard_idx
    if state_seen is not None:
      try:
        resume_seen = max(resume_seen, int(state_seen))
      except Exception:
        pass
    existing_written, max_idx, existing_written_bytes = _count_existing_shard_stats(out_dir)
    resume_written = max(existing_written, int(resume_state_written or 0))
    resume_written_bytes = max(existing_written_bytes, int(resume_state_written_bytes or 0), resume_written_bytes)
    shard_idx = max(max_idx + 1 if max_idx >= 0 else 0, int(resume_state_next_shard_idx or 0))
    clip_target_reached = bool(args.num_clips > 0 and resume_written >= args.num_clips)
    byte_target_reached = bool(args.target_bytes > 0 and resume_written_bytes >= args.target_bytes)
    if clip_target_reached or byte_target_reached:
      print(
        f"Resume: already have clips={resume_written} bytes={resume_written_bytes}. "
        f"Targets clips={args.num_clips} bytes={args.target_bytes}. Nothing to do.",
        file=sys.stderr,
      )
      return
    if (
      resume_written > 0
      and (
        resume_seen <= 0
        or (resume_state_written is not None and resume_state_written < existing_written)
        or (resume_state_written_bytes is not None and resume_state_written_bytes < existing_written_bytes)
      )
      and (not args.allow_unsafe_resume)
    ):
      _die(
        "Refusing unsafe resume: existing shards found but resume stream offset state is missing/stale.\n"
        "Re-run with --allow-unsafe-resume to proceed (may produce duplicates), "
        "or start a fresh output directory."
      )
    # Ensure we don't overwrite an existing shard index.
    while (out_dir / f"shard-{shard_idx:06d}.tar").exists():
      shard_idx += 1
    print(
      f"Resume: found clips={resume_written} bytes={resume_written_bytes}. "
      f"Starting at shard {shard_idx}. skip_seen={resume_seen}",
      file=sys.stderr,
    )

  stream = _iter_laion_stream(
    split=args.split,
    shuffle_buffer=args.shuffle_buffer,
    seed=args.seed,
    skip=resume_seen,
    num_shards=args.num_streams,
    shard_index=args.stream_index,
    strict_partition=args.strict_partition,
  )

  written = resume_written
  written_bytes_finalized = resume_written_bytes
  seen = resume_seen
  skipped = 0
  stream_errors = 0
  in_shard = 0
  shard: Optional[tarfile.TarFile] = None
  shard_tmp_path: Optional[Path] = None
  shard_final_path: Optional[Path] = None
  persisted_written = resume_written
  persisted_written_bytes = resume_written_bytes
  persisted_seen = resume_seen
  persisted_next_shard_idx = shard_idx
  t0 = time.perf_counter()
  last_state_save_t = 0.0

  def _current_written_bytes() -> int:
    total = int(written_bytes_finalized)
    if shard_tmp_path is not None and shard_tmp_path.exists():
      try:
        total += int(shard_tmp_path.stat().st_size)
      except Exception:
        pass
    return total

  def _target_reached() -> bool:
    clip_done = bool(args.num_clips > 0 and written >= args.num_clips)
    byte_done = bool(args.target_bytes > 0 and _current_written_bytes() >= args.target_bytes)
    return bool(clip_done or byte_done)

  def _commit_finalized_state() -> None:
    nonlocal persisted_written, persisted_written_bytes, persisted_seen, persisted_next_shard_idx
    persisted_written = int(written)
    persisted_written_bytes = int(written_bytes_finalized)
    persisted_seen = int(seen)
    persisted_next_shard_idx = int(shard_idx)

  def _persist_state(*, force: bool = False) -> None:
    nonlocal last_state_save_t
    now = time.perf_counter()
    if (not force) and ((now - last_state_save_t) < 5.0):
      return
    payload = {
      "written": int(persisted_written),
      "written_bytes": int(persisted_written_bytes),
      "seen": int(persisted_seen),
      "skipped": int(skipped),
      "stream_errors": int(stream_errors),
      "num_streams": int(args.num_streams),
      "stream_index": int(args.stream_index),
      "seed": int(args.seed),
      "shuffle_buffer": int(args.shuffle_buffer),
      "split": str(args.split),
      "next_shard_idx": int(persisted_next_shard_idx),
      "updated_unix": int(time.time()),
    }
    try:
      _save_resume_state(out_dir, payload)
      last_state_save_t = now
    except Exception:
      pass

  def _write_results(results: Optional[List[ClipResult]]) -> None:
    nonlocal shard, shard_tmp_path, shard_final_path, shard_idx
    nonlocal written, written_bytes_finalized, in_shard, skipped
    if not results:
      skipped += 1
      return
    remaining = max(1, args.num_clips - written) if args.num_clips > 0 else len(results)
    for res in results[:remaining]:
      if shard is None:
        shard, shard_tmp_path, shard_final_path = _open_shard(out_dir, shard_idx)

      # Write wav + json.
      _write_tar_member(shard, f"{res.clip_id}.wav", res.wav_bytes)
      try:
        meta_bytes = json.dumps(res.meta, ensure_ascii=False, default=str).encode("utf-8")
      except Exception:
        meta_bytes = json.dumps({"clip_id": res.clip_id}, ensure_ascii=False).encode("utf-8")
      _write_tar_member(shard, f"{res.clip_id}.json", meta_bytes)
      written += 1
      in_shard += 1

      if in_shard >= args.shard_size:
        _finalize_shard(shard, shard_tmp_path, shard_final_path, has_data=True)
        if shard_final_path is not None and shard_final_path.exists():
          try:
            written_bytes_finalized += int(shard_final_path.stat().st_size)
          except Exception:
            pass
        shard = None
        shard_tmp_path = None
        shard_final_path = None
        shard_idx += 1
        in_shard = 0
        _commit_finalized_state()
        _persist_state()

      if args.progress_every and written % args.progress_every == 0:
        elapsed = max(1e-9, time.perf_counter() - t0)
        rate = (written - resume_written) / elapsed
        new_bytes = max(0, _current_written_bytes() - resume_written_bytes)
        byte_rate_mib = (new_bytes / (1024.0 ** 2)) / elapsed
        print(
          f"PROGRESS written={written}/{max(args.num_clips, 0)} "
          f"written_bytes={_current_written_bytes()}/{max(args.target_bytes, 0)} "
          f"seen={seen} skipped={skipped} errors={stream_errors} "
          f"rate={rate:.2f} clips/s byte_rate={byte_rate_mib:.2f} MiB/s elapsed={elapsed:.1f}s",
          file=sys.stderr,
          flush=True,
        )
        _persist_state()

      if _target_reached():
        break

  try:
    stream_iter = iter(stream)
    fetched_seen = int(seen)
    source_exhausted = False
    pending: "deque[Tuple[int, Optional[Future[List[ClipResult] | None]]]]" = deque()
    executor = ThreadPoolExecutor(max_workers=args.decode_workers, thread_name_prefix="ffmpeg")
    try:
      while not _target_reached():
        while (not source_exhausted) and len(pending) < args.decode_prefetch:
          if args.max_examples > 0 and fetched_seen >= args.max_examples:
            source_exhausted = True
            break
          try:
            ex = next(stream_iter)
            fetched_seen += 1
          except StopIteration:
            source_exhausted = True
            break
          except Exception as exc:  # noqa: BLE001
            stream_errors += 1
            fetched_seen += 1
            pending.append((fetched_seen, None))
            print(f"WARN stream error: {exc} (errors={stream_errors})", file=sys.stderr)
            if stream_errors >= args.max_stream_errors:
              print("ERROR: too many stream errors, aborting.", file=sys.stderr)
              source_exhausted = True
              break
            stream_iter = iter(
              _iter_laion_stream(
                split=args.split,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.seed,
                skip=fetched_seen,
                num_shards=args.num_streams,
                shard_index=args.stream_index,
                strict_partition=args.strict_partition,
              )
            )
            continue

          future = executor.submit(
            _make_clips_from_example,
            ex,
            clip_samples=clip_samples,
            sample_rate=args.sample_rate,
            timeout_s=int(args.ffmpeg_timeout),
            overlap_threshold_samples=overlap_threshold_samples,
          )
          pending.append((fetched_seen, future))

        if not pending:
          break

        example_seen, future = pending.popleft()
        seen = int(example_seen)
        if future is None:
          skipped += 1
          continue
        try:
          _write_results(future.result())
        except Exception as exc:  # noqa: BLE001
          skipped += 1
          print(f"WARN decode error: {exc}", file=sys.stderr)
    finally:
      while pending:
        _example_seen, future = pending.popleft()
        if future is not None:
          future.cancel()
      executor.shutdown(wait=True, cancel_futures=True)
  finally:
    try:
      _finalize_shard(
        shard,
        shard_tmp_path,
        shard_final_path,
        has_data=(in_shard > 0),
      )
      if in_shard > 0 and shard_final_path is not None and shard_final_path.exists():
        try:
          written_bytes_finalized += int(shard_final_path.stat().st_size)
        except Exception:
          pass
        shard_idx += 1
        in_shard = 0
    except Exception:
      pass
    _commit_finalized_state()
    _persist_state(force=True)

  elapsed = max(1e-9, time.perf_counter() - t0)
  print(
    f"DONE written={written} written_bytes={written_bytes_finalized} seen={seen} skipped={skipped} "
    f"elapsed={elapsed:.1f}s rate={((written-resume_written)/elapsed):.2f} clips/s "
    f"byte_rate={(max(0, written_bytes_finalized-resume_written_bytes)/(1024.0**2))/elapsed:.2f} MiB/s",
    file=sys.stderr,
  )

  # Write a small manifest.
  manifest = {
    "dataset": "laion/LAION-Audio-300M",
    "split": args.split,
    "num_clips": written,
    "clip_seconds": float(args.clip_seconds),
    "sample_rate": int(args.sample_rate),
    "clip_samples": int(clip_samples),
    "min_overlap_tail_sec": float(args.min_overlap_tail_sec),
    "seed": int(args.seed),
    "shuffle_buffer": int(args.shuffle_buffer),
    "shard_size": int(args.shard_size),
    "num_shards": int(shard_idx + (1 if (written > 0 and in_shard > 0) else 0)),
    "num_streams": int(args.num_streams),
    "stream_index": int(args.stream_index),
    "resume_written": int(resume_written),
    "resume_written_bytes": int(resume_written_bytes),
    "num_bytes": int(written_bytes_finalized),
    "seen": int(seen),
    "next_shard_idx": int(shard_idx),
    "created_unix": int(time.time()),
    "host": os.uname().nodename if hasattr(os, "uname") else "",
  }
  (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
  if (not _target_reached()) and args.max_examples <= 0:
    _die(
      "Streaming ended before reaching a configured target "
      f"(clips={written}/{args.num_clips}, bytes={written_bytes_finalized}/{args.target_bytes})."
    )


if __name__ == "__main__":
  main()
