# HeAR Distillation Datalake

This subdirectory is the extractable/open-source unit for the distillation pipeline.
The primary driver is `datalake/run_lake.py`, which orchestrates streaming LAION audio and training the Canon-augmented ViT-S student.

## What Was Consolidated

- `datalake/run_lake.py`: adaptive orchestration of streamer + trainer.
- `stream_laion_audio_clips.py`: LAION-Audio streaming + shard writer.
- `distill_hear_vit_s_canon2d.py`: student distillation training script.
- `evaluate_distilled_hear.py`: downstream benchmark/evaluation on HF datasets.
- `benchmark_student_vs_hear.py`: throughput benchmark for student (no projection head) vs full HeAR.
- `datalake/README.md`: focused runner documentation.
- Release scaffolding: `.env`, `.env.example`, `requirements.txt`, `.gitignore`.

## Technical Changes (Canon2D)

The training stack in `distill_hear_vit_s_canon2d.py` includes 2D Canon support in addition to the original 1D Canon path:

- `--canon-2d` applies depthwise `Conv2d` over patch-token spatial grids.
- Time-axis causal mode is supported via `--canon-causal`.
- Canon placement flags (`--canon-a`, `--canon-b`, `--canon-c`, `--canon-d`, `--canon-abcd`) are retained.
- Shape guards are included; if grid/token assumptions do not match, Canon2D safely falls back to 1D Canon.

## EMA-Driven LR Scheduler

`distill_hear_vit_s_canon2d.py` uses a two-part LR policy:

- Base LR schedule:
  - `lr_base = --lr * multiplier(step)`
  - Supports `--lr-schedule none|cosine`
  - Optional linear warmup via `--lr-warmup-steps`
  - Cosine floor controlled by `--lr-min-ratio`
- EMA-adapted scaling factor (optional):
  - Enable with `--lr-gns-adapt`
  - The script periodically estimates GNS (`--gns-every`) from two gradient samples and derives an approximate optimal batch size.
  - It maintains `ema_opt_batch` with `--lr-gns-ema-beta`.
  - Update rule target is:
    - `raw_factor = ref_batch / ema_opt_batch`
    - `ref_batch` is `--lr-gns-ref-batch` (or `batch_size * grad_accum` when unset)
  - The applied factor is clamped to:
    - `[--lr-gns-min-factor, --lr-gns-max-factor]`
  - Factor updates only happen after:
    - at least `--lr-gns-min-samples` estimates, and
    - every `--lr-gns-update-every` steps

Final optimizer LR is:

`lr_current = lr_base * lr_gns_factor`

Notes:

- If `--lr-gns-adapt` is not set, `lr_gns_factor` remains `1.0`.
- GNS metrics can still be logged (controlled by `--gns-every`) even when LR adaptation is disabled.

## Repository Layout

- `datalake/run_lake.py`
- `datalake/README.md`
- `stream_laion_audio_clips.py`
- `distill_hear_vit_s_canon2d.py`
- `evaluate_distilled_hear.py`
- `benchmark_student_vs_hear.py`
- `.env.example`
- `.env` (local only)
- `requirements.txt`
- `.gitignore`

## Setup

1. Create and activate your environment.
2. Install PyTorch separately (CPU/CUDA wheel choice is platform dependent).
3. Install Python deps:

```bash
pip install -r requirements.txt
```

4. Configure environment values:

```bash
cp .env.example .env
# edit .env
```

5. Ensure `ffmpeg` is on PATH.
6. Ensure the `hear` codebase is available at either:
   - `./hear`
   - `../hear`

## Common Commands

Run these from `distillation/`.

### 1) Standard Datalake Training

```bash
uv run python3 datalake/run_lake.py \
  --data-dir data/laion_audio_lake \
  --num-streams 6 \
  --min-streams 1 \
  --reserve-low-clips 150000 \
  --reserve-high-clips 450000 \
  --train-out checkpoints/hear_vit_s_lake \
  --train-batch-size 64 \
  --train-grad-accum 1 \
  --train-num-workers 8 \
  --train-extra-args "--device cuda --max-steps 200000 --canon --canon-2d --canon-abcd --shuffle-shards"
```

Most-used flags:

- `--data-dir`: shard lake location.
- `--num-streams` / `--min-streams`: max and floor concurrent stream workers.
- `--reserve-low-clips` / `--reserve-high-clips`: hysteresis band for stream scaling.
- `--train-out`: checkpoint output directory.
- `--train-extra-args`: forwarded directly to `distill_hear_vit_s_canon2d.py`.

### 2) High-Throughput Capped Lake + Resume + EMA-LR Adapt (your pattern)

Set auth in `.env` first (`HF_TOKEN=...`). Avoid passing tokens directly in shell history.

```bash
uv run python3 datalake/run_lake.py \
  --data-dir data/laion_audio_lake2 \
  --num-streams 6 \
  --min-streams 4 \
  --lake-max-gb 60 \
  --lake-resume-fraction 0.5 \
  --prune-consumed \
  --prune-every-sec 15 \
  --reserve-low-clips 100000 \
  --reserve-high-clips 350000 \
  --hf-transfer \
  --train-out checkpoints/hear_vit_s_lake \
  --train-batch-size 128 \
  --train-grad-accum 1 \
  --train-num-workers 4 \
  --stream-extra-args "--progress-every 100" \
  --train-extra-args "--log-every 10 --val-fraction 0 --val-target-clips 10000 --val-defer-start-steps 10 --val-defer-check-every 10 --device cuda --max-steps 200000 --canon --canon-2d --canon-abcd --shuffle-shards --wandb --lr 3e-4 --val-batches 5 --val-every 250 --canon-no-pos-enc --resume-from checkpoints/hear_vit_s_lake/ckpt_latest.pt --lr-gns-adapt --lr-gns-ema-beta 0.995 --lr-gns-min-samples 100 --lr-gns-update-every 1000 --lr-gns-min-factor 0.1 --lr-gns-max-factor 1.0"
```

What this configuration does:

- caps non-validation lake size at `60 GB` and resumes streaming near `30 GB` (`0.5`).
- keeps reserve in a tighter working band (`100k` to `350k` clips).
- uses larger training batches (`128`) for higher throughput.
- resumes trainer weights from a prior checkpoint.
- enables EMA-smoothed GNS LR adaptation with conservative update cadence.

### 3) Evaluate Distilled Checkpoint

```bash
uv run python3 evaluate_distilled_hear.py \
  --embedding-model distilled \
  --ckpt checkpoints/hear_vit_s_lake/ckpt_final.pt \
  --embedding-head proj \
  --device cuda \
  --batch-size 64 \
  --probe-backend sklearn
```

### 4) Evaluate Against Full HeAR Baseline

```bash
uv run python3 evaluate_distilled_hear.py \
  --embedding-model hear-hf \
  --hf-model-id google/hear-pytorch \
  --device cuda \
  --batch-size 64 \
  --probe-backend sklearn
```

### 5) Benchmark Student (No Projection Head) vs Full HeAR

```bash
uv run python3 benchmark_student_vs_hear.py \
  --ckpt checkpoints/hear_vit_s_lake/ckpt_final.pt \
  --hear-model-id google/hear-pytorch \
  --device cuda \
  --num-clips 128 \
  --batch-size 128 \
  --warmup 1 \
  --repeats 5 \
  --save-json results/benchmark_student_vs_hear.json
```

This benchmark reports:

- student backbone embedding throughput using `student` features only (no `proj` head),
- full HeAR `pooler_output` throughput,
- relative model-only and end-to-end speedup ratios.

Notes:

- By default, evaluator runs all three datasets (`FSD50K`, `FluSense`, `Coswara`) unless you pass specific `--run-*` flags.
- Use `--cache-dir` and optional `--cache-refresh` to manage embedding cache reuse.
- `run_lake.py` loads `.env` by default, so `HF_TOKEN` and related values can be set once.
