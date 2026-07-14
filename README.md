# HeAR Distillation Datalake

Stream LAION-Audio while distilling Google HeAR into a Canon-augmented ViT-S
student. The default launcher profiles the host, bounds disk use, chooses CPU
and GPU settings, resumes safely, and runs the streamer and trainer together.

## Run on a GPU instance

The only unavoidable manual step is accepting the
[HeAR model terms](https://huggingface.co/google/hear-pytorch) and logging in to
Hugging Face. The model is gated by its publisher.

```bash
git clone https://github.com/Matthew-agi/hear-distillation-datalake.git
cd hear-distillation-datalake
./scripts/bootstrap.sh
.venv/bin/hf auth login
./run.sh
```

The bootstrap script:

1. installs `ffmpeg` on Debian/Ubuntu when necessary;
2. installs `uv` when necessary;
3. creates `.venv` and selects the compatible PyTorch backend from the host's
   installed GPU driver;
4. installs this project;
5. runs a short environment check.

`./run.sh` then starts streaming and training with hardware-aware defaults.

No `.env` file is required. Hugging Face's normal credential store is used, so
the token does not need to be copied into this repository.

## Automatic defaults

`hear-distill run` uses **up to 50% of the filesystem capacity** by default,
subject to a free-space safety floor. The resulting budget is divided among
the train lake, incoming chunks, validation, decay data, checkpoints, and
caches. The Hugging Face cache defaults inside the selected data directory so
it is governed by the same filesystem safety floor. The launcher also derives:

- stream-process count from available CPU cores;
- DataLoader workers from the remaining CPU capacity;
- training batch size and teacher superbatching from GPU memory;
- mixed precision on CUDA (BF16 when the GPU supports it);
- shard size, shuffle buffer, reserve watermarks, and pruning thresholds;
- static-shape compilation for the teacher, preprocessing, and student path.

Inspect the exact plan without starting anything:

```bash
.venv/bin/hear-distill defaults
.venv/bin/hear-distill run --dry-run
```

Change only what matters; unspecified settings remain automatic:

```bash
./run.sh --disk-fraction 0.35 --max-steps 300000
./run.sh --num-streams 4 --train-batch-size 96
```

Unknown `run` options are passed through to `datalake/run_lake.py`. Supplying
`--train-extra-args` opts into the full advanced trainer interface.

## Performance-oriented changes

- CUDA mixed precision is on by default; the old path silently used FP32.
- gradient-noise estimation is off unless requested; the old default ran two
  extra gradient probes every five steps;
- ffmpeg decodes directly to the PCM16 format stored in shards, eliminating a
  float32 intermediate and per-clip float-to-int conversion;
- PCM16 WAV files have a fast decode path in the trainer;
- HeAR preprocessing is bundled, caches invariant STFT/mel tensors, and uses a
  vectorized PCEN recurrence;
- the runner tracks reserve relative to the current invocation, so resumed and
  pruned lakes do not double-count historical consumption;
- static compilation is the fixed-batch default, with OOM backoff retained for
  teacher superbatching.

Run the focused checks and microbenchmark with:

```bash
uv run pytest
uv run python benchmarks/benchmark_audio_pipeline.py --batch-size 16
```

### Adaptive warmup dependency

The learning-rate probe is maintained in the separate
[`adaptive-warmup`](https://github.com/Matthew-agi/adaptive-warmup) package; this
repository does not carry a second implementation. Bootstrap uses an editable
`../adaptive-warmup` checkout when present. On a fresh machine it installs the
tagged standalone release from GitHub.

To propagate an adaptive-warmup update here, release a new tag in that repo and
update its pinned Git reference in `pyproject.toml`. Set
`ADAPTIVE_WARMUP_SOURCE` to test a branch or commit without changing the
checked-in default.

## Commands

```bash
# Check dependencies, GPU visibility, disk, and auth without printing secrets
.venv/bin/hear-distill doctor

# Evaluate a trained checkpoint
uv run python evaluate_distilled_hear.py \
  --embedding-model distilled \
  --ckpt checkpoints/hear_vit_s_lake/ckpt_final.pt \
  --device cuda

# Compare student and teacher throughput
uv run python benchmark_student_vs_hear.py \
  --ckpt checkpoints/hear_vit_s_lake/ckpt_final.pt \
  --device cuda
```

The original Python entrypoints remain available for existing commands. See
[architecture](docs/architecture.md) and [performance notes](docs/performance.md)
for the internal boundaries and tuning rationale.
