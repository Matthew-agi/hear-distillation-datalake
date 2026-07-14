# HeAR Audio Encoder Distillation Trainer

Train a compact audio encoder by distilling
[`google/hear-pytorch`](https://huggingface.co/google/hear-pytorch) into a
Canon-augmented ViT-S while LAION-Audio is streamed into a bounded local data
lake. One launcher detects the host, chooses safe CPU/GPU settings, limits disk
use, and supervises streaming, curation, training, pruning, validation, and
checkpointing.

The normal path requires no configuration beyond Hugging Face authentication:

```bash
git clone https://github.com/Matthew-agi/hear-distillation-datalake.git
cd hear-distillation-datalake
./scripts/bootstrap.sh
.venv/bin/hf auth login
./run.sh
```

## What this repository provides

- A continuously replenished, disk-bounded audio lake instead of a full
  dataset download.
- Hardware-aware defaults for disk, streaming processes, DataLoader workers,
  GPU batch size, mixed precision, and teacher superbatching.
- A HeAR teacher and Canon-augmented ViT-S student trained with embedding,
  contrastive, and relational distillation losses.
- Fast PCM16 streaming and decode paths plus cached, vectorized HeAR
  mel-PCEN preprocessing.
- Static-shape `torch.compile`, fused AdamW, CUDA mixed precision, and optional
  teacher superbatching.
- Optional critical-learning-rate and critical-batch-size warmup through the
  standalone [`adaptive-warmup`](https://github.com/Matthew-agi/adaptive-warmup)
  library.
- Atomic state files, bounded validation/decay reservoirs, checkpoint pruning,
  and guarded resume behavior.

## Requirements

- Linux is recommended for a training instance. CUDA is optional for smoke
  tests but expected for practical training.
- Python 3.10 or newer. Bootstrap creates a Python 3.11 environment by default.
- `git`, `curl`, and enough local storage for the selected lake budget.
- Access to the gated HeAR model. Accept the
  [model terms](https://huggingface.co/google/hear-pytorch) before logging in.

On Debian/Ubuntu, bootstrap installs `ffmpeg` and `libsndfile1` when necessary.
It also installs `uv`, creates `.venv`, chooses a compatible PyTorch backend,
installs the project, and runs an environment check. No `.env` file or copied
Hugging Face token is required.

## Inspect before running

See the detected hardware and every resolved default without starting workers:

```bash
.venv/bin/hear-distill doctor
.venv/bin/hear-distill defaults
.venv/bin/hear-distill run --dry-run
```

`doctor` reports dependency, authentication, disk, and GPU status without
printing credentials. `defaults` emits the complete runtime plan as JSON.
`--dry-run` prints the final orchestrator command.

## Automatic resource sizing

By default, the launcher uses at most **50% of the filesystem's total
capacity**, further limited by currently free space and a safety floor. The
budget includes the train lake, incoming chunks, validation and decay stores,
checkpoints, and the Hugging Face cache.

| Resource | Default policy |
| --- | --- |
| Disk budget | `min(50% of capacity, free space - safety floor)` |
| Free-space floor | 5% of capacity, clamped to 5–50 GiB |
| Train lake | 70% of the resolved disk budget |
| Stream workers | Derived from CPU count, from 1 to 8 |
| DataLoader workers | Remaining CPU capacity, capped at 12 |
| Precision | BF16 on supported CUDA GPUs, otherwise FP16; FP32 on CPU |
| Compilation | Static teacher, preprocessing, and student graphs |
| Streaming | Scales workers according to lake production versus consumption |
| Checkpoints | Every 1,000 steps, retaining the latest 20 numeric checkpoints |
| LR schedule | `3e-4`, linear warmup, then cosine decay |

The initial training batch is selected from visible GPU memory:

| GPU memory | Batch size |
| ---: | ---: |
| 75 GiB or more | 128 |
| 39–74 GiB | 96 |
| 23–38 GiB | 64 |
| 15–22 GiB | 32 |
| Less than 15 GiB | 16 |
| CPU | 8 |

GPUs with at least 39 GiB also begin with two student microbatches per teacher
forward. The trainer automatically reduces that factor after an OOM.

## How the pipeline works

```text
LAION-Audio stream
  -> partitioned download workers
  -> ffmpeg extraction to 2-second PCM16 WAV clips
  -> atomic incoming tar shards
  -> train / validation / decay curation
  -> HeAR teacher targets
  -> Canon ViT-S student updates and checkpoints
```

The lake separates four kinds of state:

```text
data/laion_audio_lake/
  incoming/     completed worker shards awaiting curation
  train/        bounded rolling training lake
  val/          stable hash-sampled validation reservoir
  decay/        bounded sample for the final decay phase
  manifests/    atomic curation and active-set metadata
  cache/        Hugging Face cache governed by the same disk budget
```

Only completed `.tar` files are exposed to curation and training. Optional
workers finish chunks before retiring, and consumed train shards are pruned
without crossing the configured reserve floor.

## Common overrides

Unspecified values remain automatic:

```bash
# Use 35% rather than 50% of the filesystem.
./run.sh --disk-fraction 0.35

# Change the run length and output locations.
./run.sh \
  --max-steps 300000 \
  --data-dir /mnt/local/hear-data \
  --train-out /mnt/local/checkpoints

# Override selected orchestrator decisions.
./run.sh --num-streams 4 --train-batch-size 96
```

Unknown `hear-distill run` arguments are passed to `datalake/run_lake.py`.
Consult [`datalake/README.md`](datalake/README.md) for direct orchestration
controls.

### Advanced trainer configuration

`--train-extra-args` replaces the launcher's default trainer argument block; it
does not append to it. Include the desired model, schedule, repeat, and runtime
flags explicitly:

```bash
TRAIN_ARGS="--device cuda --amp --repeat --shuffle-shards \
--canon --canon-2d --canon-abcd --canon-no-pos-enc \
--max-steps 200000 --lr 3e-4 --lr-schedule cosine \
--lr-warmup-steps 500 --gns-every 0"

./run.sh --train-extra-args "$TRAIN_ARGS"
```

The full trainer interface is available with:

```bash
.venv/bin/python distill_hear_vit_s_canon2d.py --help
```

## Adaptive learning-rate and batch warmup

The optional adaptive warmup measures two quantities during early training:

1. A forward-only directional search estimates the critical learning rate
   along the upcoming optimizer update. It usually needs six held-out student
   forwards and is capped at nine by default. Frozen teacher targets and audio
   preprocessing are computed only once per probe batch.
2. Two independent minibatch gradients estimate gradient noise and the
   critical batch size.

Critical sharpness is reported using
`critical_sharpness = 2 / critical_learning_rate`. The controller applies a
default 0.8 safety factor, smooths noisy measurements, and ramps toward the
selected LR and batch size.

Enable it by replacing linear warmup with `--auto-warmup`:

```bash
TRAIN_ARGS="--device cuda --amp --repeat --shuffle-shards \
--canon --canon-2d --canon-abcd --canon-no-pos-enc \
--max-steps 200000 --lr 3e-4 --lr-schedule cosine \
--auto-warmup --auto-warmup-steps 1000 --gns-every 0"

./run.sh --train-extra-args "$TRAIN_ARGS"
```

The algorithm and optimizer-direction implementation live only in
`adaptive-warmup`, pinned from GitHub in `pyproject.toml`. To test a standalone
branch or commit without changing the pin:

```bash
ADAPTIVE_WARMUP_SOURCE="git+https://github.com/Matthew-agi/adaptive-warmup.git@main" \
  ./scripts/bootstrap.sh
```

When `../adaptive-warmup` exists, bootstrap uses that sibling checkout as an
editable development override.

## Resume behavior

Streamer offsets, curation state, and manifests are persisted atomically and
reused when the same data directory is started again. Training checkpoint
resume is explicit. Add one of these flags to a complete advanced trainer
configuration:

```text
--resume-latest
--resume-from /absolute/path/to/ckpt_12000.pt
```

Optimizer state is required by default so a resumed run preserves AdamW
moments and the adaptive-warmup update direction. The orchestrator rejects
step regressions and incompatible resume state rather than silently restarting
from an earlier point.

## Monitoring and outputs

The launcher continuously reports stream production, training consumption,
reserve size, worker count, pruning, training loss, LR, batch size, and
throughput. Checkpoints default to:

```text
checkpoints/hear_vit_s_lake/
  ckpt_1000.pt
  ckpt_2000.pt
  ...
  ckpt_final.pt
  decay_phase/
```

For detailed stage timings, run the trainer with
`--optimizer-mode diagnostic`. Optional Weights & Biases logging is available
through `--wandb`, `--wandb-project`, `--wandb-entity`, and
`--wandb-run-name` in the advanced trainer arguments.

## Evaluation and benchmarking

```bash
# Evaluate a distilled checkpoint.
uv run python evaluate_distilled_hear.py \
  --embedding-model distilled \
  --ckpt checkpoints/hear_vit_s_lake/ckpt_final.pt \
  --device cuda

# Compare student and HeAR teacher throughput.
uv run python benchmark_student_vs_hear.py \
  --ckpt checkpoints/hear_vit_s_lake/ckpt_final.pt \
  --device cuda

# Microbenchmark decode and preprocessing.
uv run python benchmarks/benchmark_audio_pipeline.py --batch-size 16
```

## Repository map

| Path | Purpose |
| --- | --- |
| `run.sh` | Bootstrap-if-needed and launch the automatic pipeline |
| `scripts/bootstrap.sh` | Host dependencies, Python environment, and runtime check |
| `src/hear_distill/autotune.py` | Pure hardware-to-runtime planning policy |
| `src/hear_distill/cli.py` | `run`, `defaults`, and `doctor` commands |
| `src/hear_distill/audio.py` | Fast audio decode and HeAR preprocessing |
| `datalake/run_lake.py` | Streaming, curation, training, pruning, and resume supervisor |
| `stream_laion_audio_clips.py` | Partitioned LAION-Audio extraction and shard writing |
| `distill_hear_vit_s_canon2d.py` | Teacher/student training and checkpointing |
| `evaluate_distilled_hear.py` | Downstream embedding evaluation |
| `benchmark_student_vs_hear.py` | Student-versus-teacher throughput comparison |
| `tests/` | Unit and integration coverage |

See [`docs/architecture.md`](docs/architecture.md) and
[`docs/performance.md`](docs/performance.md) for implementation boundaries and
performance rationale.

## Development

```bash
./scripts/bootstrap.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests benchmarks
```

The test suite includes host-planning policy, fast audio paths, critical-LR
integration, streamer resume, lake control, and an end-to-end fake
streamer/trainer orchestration test.

## Troubleshooting

**HeAR download is unauthorized**

Accept the model terms in the browser, then rerun `.venv/bin/hf auth login`.
Use `.venv/bin/hear-distill doctor` to confirm the credential is visible.

**CUDA is not detected**

Run `nvidia-smi`, then inspect `.venv/bin/hear-distill defaults`. Bootstrap
selects PyTorch from the driver visible at installation time; rerun bootstrap
after fixing the host driver or container GPU passthrough.

**CUDA runs out of memory**

Lower `--train-batch-size`, set a smaller teacher batch factor, or cap adaptive
warmup with `--auto-warmup-max-batch-size`. Teacher superbatching and adaptive
batch growth both back off after OOM signals.

**Training waits for data**

Inspect the production/consumption rates in the lake status line. Increase
`--num-streams` only when streaming throughput is below training consumption;
otherwise increase DataLoader workers only when diagnostic loader-wait time is
material.

**Disk use approaches the host limit**

Lower `--disk-fraction`. The controller pauses new chunks at the lake cap and
free-space floor, but unrelated processes can still consume space outside its
budget.

## License and third-party code

This repository is released under the [MIT License](LICENSE). The bundled HeAR
audio preprocessing adaptation retains its upstream Apache 2.0 notice; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
