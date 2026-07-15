# Canon Audio Encoder Trainer

Train Canon-adapted audio Vision Transformers from the same continuously
streamed LAION-Audio source in either of two explicit modes:

- **direct pretraining** learns an encoder by reconstructing masked
  mel-PCEN spectrogram patches with a lightweight, disposable decoder;
- **distillation** learns from the gated Google HeAR teacher using embedding,
  contrastive, and relational losses.

Tiny, small, base, and large ViTs use one shared model factory. Canon widths,
MLP widths, patch grids, decoder dimensions, and memory limits adapt to the
selected model instead of assuming ViT-S.

## Start on a GPU instance

```bash
git clone https://github.com/Matthew-agi/hear-distillation-datalake.git
cd hear-distillation-datalake
./scripts/bootstrap.sh

# Direct Canon-ViT pretraining; no HeAR teacher is downloaded.
./run.sh --objective reconstruct --model-size base
```

For HeAR distillation, accept the
[HeAR model terms](https://huggingface.co/google/hear-pytorch), authenticate,
and select the other objective:

```bash
.venv/bin/hf auth login
./run.sh --objective distill --model-size small
```

The backward-compatible default is `--objective distill --model-size small`.

## What is automatic

The launcher detects CPU count, GPU memory, filesystem capacity, and free
space before it starts the lake. By default it budgets the smaller of:

```text
50% of total filesystem capacity
free space minus a 5–50 GiB safety floor
```

It then sizes the rolling train lake, incoming chunks, stable validation and
decay stores, stream processes, DataLoader workers, mixed precision, and the
training-batch ceiling. There is no model-name multiplier table. The planner
constructs the selected training graph on PyTorch's meta device and measures:

- exact trainable parameter, gradient, and AdamW-state bytes;
- batch-dependent saved-tensor bytes from the difference between batch-one and
  batch-two autograd graphs;
- the selected encoder, Canon layout, decoder, patch grid, and objective.

On CUDA, direct pretraining then runs a real batch-one/batch-two allocation
probe in the selected AMP mode. It subtracts memory already occupied by other
GPU processes and keeps a configurable reserve. This live measurement can only
lower the analytical ceiling.

| ViT | Backbone | Default decoder |
| --- | --- | --- |
| tiny | `vit_tiny_patch16_224` | 192d, 2 layers |
| small | `vit_small_patch16_224` | 256d, 2 layers |
| base | `vit_base_patch16_224` | 384d, 3 layers |
| large | `vit_large_patch16_224` | 512d, 4 layers |

Adaptive warmup is enabled by default for both objectives. It starts from the
smallest normal probe batch under the measured ceiling, estimates critical
batch size and critical learning rate, and holds that measurement batch fixed
for the entire warmup. At the WSD stable-phase handoff, the selected critical
batch is multiplied by 2x by default, rounded up to the next power of two,
clamped to the largest power of two within the measured ceiling, and applied
once. `--auto-warmup-batch-multiplier` changes the direct trainer's
multiple; distillation exposes the same policy as `--batch-opt-mult`. A real
OOM is the safety exception: it lowers and persists the ceiling with headroom.

Inspect everything without starting a worker:

```bash
.venv/bin/hear-distill doctor
.venv/bin/hear-distill defaults --objective reconstruct --model-size large
.venv/bin/hear-distill run --objective reconstruct --model-size large --dry-run
```

## Direct pretraining design

The audio waveform is converted to the same `[1, 192, 128]` HeAR mel-PCEN
image used by distillation. Patch-16 models produce a `12 × 8` grid. By
default, 75% of those embeddings are replaced in place with a learned mask
token, and the complete grid passes through the Canon encoder.

Keeping all 96 tokens is intentional. Canon2D reshapes tokens back into their
time-frequency grid; dropping masked tokens as in a conventional MAE encoder
would destroy that topology. A small Transformer decoder predicts only the
masked patch targets. The decoder is a pretraining head and the encoder is the
artifact used downstream.

New decoders are trained from scratch by default. Reusing an unrelated
AudioMAE decoder is not safe: decoder weights depend on encoder width, patch
geometry, positional layout, and decoder depth. `--decoder-checkpoint` is
supported, but loading is strict and succeeds only when all of that metadata
matches. See [the pretraining design note](docs/direct-pretraining.md).

Every direct checkpoint contains separate keys:

```text
encoder       downstream Canon-ViT weights
decoder       disposable reconstruction head
model_config  exact compatibility metadata
optim/scaler  continuation state
args/step     reproducibility and progress
```

You can initialize the encoder three ways when invoking the trainer directly:

```bash
# Default: encoder and decoder both start from random initialization.
.venv/bin/hear-pretrain --data-dir data/laion_audio_lake/train --model-size base

# timm ImageNet initialization for the encoder; decoder is still new.
.venv/bin/hear-pretrain --data-dir data/laion_audio_lake/train \
  --model-size base --encoder-pretrained

# Strictly reuse an encoder or an architecture-compatible decoder.
.venv/bin/hear-pretrain --data-dir data/laion_audio_lake/train \
  --model-size base \
  --encoder-checkpoint /path/to/checkpoint.pt \
  --decoder-checkpoint /path/to/checkpoint.pt
```

## Distillation design

Distillation freezes `google/hear-pytorch` and trains the selected Canon-ViT
plus a 512-dimensional projection using:

```text
MSE(student, teacher)
+ bidirectional student/teacher InfoNCE
+ relational similarity-matrix MSE
```

CUDA runs use mixed precision, fused AdamW when available, `torch.compile`, and
optional teacher superbatching. Critical-learning-rate and critical-batch-size
warmup is enabled by default through the standalone
[`adaptive-warmup`](https://github.com/Matthew-agi/adaptive-warmup) library.
The generated defaults include the architecture-derived batch ceiling. A
manual trainer block is still supported:

```bash
TRAIN_ARGS="--device cuda --amp --repeat --shuffle-shards \
--model-size small --canon --canon-2d --canon-abcd --canon-no-pos-enc \
--max-steps 200000 --lr 3e-4 --lr-schedule cosine \
--auto-warmup --auto-warmup-steps 1000 --gns-every 0"

./run.sh --objective distill --train-extra-args "$TRAIN_ARGS"
```

The warmup policy and probes live only in `adaptive-warmup`, pinned in
`pyproject.toml`. Bootstrap uses a sibling `../adaptive-warmup` checkout as an
editable override when one is present.

## One data source, two trainers

```text
LAION-Audio streaming dataset
  -> partitioned download and ffmpeg extraction
  -> atomic 2-second PCM16 tar shards
  -> bounded train / validation / decay curation
  -> direct masked reconstruction OR HeAR distillation
  -> encoder checkpoints
```

The lake layout is shared:

```text
data/laion_audio_lake/
  incoming/     completed worker shards awaiting curation
  train/        bounded rolling training lake
  val/          stable hash-sampled validation reservoir
  decay/        bounded final-phase sample
  manifests/    atomic active-set and curation state
  cache/        Hugging Face cache inside the same disk budget
```

Only completed tar files reach the trainers. Direct reconstruction treats the
train lake as an at-most-once queue: a loader worker atomically claims a shard,
removes it from active inventory, reads it once, and deletes it. Stale in-flight
claims are discarded after a crash rather than replayed. Distillation retains
its legacy reusable-lake behavior for checkpoint compatibility.

## Common controls

```bash
# Use 35% rather than 50% of the filesystem.
./run.sh --objective reconstruct --disk-fraction 0.35

# Put the lake and checkpoints on instance storage.
./run.sh --objective reconstruct --model-size large \
  --data-dir /mnt/local/audio-lake \
  --train-out /mnt/local/checkpoints/canon-vit-large

# Override individual orchestrator decisions.
./run.sh --objective distill --num-streams 4 --train-batch-size 64
```

Unknown `hear-distill run` arguments pass through to the lake orchestrator.
When `--train-extra-args` is supplied, it replaces the default trainer block;
include the model size, Canon flags, schedule, and data-lifecycle flags you need.

Useful help surfaces:

```bash
.venv/bin/hear-distill run --help
.venv/bin/hear-pretrain --help
.venv/bin/python distill_hear_vit_s_canon2d.py --help
.venv/bin/python datalake/run_lake.py --help
```

## Resume and outputs

Streamer offsets, lake curation state, and manifests are atomic and reusable.
Training continuation is explicit:

```text
--resume-latest
--resume-from /absolute/path/to/ckpt_12000.pt
```

Both trainers restore model, optimizer, scaler, and step. Direct pretraining
also rejects a resume whose model, patch grid, or decoder metadata differs.
The orchestrator rejects step regressions rather than silently restarting.

Evaluate either a distilled or direct-pretraining encoder with the same tool:

```bash
.venv/bin/python evaluate_distilled_hear.py \
  --embedding-model distilled \
  --embedding-head student \
  --ckpt checkpoints/canon_audio_pretrain/ckpt_final.pt \
  --device cuda
```

The benchmark accepts either the historical `student` checkpoint key or the
direct trainer's `encoder` key:

```bash
.venv/bin/python benchmark_student_vs_hear.py \
  --ckpt checkpoints/canon_audio_pretrain/ckpt_final.pt \
  --device cuda
```

## Repository layout

```text
src/hear_distill/
  audio.py                 shared decode and mel-PCEN preprocessing
  autotune.py              disk, worker, precision, and batch policy
  cli.py                   one-command objective selection
  data/shards.py           reusable or at-most-once live tar-shard dataset
  models/canon.py          single Canon implementation
  models/memory.py         graph-derived memory measurement
  models/vit.py            tiny/small/base/large factory
  models/reconstruction.py masked reconstruction encoder/decoder
  training/pretrain.py     direct-pretraining loop and checkpoints

scripts/
  bootstrap.sh             GPU-instance environment setup
  train/distill.py          organized distillation entrypoint
  train/pretrain_reconstruction.py
                            direct-pretraining entrypoint

datalake/run_lake.py       streaming and bounded-lake supervisor
distill_hear_vit_s_canon2d.py
                            legacy-compatible distillation entrypoint
evaluate_distilled_hear.py downstream embedding evaluation
benchmark_student_vs_hear.py
                            encoder-versus-HeAR throughput benchmark
```

The root research entrypoints remain for checkpoint and command compatibility;
new reusable code belongs under `src/hear_distill/`. Architecture and
performance boundaries are documented in [docs/architecture.md](docs/architecture.md)
and [docs/performance.md](docs/performance.md).

## Development

```bash
./scripts/bootstrap.sh
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check src tests benchmarks
```

## Troubleshooting

**HeAR download is unauthorized** — accept the model terms and run
`.venv/bin/hf auth login`. Direct reconstruction does not download HeAR.

**CUDA is unavailable** — check `nvidia-smi`, then rerun bootstrap after fixing
driver or container passthrough. `hear-distill defaults` shows what was found.

**CUDA runs out of memory** — adaptive warmup catches a real training OOM,
backs off with headroom, and checkpoints the lower ceiling. Use
`--train-batch-size` only when you want to impose a stricter manual cap.

**Training waits for data** — compare production and consumption in the lake
status line. Increase stream workers only when production is the bottleneck.

**Disk approaches the host limit** — lower `--disk-fraction`. The controller
also stops at its free-space floor, but cannot govern unrelated processes.

## License

This repository is released under the [MIT License](LICENSE). The bundled HeAR
audio preprocessing adaptation retains its upstream Apache 2.0 notice; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
