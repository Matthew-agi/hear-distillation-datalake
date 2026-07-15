# Architecture

The repository now has a small stable control surface around the original
research entrypoints.

```text
run.sh
  -> scripts/bootstrap.sh        environment and accelerator setup
  -> hear-distill run            host inspection and objective/model plan
     -> datalake/run_lake.py      process supervision and shared curation
        -> stream_laion_audio_clips.py
           -> hear_distill.data.staging raw NVMe queue
        -> distillation trainer OR reconstruction trainer
           -> hear_distill.audio cached HeAR preprocessing
           -> hear_distill.models shared Canon-ViT construction
```

## Control plane

`src/hear_distill/autotune.py` is policy: it combines detected CPU, disk, and
GPU resources with the architecture measurement in `models/memory.py` to make
a deterministic runtime plan. `src/hear_distill/cli.py`
owns the friendly commands and translates that plan into existing orchestrator
flags. This keeps machine policy separate from the long-running lake state
machine and makes the calculations directly testable.

## Data plane

Each streaming worker owns a deterministic Hugging Face dataset partition. A
background thread downloads small, atomic raw-MP3 shards to local NVMe while a
foreground thread immediately preprocesses the oldest available shard into
PCM16 clips. Curation routes each completed clip to exactly one of train,
validation, or WSD decay. Both trainers atomically move each selected train shard to an
in-flight area before reading it, then delete it; no shard can be claimed by a
second loader worker or replayed after restart.

The reserve controller has two distinct measurements:

- physical active-train bytes, used for disk caps and scheduling;
- claimed bytes since this invocation began, inferred from shards actually
  leaving the active inventory rather than from the configured batch ceiling.

Separating them prevents a resumed step count or an adaptive batch estimate
from being mistaken for actual data consumption.

## Training plane

`src/hear_distill/models/` is the model boundary. `vit.py` resolves the selected
ViT family size, `canon.py` adapts every Canon placement to actual attention and
MLP widths, and `reconstruction.py` owns the pretraining-only decoder. Both
training objectives consume the same 192-by-128 preprocessed spectrograms.

Distillation uses a pooled encoder representation and a projection to the HeAR
embedding width. Direct pretraining keeps the full patch grid, masks embeddings
in place, and predicts masked patches through the decoder. This full-grid
choice preserves the spatial contract required by Canon2D.

Both trainers default to adaptive warmup. The model graph supplies a memory
ceiling, the direct trainer calibrates it with live CUDA allocations, and the
standalone adaptive-warmup controller selects critical batch and LR within that
ceiling. Direct training starts at the largest power-of-two batch within that
ceiling, and the loader batch remains fixed while measurements are collected;
the default 2x WSD batch multiple is rounded up to a power of two and applied
once at the transition to stable training. OOM feedback is the only warmup-time
batch change, moves the ceiling downward, and is checkpointed.

## Compatibility boundary

The root distillation, evaluation, benchmark, and streaming scripts remain
valid entrypoints for historical commands and checkpoints. Shared data/model
code and all new objectives live under `src/hear_distill/`; executable wrappers
live under `scripts/`. This allows the remaining research entrypoints to become
thinner without changing the one-command user surface.
