# Architecture

The repository now has a small stable control surface around the original
research entrypoints.

```text
run.sh
  -> scripts/bootstrap.sh        environment and accelerator setup
  -> hear-distill run            host inspection and objective/model plan
     -> datalake/run_lake.py      process supervision and shared curation
        -> stream_laion_audio_clips.py
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

Each streaming worker owns a deterministic Hugging Face dataset partition and
writes tar shards through a temporary filename. Curation routes each completed
clip to train, validation, and decay stores. The trainer uses an iterable tar
dataset and refreshes its shard inventory while the run is active.

The reserve controller has two distinct measurements:

- physical lake bytes, used for disk caps and pruning;
- logical reserve since this invocation began, computed as initial reserve plus
  new production minus new training consumption and in-run pruning.

Separating them prevents a resumed step count from being subtracted from a lake
whose already-consumed shards were previously pruned.

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
ceiling. The loader batch remains fixed while those measurements are collected;
the default 2x WSD batch multiple is applied once at the transition to stable
training. OOM feedback is the only warmup-time batch change, moves the ceiling
downward, and is checkpointed.

## Compatibility boundary

The root distillation, evaluation, benchmark, and streaming scripts remain
valid entrypoints for historical commands and checkpoints. Shared data/model
code and all new objectives live under `src/hear_distill/`; executable wrappers
live under `scripts/`. This allows the remaining research entrypoints to become
thinner without changing the one-command user surface.
