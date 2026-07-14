# Architecture

The repository now has a small stable control surface around the original
research entrypoints.

```text
run.sh
  -> scripts/bootstrap.sh        environment and accelerator setup
  -> hear-distill run            host inspection and runtime plan
     -> datalake/run_lake.py      process supervision and curation
        -> stream_laion_audio_clips.py
        -> distill_hear_vit_s_canon2d.py
           -> hear_distill.audio cached HeAR preprocessing
```

## Control plane

`src/hear_distill/autotune.py` is pure policy: it converts detected CPU, disk,
and GPU resources into a deterministic runtime plan. `src/hear_distill/cli.py`
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

## Compatibility boundary

The root training, evaluation, benchmark, and streaming scripts remain valid
entrypoints. New reusable functionality lives under `src/hear_distill/`; this
allows the large research scripts to be split further without changing the
one-command user surface.
