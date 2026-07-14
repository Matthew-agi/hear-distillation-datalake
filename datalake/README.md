# Data lake orchestrator

`run_lake.py` coordinates streaming, curation, training, pruning, resume, and
the optional decay phase. Most users should start it through the hardware-aware
launcher:

```bash
./run.sh
```

The launcher computes the disk budget and streaming/training concurrency.
Direct invocation remains supported for experiments that need every control:

```bash
uv run python datalake/run_lake.py \
  --data-dir data/laion_audio_lake \
  --lake-max-gb 120 \
  --reserve-low-gb 8 \
  --reserve-high-gb 20 \
  --chunk-gb-per-stream 1 \
  --num-streams 4 \
  --train-extra-args "--device cuda --amp --repeat --gns-every 0"
```

The lake has four areas:

- `incoming/`: completed streamer shards awaiting curation;
- `train/`: bounded rolling training lake;
- `val/`: stable hash-sampled validation reservoir;
- `decay/`: bounded sample retained for the final decay phase.

Only completed `.tar` files are visible to curation and training. Stream state,
curation manifests, and checkpoints are written atomically. Free-space and lake
caps pause new stream chunks; hysteresis prevents rapid worker churn.
