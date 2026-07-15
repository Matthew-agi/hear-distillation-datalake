# Data lake orchestrator

`run_lake.py` coordinates staged download, preprocessing, curation, training, resume, and
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
  --train-extra-args "--device cuda --amp --gns-every 0"
```

The lake has five areas:

- `staging/`: bounded raw-MP3 shards downloaded to local NVMe;
- `incoming/`: completed streamer shards awaiting curation;
- `train/`: bounded rolling training lake;
- `val/`: stable hash-sampled validation reservoir;
- `decay/`: bounded sample retained for the final decay phase.

Each stream downloader seals a small raw shard and immediately makes it
available to concurrent ffmpeg preprocessing. The first PCM training shard is
also smaller than steady state, allowing curation and training to start while
later source shards are still downloading and preprocessing.

Only completed `.tar` files are visible to the next stage. Stream state,
curation manifests, and checkpoints are written atomically. Train shards are
claimed and deleted after one read; an empty live queue waits instead of
replaying old samples. Free-space and lake caps pause new stream chunks;
hysteresis prevents rapid worker churn.
