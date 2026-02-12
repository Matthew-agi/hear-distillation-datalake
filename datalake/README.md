# Data Lake Runner

`run_lake.py` runs streaming and training in parallel with a bounded reserve buffer.
It is intended to be launched from the `distillation/` directory.

## What it does

- Starts training (`distill_hear_vit_s_canon2d.py`) with live shard refresh.
- Runs per-worker streaming (`stream_laion_audio_clips.py`) and adapts active workers to training throughput.
- Scales worker count up when consumption is faster than production and reserve is low.
- Scales worker count down (or fully pauses) when reserve is high.
- Keeps disk pressure under control using a minimum free-space guard.
- On cold start (no shards yet), bootstraps streaming until first shard appears, then starts training.

Reserve is estimated as:

`reserve ~= written_clips - (train_step * batch_size * grad_accum)`

Write/consume rates are estimated with EMA:

- `train_rate ~= d(consumed_clips)/dt`
- `write_rate ~= d(written_clips)/dt`

## Quick start

```bash
python3 datalake/run_lake.py \
  --data-dir data/laion_audio_lake \
  --num-streams 6 \
  --min-streams 1 \
  --auto-tune-streams \
  --stream-headroom 1.10 \
  --chunk-clips-per-stream 20000 \
  --reserve-low-clips 150000 \
  --reserve-high-clips 450000 \
  --disk-min-free-gb 20 \
  --train-out checkpoints/hear_vit_s_lake \
  --train-batch-size 64 \
  --train-grad-accum 1 \
  --train-num-workers 8 \
  --train-extra-args "--device cuda --max-steps 200000 --canon --canon-2d --canon-abcd --shuffle-shards"
```

## Notes

- Streaming and training are concurrent by design.
- The trainer uses `--live-shard-refresh`, so new shards are ingested without restarting training.
- Stream chunks write only completed tar files to visible `shard-*.tar` names.
- If disk free space drops below `--disk-min-free-gb`, streaming pauses until space is available.
- `--num-streams` is the max worker cap; active workers are adjusted automatically when `--auto-tune-streams` is enabled.
- `.env` is loaded by default (`--env-file`), so `HF_TOKEN` and related values can be set once.
