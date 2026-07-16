# Performance notes

## High-impact defaults

The throughput launcher enables mixed precision on CUDA and disables periodic
gradient-noise probes unless explicitly requested. A GNS estimate performs
additional teacher/student forward and backward work, so it belongs in
diagnostic or adaptive-LR runs rather than the default throughput path.

Fixed batch sizes use static-shape `torch.compile`. GPUs with at least 40 GiB
start with a two-microbatch teacher superbatch; the trainer halves that factor
on an out-of-memory signal. CUDA event timing is reserved for explicit
diagnostic mode so teacher superbatching does not synchronize the GPU every
step, and finite-loss checks use the asynchronous CUDA assertion path.

## Audio path

Streaming used to ask ffmpeg for float32 PCM and then convert every extracted
clip to PCM16 for WAV storage. It now asks ffmpeg for PCM16 directly and wraps
byte slices in WAV headers. Training recognizes this canonical WAV layout and
decodes it without libsndfile.

Each dataset partition first seals small raw-MP3 tar shards on local NVMe. A
foreground consumer claims the first shard immediately and feeds a bounded,
ordered ffmpeg queue while the background downloader fills later shards. The
first PCM shard is smaller than steady state so curation and training can begin
quickly. Raw and decoded claims are at-most-once; stale in-flight work is
discarded rather than replayed.

HeAR mel-PCEN preprocessing is now self-contained. The Hann window and mel
projection are buffers rather than per-batch allocations. The PCEN smoother is
the closed-form vectorization of the original exponential recurrence, removing
the Python time-step loop and diagonal matrix multiplications.

## Measuring on the target host

Synthetic microbenchmarks catch regressions, but end-to-end numbers depend on
the GPU, storage, network path, and current LAION/Hugging Face service behavior.
Use the built-in diagnostic mode for a representative run:

```bash
./run.sh \
  --train-extra-args "--device cuda --amp --gns-every 0 --optimizer-mode diagnostic --optimizer-log-every 20"
```

The trainer reports loader wait, host-to-device transfer, preprocessing,
teacher, student, backward, and optimizer timing. Increase stream workers only
when write throughput or reserve is below consumption; increase DataLoader
workers only when loader wait is material.
