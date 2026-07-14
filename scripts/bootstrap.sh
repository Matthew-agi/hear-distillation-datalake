#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ]; then
      apt-get update
      apt-get install -y ffmpeg libsndfile1
    elif command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y ffmpeg libsndfile1
    else
      echo "ffmpeg is missing and sudo is unavailable." >&2
      exit 1
    fi
  else
    echo "ffmpeg is missing. Install it with the host package manager and rerun." >&2
    exit 1
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11
fi

uv pip install --python .venv/bin/python --torch-backend=auto -e ".[dev]"

ADAPTIVE_WARMUP_LOCAL="$(cd "$ROOT/.." && pwd)/adaptive-warmup"
if [ -n "${ADAPTIVE_WARMUP_SOURCE:-}" ]; then
  echo "Overriding adaptive-warmup with: $ADAPTIVE_WARMUP_SOURCE"
  uv pip install --python .venv/bin/python "$ADAPTIVE_WARMUP_SOURCE"
elif [ -f "$ADAPTIVE_WARMUP_LOCAL/pyproject.toml" ]; then
  echo "Using local adaptive-warmup checkout: $ADAPTIVE_WARMUP_LOCAL"
  uv pip install --python .venv/bin/python -e "$ADAPTIVE_WARMUP_LOCAL"
fi

echo
echo "Environment ready. Runtime check:"
.venv/bin/hear-distill doctor
echo
echo "Start with: ./run.sh"
