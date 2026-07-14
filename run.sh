#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/hear-distill ]; then
  "$ROOT/scripts/bootstrap.sh"
fi

exec .venv/bin/hear-distill run "$@"
