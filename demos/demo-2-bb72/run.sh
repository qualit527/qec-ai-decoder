#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"

if ! command -v "$BOOTSTRAP_PYTHON" >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON=python
fi

exec "$BOOTSTRAP_PYTHON" "$REPO_ROOT/scripts/run_bb72_demo.py" "$@"
