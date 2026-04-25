#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

source "$REPO_ROOT/demos/_lib/python_bin.sh"

PYTHON_BIN="${PYTHON_BIN:-$(discover_demo_python "$REPO_ROOT")}"
ENV_YAML="${ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}"
ROUNDS="${ROUNDS:-1}"
PROFILE="${PROFILE:-prod}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m cli.autoqec run "$ENV_YAML" --rounds "$ROUNDS" --profile "$PROFILE"
