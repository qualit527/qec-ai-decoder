#!/usr/bin/env bash
# Demo 4 narrated walkthrough. Produces structured ASCII visualizations
# plus an optional PNG scoreboard.
#
# Same exit semantics as run.sh: 0 when the cheat is rejected (verdict in
# {FAILED, SUSPICIOUS}); non-zero on a VERIFIED verdict (verifier is broken).
#
# Env overrides:
#   PYTHON=/path/to/python   override interpreter (else picks project .venv)
#   AUTOQEC_ENV_YAML=...     override env YAML (defaults to surface_d5_depol)
#   RUN_DIR=...              override run directory (defaults to runs/demo-4/round_0)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/demos/_lib/python_bin.sh"
PY="${PYTHON:-$(discover_demo_python "$REPO_ROOT")}"

ENV_YAML="${AUTOQEC_ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}"
RUN_DIR="${RUN_DIR:-runs/demo-4/round_0}"

"$PY" "$HERE/present.py" \
    --env-yaml "$ENV_YAML" \
    --run-dir "$RUN_DIR" \
    "$@"
rc=$?

if [[ $rc -eq 0 ]]; then
    echo ""
    echo "==> rendering HTML visualization"
    "$PY" "$HERE/present_html.py" --run-dir "$RUN_DIR" || true
fi

exit $rc
