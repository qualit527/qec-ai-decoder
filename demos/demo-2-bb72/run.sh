#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-./.venv/bin/python}
MODE=${MODE:-prod}

if [[ "$MODE" == "fast" ]]; then
  ROUNDS=3
  PROFILE=dev
else
  ROUNDS=${ROUNDS:-10}
  PROFILE=${PROFILE:-prod}
fi

"$PYTHON_BIN" -m cli.autoqec run \
  autoqec/envs/builtin/bb72_depol.yaml \
  --rounds "$ROUNDS" \
  --profile "$PROFILE" \
  --no-llm

RUN_DIR=$(ls -t runs | head -1)
echo "Candidate Pareto:"
cat "runs/$RUN_DIR/candidate_pareto.json" 2>/dev/null || echo "(no candidate pareto yet)"
