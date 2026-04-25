#!/usr/bin/env bash
# Demo 1 no-LLM smoke path. Runs three dev-profile rounds on surface_d5
# with randomly-sampled seed templates from autoqec/example_db/.
#
# For the real AutoQEC experience, use the `/autoqec-run` skill inside
# a Claude Code session. See demos/demo-1-surface-d5/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/demos/_lib/python_bin.sh"
PY="${PYTHON:-$(discover_demo_python "$REPO_ROOT")}"

ROUNDS="${ROUNDS:-3}"
PROFILE="${PROFILE:-dev}"
ENV_YAML="${ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}"

"$PY" -m cli.autoqec run "$ENV_YAML" \
    --rounds "$ROUNDS" \
    --profile "$PROFILE" \
    --no-llm

RUN_ID="$(ls -t runs | head -1)"
echo ""
echo "=== Demo 1 (no-LLM) complete ==="
echo "Run dir: runs/$RUN_ID"
echo "History: $(wc -l < runs/$RUN_ID/history.jsonl) rounds"
if [ -f "runs/$RUN_ID/candidate_pareto.json" ]; then
    echo "Candidate Pareto: $(cat runs/$RUN_ID/candidate_pareto.json)"
else
    echo "Candidate Pareto: (none yet)"
fi
