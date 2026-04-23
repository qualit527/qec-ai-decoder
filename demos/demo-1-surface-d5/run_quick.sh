#!/usr/bin/env bash
# Demo 1 no-LLM smoke path. Runs three dev-profile rounds on surface_d5
# with a pinned bundled seed template for reproducibility.
#
# For the real AutoQEC experience, use the `/autoqec-run` skill inside
# a Claude Code session. See demos/demo-1-surface-d5/README.md.
set -euo pipefail

ROUNDS="${ROUNDS:-3}"
PROFILE="${PROFILE:-dev}"
ENV_YAML="${ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEMPLATE_NAME="${TEMPLATE_NAME:-gnn_small}"

OUTPUT="$("$PYTHON_BIN" -m cli.autoqec run "$ENV_YAML" \
    --rounds "$ROUNDS" \
    --profile "$PROFILE" \
    --no-llm \
    --template-name "$TEMPLATE_NAME")"
printf '%s\n' "$OUTPUT"

RESULT_JSON="$(printf '%s\n' "$OUTPUT" | sed -n 's/^AUTOQEC_RESULT_JSON=//p' | tail -1)"
RUN_DIR="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read())["run_dir"])' <<<"$RESULT_JSON")"
echo ""
echo "=== Demo 1 (no-LLM) complete ==="
echo "Template: $TEMPLATE_NAME"
echo "Run dir: $RUN_DIR"
echo "History: $(wc -l < "$RUN_DIR/history.jsonl") rounds"
[ -f "$RUN_DIR/candidate_pareto.json" ] && echo "Candidate Pareto: $(cat "$RUN_DIR/candidate_pareto.json")"
