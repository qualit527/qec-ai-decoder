#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  elif [[ -x "/home/jinguxie/qec-ai-decoder/.venv/bin/python" ]]; then
    PYTHON_BIN="/home/jinguxie/qec-ai-decoder/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
ENV_YAML=${ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}
PROFILE=${PROFILE:-dev}
ROUNDS=${ROUNDS:-1}
N_SHOTS=${N_SHOTS:-256}
N_SEEDS=${N_SEEDS:-2}
EXTRACT_ROOT=${EXTRACT_ROOT:-runs/demo-6-replay}
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/autoqec-mplconfig}
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RUN_OUTPUT="$("$PYTHON_BIN" -m cli.autoqec run "$ENV_YAML" --rounds "$ROUNDS" --profile "$PROFILE" --no-llm)"
printf '%s\n' "$RUN_OUTPUT"

RESULT_LINE=$(printf '%s\n' "$RUN_OUTPUT" | grep '^AUTOQEC_RESULT_JSON=')
RUN_DIR=$(
  RESULT_LINE="$RESULT_LINE" "$PYTHON_BIN" - <<'PY'
import json
import os

line = os.environ["RESULT_LINE"]
print(json.loads(line.split("=", 1)[1])["run_dir"])
PY
)

ROUND_DIR="$RUN_DIR/round_1"
"$PYTHON_BIN" -m cli.autoqec verify "$ROUND_DIR" --env "$ENV_YAML" --n-shots "$N_SHOTS" --n-seeds "$N_SEEDS"

PACKAGE_OUTPUT="$("$PYTHON_BIN" -m cli.autoqec package-run "$RUN_DIR")"
printf '%s\n' "$PACKAGE_OUTPUT"

PACKAGE_RESULT_LINE=$(printf '%s\n' "$PACKAGE_OUTPUT" | grep '^AUTOQEC_RESULT_JSON=')
PACKAGE_PATH=$(
  RESULT_LINE="$PACKAGE_RESULT_LINE" "$PYTHON_BIN" - <<'PY'
import json
import os

line = os.environ["RESULT_LINE"]
print(json.loads(line.split("=", 1)[1])["package_path"])
PY
)

"$PYTHON_BIN" -m autoqec.tools.advisor_replay \
  --run-dir "$RUN_DIR" \
  --package-path "$PACKAGE_PATH" \
  --env "$ENV_YAML" \
  --python-bin "$PYTHON_BIN" \
  --n-shots "$N_SHOTS" \
  --n-seeds "$N_SEEDS" \
  --extract-root "$EXTRACT_ROOT"

echo ""
echo "=== Demo 6 complete ==="
echo "Run dir: $RUN_DIR"
echo "Round manifest: $ROUND_DIR/artifact_manifest.json"
echo "Package: $PACKAGE_PATH"
