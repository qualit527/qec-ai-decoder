#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_YAML=${ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}
PROFILE=${PROFILE:-dev}
ROUNDS=${ROUNDS:-1}
N_SHOTS=${N_SHOTS:-256}
N_SEEDS=${N_SEEDS:-2}
EXTRACT_ROOT=${EXTRACT_ROOT:-runs/demo-6-replay}
if [[ -z "${MPLCONFIGDIR:-}" ]]; then
  MPLCONFIGDIR="$(mktemp -d 2>/dev/null || "$PYTHON_BIN" - <<'PY'
import tempfile

print(tempfile.mkdtemp(prefix="autoqec-mplconfig-"))
PY
)"
  export MPLCONFIGDIR
fi
mkdir -p "$MPLCONFIGDIR"
PYTHONPATH=$(
  REPO_ROOT="$PWD" EXISTING_PYTHONPATH="${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import os

parts = [os.environ["REPO_ROOT"]]
existing = os.environ.get("EXISTING_PYTHONPATH")
if existing:
    parts.append(existing)
print(os.pathsep.join(parts))
PY
)
export PYTHONPATH

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
