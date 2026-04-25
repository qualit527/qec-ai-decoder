#!/usr/bin/env bash
set -u -o pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find a Python interpreter. Set PYTHON_BIN=/path/to/python." >&2
    exit 127
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-runs/demo-6-showcase}"
LOG_DIR="$OUTPUT_DIR/logs"
PHASE_JSON="$OUTPUT_DIR/phases.json"
rm -rf "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

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

start=$("$PYTHON_BIN" - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)
env PYTHON_BIN="$PYTHON_BIN" bash demos/demo-6-advisor-replay/run.sh > "$LOG_DIR/demo6.stdout" 2> "$LOG_DIR/demo6.stderr"
exit_code=$?
end=$("$PYTHON_BIN" - <<'PY'
import time
print(f"{time.time():.6f}")
PY
)
START="$start" END="$end" EXIT_CODE="$exit_code" "$PYTHON_BIN" - <<'PY' > "$PHASE_JSON"
import json
import os

print(json.dumps([{
    "name": "demo6",
    "cmd": "bash demos/demo-6-advisor-replay/run.sh",
    "exit_code": int(os.environ["EXIT_CODE"]),
    "elapsed_s": float(os.environ["END"]) - float(os.environ["START"]),
    "stdout": "runs/demo-6-showcase/logs/demo6.stdout",
    "stderr": "runs/demo-6-showcase/logs/demo6.stderr",
}], indent=2, ensure_ascii=False))
PY

"$PYTHON_BIN" demos/demo-6-advisor-replay/showcase/build_report.py --root "$PWD" --output-dir "$OUTPUT_DIR" --phase-file "$PHASE_JSON"
report_exit=$?
summary_path="$OUTPUT_DIR/summary.json"
"$PYTHON_BIN" - <<'PY' "$summary_path"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
report = summary_path.with_name("report.html").resolve()
print(summary["healthy_line"])
print(f"Report HTML: {report}")
print(f"File URL: {report.as_uri()}")
print("If artifact links do not open, serve from repo root:")
print("  python -m http.server 8768 --bind 127.0.0.1")
print("Then open:")
print("  http://127.0.0.1:8768/runs/demo-6-showcase/report.html")
PY
exit "$report_exit"
