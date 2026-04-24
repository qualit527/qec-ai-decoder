#!/usr/bin/env bash
# Demo 1 live-LLM path. Runs the real CLI research loop and reports the
# authoritative run_dir, backend/model matrix, torch/CUDA runtime, and a
# compact per-round summary from history.jsonl.
set -euo pipefail

ROUNDS="${ROUNDS:-3}"
PROFILE="${PROFILE:-dev}"
ENV_YAML="${ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

AUTOQEC_IDEATOR_BACKEND="${AUTOQEC_IDEATOR_BACKEND:-github-models}"
AUTOQEC_IDEATOR_MODEL="${AUTOQEC_IDEATOR_MODEL:-openai/gpt-4.1-mini}"
AUTOQEC_CODER_BACKEND="${AUTOQEC_CODER_BACKEND:-github-models}"
AUTOQEC_CODER_MODEL="${AUTOQEC_CODER_MODEL:-openai/gpt-4.1}"
AUTOQEC_ANALYST_BACKEND="${AUTOQEC_ANALYST_BACKEND:-github-models}"
AUTOQEC_ANALYST_MODEL="${AUTOQEC_ANALYST_MODEL:-openai/gpt-4.1-mini}"

TORCH_RUNTIME="$("$PYTHON_BIN" - <<'PY'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "torch.version.cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    "device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"

printf 'Torch runtime: %s\n' "$TORCH_RUNTIME"
printf 'Backend matrix: ideator=%s/%s coder=%s/%s analyst=%s/%s\n' \
    "$AUTOQEC_IDEATOR_BACKEND" "$AUTOQEC_IDEATOR_MODEL" \
    "$AUTOQEC_CODER_BACKEND" "$AUTOQEC_CODER_MODEL" \
    "$AUTOQEC_ANALYST_BACKEND" "$AUTOQEC_ANALYST_MODEL"

RUN_OUTPUT="$(
    AUTOQEC_IDEATOR_BACKEND="$AUTOQEC_IDEATOR_BACKEND" \
    AUTOQEC_IDEATOR_MODEL="$AUTOQEC_IDEATOR_MODEL" \
    AUTOQEC_CODER_BACKEND="$AUTOQEC_CODER_BACKEND" \
    AUTOQEC_CODER_MODEL="$AUTOQEC_CODER_MODEL" \
    AUTOQEC_ANALYST_BACKEND="$AUTOQEC_ANALYST_BACKEND" \
    AUTOQEC_ANALYST_MODEL="$AUTOQEC_ANALYST_MODEL" \
    "$PYTHON_BIN" -m cli.autoqec run "$ENV_YAML" --rounds "$ROUNDS" --profile "$PROFILE"
)"
printf '%s\n' "$RUN_OUTPUT"

RESULT_JSON="$(printf '%s\n' "$RUN_OUTPUT" | sed -n 's/^AUTOQEC_RESULT_JSON=//p' | tail -1)"
RUN_DIR="$("$PYTHON_BIN" -c 'import json,sys; print(json.loads(sys.stdin.read())["run_dir"])' <<<"$RESULT_JSON")"

echo ""
echo "=== Demo 1 (live LLM) complete ==="
echo "Run dir: $RUN_DIR"
echo "History path: $RUN_DIR/history.jsonl"

"$PYTHON_BIN" - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

run_dir = pathlib.Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in run_dir.joinpath("history.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
print(f"Rounds: {len(rows)}")
for row in rows:
    round_idx = row.get("round")
    status = row.get("status")
    verdict = row.get("verdict")
    delta = row.get("delta_ler")
    train_s = row.get("train_wallclock_s")
    vram = row.get("vram_peak_gb")
    print(
        f"round_{round_idx}: status={status} verdict={verdict} "
        f"delta_ler={delta} train_wallclock_s={train_s} vram_peak_gb={vram}"
    )
PY
