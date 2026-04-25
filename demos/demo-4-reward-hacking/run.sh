#!/usr/bin/env bash
# Demo 4 — reward-hacking detection (issue #39).
#
# Builds a MemorizerPredecoder that cheats by memorising training-seed
# syndromes, then runs `cli.autoqec verify` on holdout seeds. The verdict
# MUST be SUSPICIOUS or FAILED. Exits non-zero if ever VERIFIED.
#
# Usage:
#   bash demos/demo-4-reward-hacking/run.sh            # fast smoke (~1 min)
#   bash demos/demo-4-reward-hacking/run.sh --full     # test-plan budget (~15-20 min)
#   bash demos/demo-4-reward-hacking/run.sh --n-shots 20000 --n-seeds 10
#
# Env overrides:
#   PYTHON=/path/to/python   override interpreter (else picks project .venv)
#   AUTOQEC_ENV_YAML=...     override env YAML (defaults to surface_d5_depol)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/demos/_lib/python_bin.sh"
PY="${PYTHON:-$(discover_demo_python "$REPO_ROOT")}"
ENV_YAML="${AUTOQEC_ENV_YAML:-autoqec/envs/builtin/surface_d5_depol.yaml}"
RUN_DIR="runs/demo-4/round_0"
MODE="fast"
N_SHOTS=5000
N_SEEDS=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fast)
            MODE="fast"; N_SHOTS=5000; N_SEEDS=5; shift
            ;;
        --full)
            MODE="full"; N_SHOTS=200000; N_SEEDS=50; shift
            ;;
        --n-shots)
            N_SHOTS="$2"; shift 2
            ;;
        --n-seeds)
            N_SEEDS="$2"; shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"; shift 2
            ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

echo "==> demo-4 reward-hacking detection"
echo "    mode=$MODE  n_shots=$N_SHOTS  n_seeds=$N_SEEDS"
echo "    env=$ENV_YAML"
echo "    run_dir=$RUN_DIR"

mkdir -p "$RUN_DIR"

echo ""
echo "==> building MemorizerPredecoder (reads train-seed syndromes, memorises perfect corrections)"
"$PY" - <<PYEOF
from pathlib import Path
from autoqec.cheaters.memorize import save_memorizer_ckpt
from autoqec.envs.schema import load_env_yaml
from autoqec.runner.data import load_code_artifacts
env = load_env_yaml("$ENV_YAML")
artifacts = load_code_artifacts(env)
save_memorizer_ckpt(Path("$RUN_DIR/checkpoint.pt"), env_spec=env, artifacts=artifacts)
print("  saved checkpoint:", "$RUN_DIR/checkpoint.pt")
PYEOF

echo ""
echo "==> running independent verifier on holdout seeds (never seen during memorisation)"
"$PY" -m cli.autoqec verify "$RUN_DIR" \
    --env "$ENV_YAML" \
    --n-shots "$N_SHOTS" \
    --n-seeds "$N_SEEDS"

echo ""
echo "==> verification_report.md"
cat "$RUN_DIR/verification_report.md"

echo ""
echo "==> asserting verdict is not VERIFIED"
VERDICT="$("$PY" -c "
import json
with open('$RUN_DIR/verification_report.json') as f:
    print(json.load(f)['verdict'])
")"
echo "    verdict=$VERDICT"

case "$VERDICT" in
    FAILED|SUSPICIOUS)
        echo ""
        echo "ok: reward-hacking checkpoint correctly rejected."
        echo ""
        echo "==> rendering HTML visualization"
        "$PY" "$HERE/present_html.py" --run-dir "$RUN_DIR" || true
        exit 0
        ;;
    VERIFIED)
        echo "" >&2
        echo "FAILURE: memorizer was VERIFIED. The verifier is broken or the" >&2
        echo "seed policy lets train and holdout overlap. Investigate" >&2
        echo "autoqec/eval/independent_eval.py before shipping." >&2
        exit 1
        ;;
    *)
        echo "" >&2
        echo "FAILURE: unexpected verdict '$VERDICT'." >&2
        exit 1
        ;;
esac
