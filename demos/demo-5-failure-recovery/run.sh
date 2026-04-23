#!/usr/bin/env bash
# Demo 5: show /diagnose-failure identifies root cause of a broken config.
set -euo pipefail
mkdir -p runs/demo-5/round_0
cp demos/demo-5-failure-recovery/bad_config.yaml runs/demo-5/round_0/config.yaml
echo "ValueError: hidden_dim must be >= 4" > runs/demo-5/round_0/train.log
cat > runs/demo-5/round_0/metrics.json <<EOF
{"status": "compile_error", "status_reason": "hidden_dim validation failed",
 "train_wallclock_s": 0.5, "eval_wallclock_s": 0, "vram_peak_gb": 0}
EOF
python -m cli.autoqec diagnose runs/demo-5/round_0
echo ""
echo "Now a human runs /diagnose-failure runs/demo-5 to get an LLM-authored fix."
