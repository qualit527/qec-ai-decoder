#!/usr/bin/env bash
# Demo 5: show /diagnose-failure identifies root cause of a broken config.
# Covers: compile_error, NaN loss, OOM.
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-./.venv/bin/python}
DEMO_DIR="runs/demo-5"
rm -rf "$DEMO_DIR"

# ── Failure mode 1: compile_error (bad config) ──────────────────────
mkdir -p "$DEMO_DIR/round_0"
cp demos/demo-5-failure-recovery/bad_config.yaml "$DEMO_DIR/round_0/config.yaml"
cat > "$DEMO_DIR/round_0/metrics.json" <<'EOF'
{"status": "compile_error", "status_reason": "hidden_dim validation failed: -1 < 4", "train_wallclock_s": 0.5, "eval_wallclock_s": 0, "vram_peak_gb": 0, "n_params": 0, "delta_ler": null}
EOF
echo "ValueError: hidden_dim must be >= 4, got -1" > "$DEMO_DIR/round_0/train.log"

echo "=== Failure mode 1: compile_error ==="
"$PYTHON_BIN" -m cli.autoqec diagnose "$DEMO_DIR/round_0"
echo ""

# ── Failure mode 2: NaN loss (killed_by_safety) ─────────────────────
mkdir -p "$DEMO_DIR/round_1"
cat > "$DEMO_DIR/round_1/config.yaml" <<'EOF'
type: gnn
output_mode: soft_priors
gnn:
  layers: 2
  hidden_dim: 32
  message_fn: mlp
  aggregation: sum
  normalization: none
  residual: false
  edge_features: []
head: linear
training:
  learning_rate: 10.0
  batch_size: 64
  epochs: 2
  loss: bce
  profile: dev
EOF
cat > "$DEMO_DIR/round_1/metrics.json" <<'EOF'
{"status": "killed_by_safety", "status_reason": "NaN rate 0.500", "train_wallclock_s": 1.2, "eval_wallclock_s": 0, "vram_peak_gb": 0.8, "n_params": 2048, "delta_ler": null}
EOF
printf "0\tnan\n1\tnan\n2\tnan\n" > "$DEMO_DIR/round_1/train.log"

echo "=== Failure mode 2: NaN loss ==="
"$PYTHON_BIN" -m cli.autoqec diagnose "$DEMO_DIR/round_1"
echo ""

# ── Failure mode 3: OOM ─────────────────────────────────────────────
mkdir -p "$DEMO_DIR/round_2"
cp "$DEMO_DIR/round_1/config.yaml" "$DEMO_DIR/round_2/config.yaml"
cat > "$DEMO_DIR/round_2/metrics.json" <<'EOF'
{"status": "train_error", "status_reason": "RuntimeError: CUDA out of memory while allocating tensor", "train_wallclock_s": 3.0, "eval_wallclock_s": 0, "vram_peak_gb": 23.8, "n_params": 500000, "delta_ler": null}
EOF
echo "RuntimeError: CUDA out of memory while allocating tensor" > "$DEMO_DIR/round_2/train.log"

echo "=== Failure mode 3: OOM ==="
"$PYTHON_BIN" -m cli.autoqec diagnose "$DEMO_DIR/round_2"
echo ""

# ── Diagnose from run_dir (picks latest round automatically) ────────
echo "=== Diagnose from run_dir (auto-selects round_2) ==="
"$PYTHON_BIN" -m cli.autoqec diagnose "$DEMO_DIR"
echo ""

# ── Generate diagnosis.md for each failed round ─────────────────────
"$PYTHON_BIN" demos/demo-5-failure-recovery/generate_diagnosis.py "$DEMO_DIR"

echo ""
echo "Prepared synthetic failures for /diagnose-failure:"
echo "  /diagnose-failure runs/demo-5"
echo "  /diagnose-failure runs/demo-5/round_0"
echo "Demo 5 complete. Review diagnosis.md files in $DEMO_DIR/round_*/"
