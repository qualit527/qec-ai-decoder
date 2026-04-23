#!/usr/bin/env bash
# Demo 4: verify that a hand-crafted cheating predecoder (MemorizerPredecoder)
# fails independent verification.
set -euo pipefail
mkdir -p runs/demo-4/round_0
python -c "
from pathlib import Path
from autoqec.cheaters.memorize import save_memorizer_ckpt
from autoqec.envs.schema import load_env_yaml
from autoqec.runner.data import load_code_artifacts
env = load_env_yaml('autoqec/envs/builtin/surface_d5_depol.yaml')
artifacts = load_code_artifacts(env)
save_memorizer_ckpt(Path('runs/demo-4/round_0/checkpoint.pt'), env_spec=env, artifacts=artifacts)
"
python -m cli.autoqec verify runs/demo-4/round_0 \
  --env autoqec/envs/builtin/surface_d5_depol.yaml --n-shots 20000
cat runs/demo-4/round_0/verification_report.md
