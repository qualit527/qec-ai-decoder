# Demo 1 - AutoQEC on `surface_d5_depol`

This demo is the advisor-facing `surface_d5` walkthrough for the current
AutoQEC harness. It covers two paths:

- Path A - live LLM CLI: the real `ideator -> coder -> runner -> analyst`
  loop, with the subagents dispatched by an online model backend.
- Path B - no-LLM baseline: a pinned bundled template for smoke-testing the
  runner without any model calls.

## Path A - live LLM CLI

Run the committed live-demo wrapper:

```bash
bash demos/demo-1-surface-d5/run_live.sh
```

By default the script sets:

```bash
AUTOQEC_IDEATOR_BACKEND=github-models
AUTOQEC_IDEATOR_MODEL=openai/gpt-4.1-mini
AUTOQEC_CODER_BACKEND=github-models
AUTOQEC_CODER_MODEL=openai/gpt-4.1-mini
AUTOQEC_ANALYST_BACKEND=github-models
AUTOQEC_ANALYST_MODEL=openai/gpt-4.1-mini
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml --rounds 3 --profile dev
```

Why `github-models` here: this host already has `gh` auth, and this backend
is stable in non-interactive CLI mode. The wrapper prints the authoritative
`run_dir`, the backend/model matrix, `torch.version.cuda`,
`torch.cuda.get_device_name(0)`, and one line per round from `history.jsonl`.

Prerequisites:

- `gh auth login` must already be configured for GitHub Models requests.
- The active Python env must have CUDA-enabled torch if you want the GPU
  numbers below to reproduce. The validated run used `torch 2.6.0+cu124`.

## Path B - no-LLM baseline

Run the pinned smoke path:

```bash
TEMPLATE_NAME=gnn_small bash demos/demo-1-surface-d5/run_quick.sh
```

Under the hood this is:

```bash
python -m cli.autoqec run \
    autoqec/envs/builtin/surface_d5_depol.yaml \
    --rounds 3 \
    --profile dev \
    --no-llm \
    --template-name gnn_small
```

`run_quick.sh` now parses the authoritative `AUTOQEC_RESULT_JSON` payload
instead of guessing the newest `runs/` directory. The no-LLM path writes
`candidate_pareto.json`; that is an unverified candidate path until it is
followed by verification.

## Expected Outputs

Live runs write:

```text
runs/<run_id>/
|- history.jsonl
|- log.md
|- pareto.json
`- round_<N>/
   |- config.yaml
   |- train.log
   |- checkpoint.pt
   `- metrics.json
```

No-LLM runs write the same per-round artifacts, plus `candidate_pareto.json`
and a convenience `history.json`.

The committed live snapshot is under
`expected_output/live_llm_gpu_round3/`:

- `expected_output/live_llm_gpu_round3/history.jsonl`
- `expected_output/live_llm_gpu_round3/log.md`
- `expected_output/live_llm_gpu_round3/pareto.json`
- `expected_output/live_llm_gpu_round3/runtime_env.json`
- `expected_output/live_llm_gpu_round3/cpu_gpu_same_config_comparison.json`
- `expected_output/live_llm_gpu_round3/round_{1,2,3}_config.yaml`
- `expected_output/live_llm_gpu_round3/round_{1,2,3}_metrics.json`

## Validated Live GPU Run

Committed snapshot:

- run dir: `runs/20260424-152254`
- backend/model: `github-models` + `openai/gpt-4.1-mini` for ideator, coder,
  and analyst
- torch: `2.6.0+cu124`
- `torch.version.cuda`: `12.4`
- `torch.cuda.get_device_name(0)`: `NVIDIA GeForce RTX 4090`

Round-level summary from `expected_output/live_llm_gpu_round3/history.jsonl`:

| Round | Status | Verdict | delta_ler | train_wallclock_s | vram_peak_gb | FLOPs/syndrome | n_params |
|---|---|---|---:|---:|---:|---:|---:|
| 1 | ok | ignore | 0.0 | 2.3007 | 9.8484 | 962,820,672 | 294,721 |
| 2 | ok | ignore | 0.0 | 1.4758 | 9.8484 | 724,148,544 | 253,377 |
| 3 | ok | candidate | 0.0 | 1.2614 | 3.7483 | 724,148,544 | 253,377 |

Interpretation:

- The live LLM branch really dispatched all three agent roles and completed
  three training/eval rounds on GPU in dev profile.
- All three rounds matched the dev-profile baseline LER, so
  `expected_output/live_llm_gpu_round3/pareto.json` is empty.
- Dev-profile numbers are intentionally noisy; treat this as a contract/demo
  run, not a publishable performance claim.

## CPU vs GPU

The file `expected_output/live_llm_gpu_round3/cpu_gpu_same_config_comparison.json`
re-runs the exact `round_3_config.yaml` once on GPU and once on CPU:

| Device | train_wallclock_s | eval_wallclock_s | vram_peak_gb |
|---|---:|---:|---:|
| GPU | 1.1619 | 0.2012 | 3.7483 |
| CPU | 29.4104 | 0.6929 | 0.0 |

That is a `25.31x` GPU speedup on training wall-clock for the same config.

## Caveats

- `history.jsonl` is the authoritative round ledger. The analyst-written
  one-line summaries prove the LLM branch ran, but the scalar metrics in
  `metrics.json` remain the source of truth.
- `candidate` in this demo is not a verified scientific claim. Proper
  holdout verification still belongs to the verifier workflow.
- `dev` profile uses very small shot counts, so `delta_ler` is mostly a
  structural smoke signal here rather than a meaningful decoder ranking.
