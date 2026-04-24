# Demo 2 — BB72 qLDPC AutoQEC via OSD

Shows that the same AutoQEC no-LLM harness used on the surface-code demo also
works on the `bb72_depol` qLDPC environment, but routes the classical decoder
through OSD instead of MWPM.

## What this demo does

End-to-end path:

`bb72_depol.yaml` -> no-LLM seed template -> Runner train/eval -> `classical_backend: osd` -> `history.jsonl` + `metrics.json` + `checkpoint.pt` + `train.log` + `candidate_pareto.json`

One round does four things:

1. Loads `autoqec/envs/builtin/bb72_depol.yaml`, which points at the BB72
   parity-check artifact (`circuits/bb72_Hx.npy`) and explicitly sets
   `classical_backend: osd`.
2. Picks a shipped seed `PredecoderDSL` template because this demo stays on the
   stable `--no-llm` path.
3. Trains and evaluates the neural predecoder with the Runner, writing
   `round_1/config.yaml`, `train.log`, `checkpoint.pt`, and `metrics.json`.
4. Records the successful round into `history.jsonl` / `history.json` and
   refreshes `candidate_pareto.json`.

## Why BB72 Is Different From Demo 1

`surface_d5_depol` routes through MWPM / PyMatching. BB72 is a qLDPC parity
check and therefore goes through `classical_backend: osd` in
`autoqec/envs/builtin/bb72_depol.yaml`. That means the same AutoQEC harness is
unchanged, but the classical backend and code family differ:

- Surface demo: circuit-level surface code + MWPM.
- BB72 demo: qLDPC parity-check matrix + OSD.

## How To Run

Recommended smoke path:

```bash
bash demos/demo-2-bb72/run.sh
```

Explicit modes:

```bash
MODE=fast bash demos/demo-2-bb72/run.sh
MODE=dev bash demos/demo-2-bb72/run.sh
MODE=prod bash demos/demo-2-bb72/run.sh
```

Mode defaults:

- `fast`: 1 round, `profile=dev`
- `dev`: 3 rounds, `profile=dev`
- `prod`: 10 rounds, `profile=prod`

Direct CLI (same smoke path as `MODE=fast`):

```bash
python -m cli.autoqec run autoqec/envs/builtin/bb72_depol.yaml --rounds 1 --profile dev --no-llm
```

You can still override `ROUNDS`, `PROFILE`, or `PYTHON_BIN` when invoking the
wrapper. The wrapper is worktree-safe: it walks upward until it finds a shared
`.venv/bin/python`, so it also works from `.worktrees/<branch>/`.

## Expected Outputs

After a successful no-LLM run:

```text
runs/<run_id>/
├── history.json
├── history.jsonl
├── candidate_pareto.json
└── round_1/
    ├── config.yaml
    ├── train.log
    ├── checkpoint.pt
    └── metrics.json
```

Committed reference snapshot:

- `demos/demo-2-bb72/expected_output/sample_run/history.jsonl`
- `demos/demo-2-bb72/expected_output/sample_run/candidate_pareto.json`
- `demos/demo-2-bb72/expected_output/sample_run/round_1_config.yaml`
- `demos/demo-2-bb72/expected_output/sample_run/round_1_metrics.json`

The snapshot contains at least one successful BB72 round with
`metrics.json.status == "ok"`.

## Runtime Expectations

See `demos/demo-2-bb72/runtime.md` for separate CPU, GPU, and OSD-cost notes.

## How We Check This Demo Is Correct

- Structural contract: `tests/test_demo_smoke_contracts.py` already exercises
  the no-LLM CLI path for both `surface_d5_depol.yaml` and `bb72_depol.yaml`
  and checks `history.jsonl`, `candidate_pareto.json`, and round artifacts.
- Demo wrapper behavior: `tests/test_run_bb72_demo.py` checks the new
  fast/dev/prod defaults and the worktree-safe Python resolution path.
- Demo assets: `tests/test_demo_bb72_assets.py` checks that this README keeps
  the OSD/MWPM explanation, that runtime guidance exists, and that the
  committed snapshot parses as a successful round.

## Known Limitations

- The BB72 env uses a manually constructed parity-check matrix artifact.
- The current metric is an exact-recovery surrogate, not a true logical error
  rate, because logical-operator metadata is not available in this repo yet.
- The emitted candidate Pareto is unverified demo output. The authoritative
  verified Pareto front still belongs to the separate verification flow.
