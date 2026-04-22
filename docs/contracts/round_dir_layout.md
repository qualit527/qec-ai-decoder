# Round-dir + run-dir layout

**Frozen on:** 2026-04-21 (Day-2 handshake, Chen Jiahan)
**Change policy:** edits require PR touching this file with Chen / Lin / Xie
sign-off. Do not rename fields here without updating
`docs/contracts/interfaces.md` §2.2 and the Runner + orchestration tests.

## Directory shape

```
runs/<run_id>/                 ← the RUN (one /autoqec-run invocation)
├── history.jsonl              ← orchestration: one line per round
├── log.md                     ← orchestration: narrative, human-readable
├── pareto.json                ← orchestration: current Pareto front (≤ 5)
└── round_<N>/                 ← one ROUND (one Runner invocation)
    ├── config.yaml            ← Runner: dump of predecoder_config dict
    ├── train.log              ← Runner: one `<step>\t<loss>` per line
    ├── checkpoint.pt          ← Runner: state_dict + dsl_config + output_mode
    ├── metrics.json           ← Runner: RoundMetrics dump (§2.2)
    └── verification_report.md ← Verifier (Xie, Day-3): VerifyReport prose
```

`<run_id>` is UTC `YYYYMMDD-HHMMSS` (produced by `cli/autoqec.py::run`).

## Writer ownership (no file is written by two sides)

| Path | Writer | When |
|---|---|---|
| `runs/<run_id>/` | `cli/autoqec.py::run` | on run start |
| `history.jsonl` | `RunMemory.append_round` | after each round's Runner + Analyst |
| `log.md` | `RunMemory.append_log` | after each round's Analyst |
| `pareto.json` | `RunMemory.update_pareto` | after the Pareto refresh at round end |
| `round_<N>/` | `run_round` | at round start |
| `config.yaml` | `run_round` | before training |
| `train.log` | `run_round` | during training (overwritten once at end) |
| `checkpoint.pt` | `run_round` | after training |
| `metrics.json` | `run_round` | at round end |
| `verification_report.md` | Verifier (Xie) | when the Analyst verdict is `candidate` |

The orchestration side writes **only** at the run root; the Runner writes
**only** inside `round_<N>/`. The two sides never race on the same file.

## Reader contract

| Reader | Reads | Purpose |
|---|---|---|
| Analyst subagent | `round_<N>/metrics.json` | Round summary + verdict |
| Ideator subagent | `history.jsonl`, `pareto.json` (via L3) | Avoid re-proposal, target Pareto gaps |
| `machine_state` tool | `history.jsonl` | Round timings + killed counts |
| Verifier (Day-3) | `round_<N>/checkpoint.pt`, `round_<N>/config.yaml` | Independent holdout eval |
| `/review-log` skill | `log.md`, `history.jsonl` | Retrospective |

## Required fields per file

### `history.jsonl` — one line per round, superset of `RoundMetrics`

The orchestrator is free to add keys on top of the `RoundMetrics` dump.
The names it does add are frozen here:

- `round` — int, 1-indexed
- `hypothesis` — str, the Ideator's one-sentence proposal
- `verdict` — `"candidate"` or `"ignore"` from the Analyst
- plus every field in `RoundMetrics` (`status`, `delta_ler`,
  `ler_plain_classical`, `ler_predecoder`, `flops_per_syndrome`,
  `n_params`, `train_wallclock_s`, `eval_wallclock_s`, `vram_peak_gb`,
  `checkpoint_path`, `training_log_path`, `status_reason` when not `ok`)

### `metrics.json` — exactly `RoundMetrics` (§2.2)

Absolute paths for `checkpoint_path` and `training_log_path`. No extra keys.

### `pareto.json` — list of dicts, sorted by `-delta_ler`

Each entry at minimum: `{"round": int, "delta_ler": float,
"flops_per_syndrome": int, "n_params": int, "checkpoint_path": str}`.
Capped to 5 entries (longer fronts stored in `history.jsonl`).

### `config.yaml` — literal dump of `RunnerConfig.predecoder_config`

The same DSL dict that the Coder subagent produced, validated against
`PredecoderDSL` before training starts.

### `checkpoint.pt` — `torch.save` of

```python
{
  "class_name": type(model).__name__,
  "state_dict": model.state_dict(),
  "output_mode": model.output_mode,   # "hard_flip" | "soft_priors"
  "dsl_config": predecoder_config,    # same dict as config.yaml
}
```

### `train.log` — `<step_idx>\t<loss>` per line

Tab-separated. One line per batch. Used by `/diagnose-failure` and by
the `machine_state` `params_vs_time` scatter.

## Invariants (enforced by tests)

- `round_<N>/` exists **before** any `history.jsonl` entry with
  `round == N` is written. (Covered by integration test in
  `test_runner_smoke.py`; expand in Day-3.)
- `metrics.json` `checkpoint_path` points inside `round_<N>/`.
  (Covered in `test_orchestration_stub.py::test_l3_for_analyst_metrics_path_is_absolute`
  for the path-resolution half; Runner-side covered by `test_runner_smoke`.)
- Text files are UTF-8. Windows CI must not fall back to the locale
  code page. (Covered in
  `test_orchestration_stub.py::test_run_memory_append_log_roundtrips_utf8`.)
- Subagent response JSON validates against
  `IdeatorResponse`/`CoderResponse`/`AnalystResponse` before its
  contents are mirrored into `history.jsonl`. (Covered in
  `test_orchestration_stub.py::test_parse_response_enforces_*`.)

## Non-goals

- No per-run database. `runs/` is append-only disk; post-processing can
  build whatever index it wants on top of `history.jsonl`.
- No automatic cleanup. Old `runs/` directories persist until a human
  removes them; `.gitignore` already excludes the directory so this is
  safe.
