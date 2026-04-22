# Round-dir + run-dir layout

**Frozen on:** 2026-04-21 (Day-2 handshake, Chen Jiahan)
**Change policy:** edits require PR touching this file with Chen / Lin / Xie
sign-off. Do not rename fields here without updating
`docs/contracts/interfaces.md` §2.2 and the Runner + orchestration tests.

**Reading convention**

- Statements in this doc describe **current behaviour** of the code on
  `main` + `feat/chen-orchestration` as of 2026-04-21.
- Known gaps between this doc and current code are tagged
  **[TODO-fill-in]** and name the owner who should close them.

## Directory shape

```
runs/<run_id>/                 ← the RUN (one /autoqec-run invocation)
├── history.jsonl              ← orchestration: one line per round
├── log.md                     ← orchestration: narrative, human-readable
├── pareto.json                ← orchestration: current authoritative Pareto front (≤ 5)
├── candidate_pareto.json      ← CLI demo path: unverified candidate front (optional)
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
| `candidate_pareto.json` | `cli/autoqec.py::run` via `RunMemory(..., pareto_filename=...)` | after each demo round in the no-LLM CLI path |
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

`candidate_pareto.json` is demo/reporting output only. Orchestration L2/L3 readers must continue to treat `pareto.json` as the authoritative, verifier-owned front.

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

`checkpoint_path` and `training_log_path` inherit the absoluteness of the
`RunnerConfig.round_dir` the caller passes in. `scripts/e2e_handshake.py`
and `scripts/run_single_round.py` pass absolute paths (via
`Path(...).resolve()`), so their rounds record absolute paths in
`metrics.json`. **[TODO-fill-in, Lin]** `cli/autoqec.py::run` currently
composes `run_dir = Path("runs") / run_id` (cwd-relative) and forwards
that to `RunnerConfig.round_dir` — tighten with `.resolve()` before
passing, so every writer produces absolute paths regardless of cwd.

No extra keys beyond `RoundMetrics`.

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

Tab-separated. One line per batch. Consumed by `/diagnose-failure`
(Xie, Day-3). The `machine_state` `params_vs_time` scatter does **not**
read `train.log` — it derives `(n_params, train_wallclock_s +
eval_wallclock_s)` from `history.jsonl`.

## Invariants

**Enforced now:**

- Orchestration-written text files (`history.jsonl`, `log.md`,
  `pareto.json`) are opened with explicit `encoding="utf-8"` and
  `json.dumps(..., ensure_ascii=False)` — Chinese, Δ, and other
  non-ASCII content round-trips cleanly on Windows. Covered in
  `test_orchestration_stub.py::test_run_memory_append_log_roundtrips_utf8`.
- Subagent response JSON validates against
  `IdeatorResponse`/`CoderResponse`/`AnalystResponse` before its
  contents are mirrored into `history.jsonl`. Covered in
  `test_orchestration_stub.py::test_parse_response_enforces_*`.
- `l3_for_analyst` passes an absolute `metrics_path` to the Analyst
  even when the caller supplied a relative `round_dir`. Covered in
  `test_orchestration_stub.py::test_l3_for_analyst_metrics_path_is_absolute`.
- `_gpu_snapshot` returns `{}` on any CUDA/driver failure, not just
  missing torch. Covered in
  `test_machine_state.py::test_gpu_snapshot_swallows_driver_errors_from_is_available`.

**[TODO-fill-in] aspirational, not enforced yet:**

- `round_<N>/` exists **before** any `history.jsonl` entry with
  `round == N` is written. (Needs a cross-component integration test in
  Day-3 after the orchestrator → Runner loop is wired.)
- Runner-written text files (`config.yaml`, `train.log`, `metrics.json`)
  use `encoding="utf-8"` — current Runner code relies on locale default
  (see `autoqec/runner/runner.py:61, 163, 230`). **[TODO-fill-in, Lin]**
  add explicit `encoding="utf-8"`.

## Non-goals

- No per-run database. `runs/` is append-only disk; post-processing can
  build whatever index it wants on top of `history.jsonl`.
- No automatic cleanup. Old `runs/` directories persist until a human
  removes them; `.gitignore` already excludes the directory so this is
  safe.
