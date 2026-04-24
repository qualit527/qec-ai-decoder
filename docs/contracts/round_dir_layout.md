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
| `round_<N>_pointer.json` | `run_round` via `autoqec.runner.pointer.write_round_pointer` | at round end, whenever `cfg.branch` and `cfg.code_cwd` are both set |
| `artifact_manifest.json` | `run_round` via `autoqec.runner.artifact_manifest.write_artifact_manifest` | at round end for successful Runner rounds |
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

`run_round` resolves `RunnerConfig.round_dir` before it writes artifacts,
so `checkpoint_path` and `training_log_path` are absolute in every
emitted `metrics.json`. Downstream readers should treat those fields as
the canonical artifact locations instead of reconstructing paths by hand.

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
- Runner-written text files (`config.yaml`, `train.log`, `metrics.json`)
  are written with explicit `encoding="utf-8"`.
- Failure-path rounds still emit `metrics.json`, and absent artifacts stay
  `null` in `RoundMetrics` instead of claiming files that were never
  produced.
- `round_<N>/round_<N>_pointer.json` is written for worktree-branch
  rounds so startup reconciliation can recover the `round_attempt_id`.
  Covered in `test_pointer_writer.py` (unit) and the worktree
  subprocess integration tests.
- `round_<N>/artifact_manifest.json` is written with the canonical nested
  schema from `autoqec.runner.artifact_manifest.write_artifact_manifest`,
  capturing `repo.commit_sha`, `repo.dirty`, package versions,
  `environment.env_yaml_sha256`, `round.dsl_config_sha256`, and
  `round.command_line`.
  Manifest-writer failures never fail the round — they degrade to a
  `round_<N>/manifest_error.txt` note. Covered in
  `test_artifact_manifest.py` (unit).

**[TODO-fill-in] aspirational, not enforced yet:**

- `round_<N>/` exists **before** any `history.jsonl` entry with
  `round == N` is written. (Needs a cross-component integration test in
  Day-3 after the orchestrator → Runner loop is wired.)
## Non-goals

- No per-run database. `runs/` is append-only disk; post-processing can
  build whatever index it wants on top of `history.jsonl`.
- No automatic cleanup. Old `runs/` directories persist until a human
  removes them; `.gitignore` already excludes the directory so this is
  safe.

---

## §15 Additions

Added 2026-04-22 for the worktree experiment model. Authoritative
source: `docs/superpowers/specs/2026-04-20-autoqec-design.md` §15.5
(pointer file) and §15.7 (pareto / pareto_preview split).

### `round_N_pointer.json` (committed inside the worktree)

**Enforced as of 2026-04-23 (Runner writes this whenever the round
runs with a branch).** Producer: `autoqec.runner.runner.run_round` via
`autoqec.runner.pointer.write_round_pointer`. The pointer is written
unconditionally inside the `if config.branch is not None and
config.code_cwd is not None:` block — i.e. whenever the round runs on
a worktree branch — so §15.10 startup reconciliation can always
recover `round_attempt_id` even when the post-training `git commit`
step fails.

Written by the worktree-side Runner after training finishes, then
committed to the experiment branch `exp/<run_id>/<NN>-<slug>`. This is
the single on-branch file linking the committed code state to the
external shared `runs/<run_id>/round_<N>/` artifact directory.

Provenance fields are **all REQUIRED**; consumers reject pointer files
missing any. `round_attempt_id` is persisted here so §15.10 startup
reconciliation can recover it if the process crashes between
pointer-commit and `history.jsonl` append.

```json
{
  "run_id": "<YYYYMMDD-HHMMSS>",
  "round_idx": <int>,
  "round_attempt_id": "<UUID>",
  "branch": "exp/<run_id>/<NN>-<slug>",
  "commit_sha": "<full SHA>",
  "fork_from": "baseline" | "<branch>" | [<branches>],
  "fork_from_canonical": "<sorted|joined>",
  "fork_from_ordered": <list or null>,
  "provenance": {
    "env_yaml_sha256":         "<sha256>",
    "dsl_config_sha256":       "<sha256>",
    "requirements_fingerprint": "<short>",
    "repo_root_resolved":      "<absolute path>"
  },
  "metrics_summary": {
    "delta_vs_parent":    <float>,
    "flops_per_syndrome": <int>,
    "n_params":           <int>,
    "status":             "ok" | "killed_by_safety" | ...
  },
  "artifact_paths": {
    "checkpoint": "<absolute>",
    "metrics":    "<absolute>",
    "train_log":  "<absolute>"
  }
}
```

`artifact_paths.*` are **absolute** so the pointer survives relocation
of the orchestrator's cwd. `provenance.repo_root_resolved` enables
relative-path reconstruction if the run is moved to a new machine.
`requirements_fingerprint` is a short string derived from the frozen
dependency set (e.g. `pip freeze | sha256 | head`); post-MVP will
upgrade to a full lockfile digest.

`round_attempt_id` and `reconcile_id` are mutually exclusive per §15.2:
a row is either a real attempt or a reconciliation synthetic, never
both. The pointer file only ever carries `round_attempt_id` — synthetic
rows are produced by §15.10 startup reconciliation and never have a
pointer file.

Worktree scope reminder (from §15.5): `runs/<id>/round_N/checkpoint.pt`,
`metrics.json`, and `train.log` stay **outside the worktree** on a
shared path; every branch references them by absolute path through this
pointer file.

### `pareto.json` — complete non-dominated archive (NOT capped)

Replaces the previous "capped to 5 entries" behaviour. `pareto.json`
stores the **full non-dominated set** of VERIFIED branches. Admission
requires a committed round (verdict=VERIFIED), so `commit_sha` and
`branch` are always non-null; compose-conflict rows never appear here.

Row schema:

```json
{"round_attempt_id":           "<UUID>",
 "commit_sha":                  "<SHA>",
 "branch":                      "exp/.../<NN>-<slug>",
 "delta_vs_baseline_holdout":   <float>,
 "paired_eval_bundle_id":       "<bundle-id>",
 "flops_per_syndrome":          <int>,
 "n_params":                    <int>,
 "verdict":                     "VERIFIED",
 "fork_from":                   "baseline" | "<branch>" | [<branches>],
 "fork_from_canonical":         "<sorted|joined>",
 "compose_mode":                "pure" | "with_edit" | null}
```

See `docs/superpowers/specs/2026-04-20-autoqec-design.md` §15.2 for the
`round_recorder.py` update from top-5 sort to non-dominated filter.

### `pareto_preview.json` — derived top-5 (for L2 Ideator context)

Regenerated after every `pareto.json` mutation; sorted by
`-delta_vs_baseline_holdout` and truncated to 5. Consumers that read
only the preview MUST NOT claim to report the full archive.

### `.worktrees/` (run-scoped, shared across rounds)

`.gitignore` excludes `.worktrees/`. Worktree directory naming:
`.worktrees/exp-<run_id>-<NN>-<slug>/`. Branch naming:
`exp/<run_id>/<NN>-<slug>`; orphan recovery by §15.10 reconciliation
renames to `quarantine/<run_id>/<remainder>`.
