# Live LLM 3-Round Runs — 2026-04-24

This note archives the first successful non-nested live-LLM 3-round runs for
the AutoQEC CLI loop after landing the worktree/live-run hardening fixes in
commit `c4448f7`.

## Routing used

- Date: 2026-04-24
- Repo branch used for the runs: `feat/issue-51-live-llm-verification`
- Code commit used for both runs: `c4448f7`
- Backend routing used for the runs:
  - `AUTOQEC_IDEATOR_BACKEND=claude-cli`
  - `AUTOQEC_CODER_BACKEND=claude-cli`
  - `AUTOQEC_ANALYST_BACKEND=claude-cli`
  - model for all three roles: `claude-haiku-4-5`

`codex exec` was still returning `401 Unauthorized` in this shell during the
verification session, so the live artifacts below validate the real CLI DAG and
worktree path with `claude-cli` on all three roles rather than the mixed
Codex+Claude routing originally planned in issue #51.

## Commands

```bash
AUTOQEC_IDEATOR_BACKEND=claude-cli AUTOQEC_IDEATOR_MODEL=claude-haiku-4-5 \
AUTOQEC_CODER_BACKEND=claude-cli AUTOQEC_CODER_MODEL=claude-haiku-4-5 \
AUTOQEC_ANALYST_BACKEND=claude-cli AUTOQEC_ANALYST_MODEL=claude-haiku-4-5 \
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml --rounds 3 --profile dev

AUTOQEC_IDEATOR_BACKEND=claude-cli AUTOQEC_IDEATOR_MODEL=claude-haiku-4-5 \
AUTOQEC_CODER_BACKEND=claude-cli AUTOQEC_CODER_MODEL=claude-haiku-4-5 \
AUTOQEC_ANALYST_BACKEND=claude-cli AUTOQEC_ANALYST_MODEL=claude-haiku-4-5 \
python -m cli.autoqec run autoqec/envs/builtin/bb72_depol.yaml --rounds 3 --profile dev
```

## Archived artifacts

- `runs/20260424-154450/`
- `runs/20260424-155219/`
- `docs/verification/artifacts/autoqec-surface-d5-live-3rounds-2026-04-24.tar.gz`
- `docs/verification/artifacts/autoqec-bb72-live-3rounds-2026-04-24.tar.gz`

## Summary

| Env | Run dir | History rows | Statuses | Fork nodes | Machine-state calls | Agent input / output tokens | Agent wall-clock | Train+eval wall-clock | Cost |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| `surface_d5_depol` | `runs/20260424-154450` | 3 | `ok`, `train_error`, `ok` | 4 | 3 | 439,038 / 10,140 | 239.128 s | 73.310 s | $0.522608 |
| `bb72_depol` | `runs/20260424-155219` | 3 | `ok`, `train_error`, `ok` | 4 | 3 | 337,638 / 7,441 | 159.757 s | 7.291 s | $0.398497 |

## What Landed

- The live CLI DAG now runs each round from a committed experiment worktree.
- Coder subprocesses execute inside the experiment worktree `cwd`.
- Ideator prompt context is persisted to `round_N/ideator_prompt.json`.
- Role responses now persist usage metadata in `*_response.json`.
- `log.md` records `{"tool_call": "machine_state", ...}` once per round.
- Pointer commits now use a conventional-commit subject:
  `chore(pointer): ...`
- Worktree child rounds now keep the parent venv interpreter instead of
  resolving back to the base Conda Python.
- Runner training failures now degrade into `status="train_error"` rows instead
  of aborting the whole multi-round live run.

## Acceptance Review

### Confirmed

- Both live commands exited 0 and wrote exactly 3 `history.jsonl` rows.
- Every `round_attempt_id` was unique in both runs.
- `fork_graph.json` exists and has 4 nodes in both runs.
- All branch-tip commit subjects match the conventional-commit regex because
  the tip is now the `chore(pointer): ...` provenance commit.
- `machine_state` was logged 3 times in each run.
- Tarball archives were produced for both runs.
- Total token usage stayed within the provisional 500K input / 80K output
  budget for both environments.

### Deviations

- Both runs emitted `{"warning": "all_rounds_baseline_fork_from", "rounds": 3}`
  in `log.md`. The ideator never selected a non-baseline parent, so the older
  issue text's stricter "`parent != baseline` at least once" expectation was
  not met in these artifacts.
- `pareto.json` stayed empty in both runs because no round produced a
  `VERIFIED` holdout result.
- Round 2 in both runs became `train_error` because the live LLM proposed a
  non-differentiable candidate; this is now recorded as a normal history row
  rather than aborting the run.
- `bb72`'s backend is evidenced by the archived env path in
  `round_1/artifact_manifest.json`:
  `autoqec/envs/builtin/bb72_depol.yaml` hashes to the run and declares
  `classical_backend: osd`. By current repo contract, `round_1/config.yaml`
  remains a literal predecoder DSL dump and does not itself embed
  `classical_backend`.

## Evidence pointers

- `surface_d5` machine-state log: `runs/20260424-154450/log.md`
- `bb72` machine-state log: `runs/20260424-155219/log.md`
- `surface_d5` ideator prompt snapshot: `runs/20260424-154450/round_1/ideator_prompt.json`
- `bb72` ideator prompt snapshot: `runs/20260424-155219/round_1/ideator_prompt.json`
- `surface_d5` usage envelope example: `runs/20260424-154450/round_1/ideator_response.json`
- `bb72` usage envelope example: `runs/20260424-155219/round_1/ideator_response.json`
