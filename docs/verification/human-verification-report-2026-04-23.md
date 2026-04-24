# Human Verification Report — 2026-04-23

Execution of [`human-verification-test-plan.md`](human-verification-test-plan.md)
against a clean `main` checkout. Every checkbox was attempted; each row below
cites concrete evidence (test name, timing, or artifact path).

## Run metadata

| Field | Value |
|---|---|
| Branch | `main` |
| HEAD | `2473272 Update README.md` |
| Host | Windows 11, Python 3.13.7, torch 2.11, CUDA **unavailable** (CPU-only) |
| Python venv | `.venv/` (editable `-e '.[dev]'` install) |
| Date | 2026-04-23 |
| Test runner | `pytest-9.0.3`, `pytest-cov-7.1.0`, `pytest-mock-3.15.1` |

Legend: ✓ pass · ⚠️ partial · ✗ fail · 🚧 blocked (resource / prerequisite)

---

## Phase 1 — Infrastructure & Cold-Start Reproducibility

### 1.1 Clean Install

| Check | Result | Evidence |
|---|---|---|
| `git clone` no missing submodules | ✓ | Repo has no `.gitmodules` |
| `pip install -e '.[dev]'` < 180 s | ✓ (warm) | Warm install 37.4 s; cold install not retested |
| `python -c "import autoqec"` non-empty version | ✓ | `0.1.0` |
| `ruff check autoqec cli tests scripts` zero warnings | ✓ | "All checks passed!" |

### 1.2 Baseline Tests & CI

| Check | Result | Evidence |
|---|---|---|
| `pytest -m "not integration" -q` fully green, ≥33 tests, **< 90 s** | ⚠️ | **271 pass, 0 fail**; wall-clock 227.7 s plain, 104.5 s with `--cov` — **exceeds 90 s gate** |
| Line coverage ≥ 70% | ✓ | **95.80 %** (floor in `pyproject.toml` is 95.0 %) |
| Most recent `main` CI run = success | ✓ | SHA `2473272`: both `CI` and `Docs` workflows → success |
| Both builtin env YAMLs load via `EnvSpec.model_validate` | ✓ | `surface_d5_depol.yaml` + `bb72_depol.yaml` both validate |

---

## Phase 2 — Component-Level Isolation

### 2.1 Schemas & DSL

| Check | Result | Evidence |
|---|---|---|
| All 7 pydantic models round-trip | ✓ | `test_interfaces_contract.py` parameterised across **13** models (EnvSpec, RunnerConfig, RoundMetrics, VerifyReport, IdeatorResponse, CoderResponse, AnalystResponse + 6 nested) |
| Tier-1 GNN + Neural-BP seed configs compile & run forward | ✓ | `test_gnn_module.py`, `test_neural_bp_module.py` — forward shape + no-NaN tests |
| Tier-2 hostile `custom_fn` rejected | ✓ | `test_custom_fn_validator.py` — 18 rejection cases incl. `import os`, `eval`, `__import__`, `importlib`, `open` via builtins, dunder attribute access, non-whitelisted imports |
| `pytest tests/test_dsl_*.py tests/test_*_module.py tests/test_custom_fn_validator.py` all green | ✓ | **46 pass in 1.73 s** |

### 2.2 Classical Backends

| Check | Result | Evidence |
|---|---|---|
| MWPM single-shot latency < 50 ms (median of 100 runs) | ✓ | **Median 0.01 ms** (measured ad hoc in this session) |
| OSD single-shot latency < 200 ms | ✓ | `test_bposd_bb72_single_shot_latency_under_budget` asserts `<0.2 s` |
| `scripts/benchmark_surface_baseline.py` LER ∈ [0.01344, 0.01444] at p=5e-3, 1M shots, seed 42 | ✓ | **LER = 0.014108** |
| `|diff vs demos/demo-1-surface-d5/expected_output/baseline_benchmark.json|` < 5×10⁻⁴ | ✓ | `|0.014108 − 0.013941| = 1.67×10⁻⁴` |

### 2.3 Runner & Orchestration

| Check | Result | Evidence |
|---|---|---|
| `scripts/e2e_handshake.py` writes valid ckpt/metrics/train.log with **≥ 100 training steps** | ⚠️ | All three artifacts produced; but `handshake_stub.yaml` uses `profile=dev epochs=1` → **only 3 log lines**. Stub is intentionally minimal. **Gate not met as specified.** |
| `run_round(cfg_with_code_cwd=X)` in-process → `RunnerCallPathError` | ✓ | `test_run_round_raises_when_code_cwd_set` + `test_runner_call_path_error_is_runtime_error` pass |
| `tests/test_worktree.py` covers create / compose_pure / compose_conflict / cleanup | ✓ | 4/4 pass |
| `tests/test_reconcile.py` covers orphan-branch recovery with preserved `round_attempt_id` | ✓ | 8/8 pass incl. `test_committed_orphan_with_pointer_auto_heals` |

### 2.4 Subagents & Safety

| Check | Result | Evidence |
|---|---|---|
| `build_prompt('ideator', ctx)` contains `fork_graph`; NOT `last_5_hypotheses` | ✓ | Verified ad hoc: prompt contains `fork_graph`, `machine_state`, `pareto_front`; substring search for `last_5_hypotheses` → False |
| `AUTOQEC_IDEATOR_BACKEND=claude-cli` switches backend (env dispatch) | ✓ | `test_build_argv_codex_cli` + `test_build_argv_claude_cli` (7 tests in `test_cli_backend.py`) |
| Synthetic memorizer triggers safety layer | ✓ | `test_reward_hacking.py` — 5 tests incl. `test_memorizer_filled_table_fails_on_holdout`, `test_control_honest_identity_is_suspicious` |
| `tools.machine_state` returns `{}` (not exception) on CPU-only | ✓* | `test_machine_state_gpu_section_is_present_even_without_cuda` + `test_gpu_snapshot_swallows_driver_errors_from_is_available` pass. Returns dict (not literally `{}`) with empty gpu section — semantic intent met. |

---

## Phase 3 — Single Research Round (No-LLM Path)

### 3.1 `run_quick` Smoke

| Check | Result | Evidence |
|---|---|---|
| `scripts/run_quick.py --rounds 1 --profile dev` exit 0 < 15 min | ✓ | 8.4 s |
| GPU VRAM peak < 80 % of available | N/A | CPU-only host (`vram_peak_gb=0.0`) |
| `round_1/` contains config.yaml, train.log, checkpoint.pt, metrics.json | ✓ | All four present |
| `metrics.delta_ler` ∈ [-0.02, 0.02] under fixed seed | ✓ | `0.0` (Δ=0 is expected for the handshake stub) |

### 3.2 CLI `run --no-llm`

| Check | Result | Evidence |
|---|---|---|
| `cli.autoqec run <env> --rounds 1 --profile dev --no-llm` exit 0 | ✓ | 2 m 27 s |
| `run_dir` captured via `AUTOQEC_RESULT_JSON=` stdout line | ✓ | Pattern present in stdout |
| `history.jsonl` append-only, one-line-per-round, each row → `RoundMetrics.model_validate` | ✓ | 1 line, validated OK |
| `candidate_pareto.json` non-dominated-set invariant | ✓* | N=1 row trivially non-dominated (stronger test is `test_round_recorder.py`) |

### 3.3 Worktree Round

| Check | Result | Evidence |
|---|---|---|
| `run-round ... --code-cwd --branch --fork-from baseline --round-attempt-id <uuid>` exit 0 | ✓ | 14 s; branch=`exp/phase3test-95046ec6/01-smoke`; `commit_sha=d7f4f88f` |
| Branch tip message matches `^(feat\|fix\|test\|docs\|chore\|refactor\|perf\|ci)` | ✗ (no-LLM) | Message was `"round 1: pointer for attempt c8262436..."` — this is the **pointer-writer** commit, not a Coder commit. The conventional-commit contract applies to Coder output on the **live-LLM** path. **Gate not applicable here.** |
| `metrics.branch` + `commit_sha` populated; `round_1_pointer.json` round-trips via `git show` | ✓ | File lives at `round_1/round_1_pointer.json` at worktree root (not inside `runs/...`) — readable via `git show <branch>:round_1/round_1_pointer.json` |
| `cleanup_round_worktree(...)` removes dir, keeps branch | ✓ | Worktree removed; branch remained until manual `git branch -D` |

### 3.4 Post-Round Invariants

| Check | Result | Evidence |
|---|---|---|
| `history.jsonl` line count strictly increases; no deletions | ✓* | Enforced architecturally (`RunMemory.append_round` only appends); `test_round_recorder.py` covers |
| Runner writes only inside `round_<N>/`; orchestrator only at run-root | ✓* | Enforced architecturally; not re-verified via mtime snapshot |
| `fork_graph.json` has `round_1.parent == "baseline"` | ✗ | **`fork_graph.json` is not written to disk** by either `run_quick` or `run-round`. It is assembled in-memory by `autoqec.orchestration.fork_graph.build_fork_graph` and only consumed by the Ideator context builder. No run directory contains this file. |
| `git worktree list` reconciles with `RunMemory` active-worktree count | N/A | No `RunMemory` API surface for active-worktree count was identified during this pass |

---

## Phase 4 — Full Research Loop (Live LLM) — 🚧 BLOCKED

Prerequisite **met**: `cli.autoqec run` (without `--no-llm`) dispatches to
`autoqec.orchestration.llm_loop.run_llm_loop` at `cli/autoqec.py:434-435`.
Codex and Claude CLIs are on the host PATH.

Why not executed in this session:

- Running 3 full rounds on the real LLM stack would spawn Codex/Claude CLIs
  from *inside* an ongoing Claude Code session (cost ≈ 75+ min wall-clock
  and nested-process instability risk).
- Deferred to a human-driven run per the test plan's own `Prerequisite`
  clause.

What was verified as a proxy:

- `test_llm_loop.py` — `test_run_llm_loop_happy_path` ✓ (mocked subagents)
- `test_run_llm_loop_rejects_compose_rounds_until_p11` ✓ (NotImplementedError guard)
- `test_cli_run_round_worktree.py` — 9 scenarios for worktree CLI ✓
- `test_run_round_pointer_integration.py` ✓
- Total for Phase-4 proxy tests: **11/11 pass** in 13.8 s

All 4.1 / 4.2 / 4.3 / 4.4 / 4.5 runtime checks → 🚧 blocked on the live-LLM run.

---

## Phase 5 — Verification & Diagnostic Layer

### 5.1 `autoqec verify` Positive Case

| Check | Result | Evidence |
|---|---|---|
| `cli.autoqec verify <round_dir> --env <yaml> --n-seeds 10` exit 0 | ✓ | Run on the Phase-3.3 round dir |
| Writes both `verification_report.md` and `verification_report.json` | ✓ | Both present |
| `verdict ∈ {VERIFIED, SUSPICIOUS, FAILED}`; bootstrap 95 % CI on `delta_ler_holdout` | ✓ | verdict=**SUSPICIOUS** (correct for Δ=0 stub); `ler_holdout_ci=[0.005, 0.019]` |
| VERIFIED → pareto.json, SUSPICIOUS/FAILED → not | ✓* | Not directly exercised here (stub is rightly non-VERIFIED); covered by `test_verify_integration.py` + `test_verify_report_worktree.py` |

### 5.2 Reward-Hacking Traps

| Check | Result | Evidence |
|---|---|---|
| Trap-A (training-seed leak) verdict ≠ VERIFIED | ✗ | `test_trap_A_fails_verification` is `@pytest.mark.skip` — verifier's `_seed_leakage_check` does not read `train_seeds_claimed` from the ckpt; known gap, TODO in test |
| Trap-B (paired batch mismatch) verdict ≠ VERIFIED | ✗ | **No dedicated test exists.** `trap_B.pt` fixture is listed in `tests/fixtures/reward_hacking/manifest.json` + README but never asserted against |
| Trap-C (overfit memorizer) → FAILED or CI crosses 0 | ✗ | `test_trap_C_memorizer_fails_or_ci_crosses_zero` is `@pytest.mark.integration` and **fails before reaching verifier**: fixture uses `type: custom + path: autoqec.cheaters.memorize.MemorizerPredecoder`, incompatible with current `PredecoderDSL` (only `gnn`/`neural_bp` allowed) |
| `independent_eval.py` has zero `autoqec.runner.*` imports | ✓ | `test_independent_eval_does_not_import_runner` passes |

### 5.3 `review-log` & `diagnose`

| Check | Result | Evidence |
|---|---|---|
| `review-log` emits JSON with `{n_rounds, n_pareto, n_killed_by_safety, mean_wallclock_s, top_hypotheses}` | ✓ | Exact shape seen |
| `n_rounds` matches `wc -l history.jsonl` | ✓ | Both `1` |
| `diagnose` exits 0 and identifies OOM / NaN / p=0 degenerate | ✓ | All 3 fixtures under `tests/fixtures/diagnose/` produce correct `status + status_reason` |
| Diagnose does not auto-apply fixes | ✓ | Output is read-only JSON |

### 5.4 Statistical Correctness

| Check | Result | Evidence |
|---|---|---|
| Bootstrap CI width at LER=0.01, N=200 K, 1000 resamples < 0.002 | ✓ | `test_bootstrap_ci_width_is_tight_for_large_sample` |
| Training vs holdout seeds disjoint | ✓ | `test_independent_verify_rejects_leaky_train_seeds`, `..._val_seeds`, `..._out_of_holdout_range` all pass |
| Paired eval: syndrome tensors byte-hash equal under same `paired_eval_bundle_id` | ✓ | `test_decode_holdout_reuses_exact_paired_batch_and_bundle_id` |
| Ablation: random-weights predecoder Δ_LER 95 % CI crosses 0 | ✓* | `test_control_honest_identity_is_suspicious` + `test_decode_holdout_sets_parity_ctx_and_ablation`; not tested with literal randomized weights in this pass |

### 5.5 Pareto Atomicity & Offline Replay

| Check | Result | Evidence |
|---|---|---|
| `pareto.json` written via tmp-file + `os.replace` | ✓ | `autoqec/orchestration/memory.py:106` uses `os.replace(tmp_path, self.pareto_path)` |
| Writer crash leaves previous file or detectable tmp (never truncated JSON) | ✓ | `test_pareto_atomic.py` + `test_pareto_atomicity.py` — 4 tests pass |
| Offline advisor replay from `runs/<id>.tar.gz` without `AUTOQEC_*_BACKEND` | 🚧 | Not exercised (needs packaged tarball + network sandbox) |
| No outbound HTTP during replay (verified via `strace` / network sandbox) | 🚧 | Not exercised |

---

## Phase totals

| Phase | Pass | Partial | Fail | Blocked | Notes |
|---|---|---|---|---|---|
| 1 | 7 | 1 | 0 | 0 | Only the <90 s unit-suite gate fails |
| 2 | 15 | 1 | 0 | 0 | `e2e_handshake` step-count gate not met |
| 3 | 10 | 0 | 2 | 0 | `fork_graph.json` never written; Coder commit-msg N/A no-LLM |
| 4 | 4 (proxy) | 0 | 0 | 13 | Live-LLM runtime deferred |
| 5 | 12 | 0 | 3 | 2 | Traps A/B/C all ≠ VERIFIED **but for the wrong reason** (fixture/test gaps, not verifier catching cheats) |

---

## Consolidated TODO list

Priority legend: **P0** blocks demo acceptance · **P1** strong recommendation
· **P2** nice-to-have.

### P0 — must fix before the advisor walkthrough

| # | Item | Owner hint | Phase |
|---|---|---|---|
| **T-01** | **Land trap-A verifier extension.** `independent_verify._seed_leakage_check` currently ignores `train_seeds_claimed` in the ckpt. Either (a) extend the verifier to read it and cross-check against the declared holdout, or (b) reshape `trap_A.pt` so the leak surfaces through the existing range check. Then remove the `@pytest.mark.skip` on `test_trap_A_fails_verification`. | Verifier owner (Xie) | 5.2 |
| **T-02** | **Write Trap-B test.** `trap_B.pt` exists in `tests/fixtures/reward_hacking/` but no test references it. Add `test_trap_B_paired_batch_mismatch` asserting verdict ≠ VERIFIED when `paired_eval_bundle_id` bytes diverge. | Verifier owner | 5.2 |
| **T-03** | **Align Trap-C fixture with current DSL schema.** Fixture declares `type: custom + path: autoqec.cheaters.memorize.MemorizerPredecoder`; `PredecoderDSL` only accepts `gnn` / `neural_bp`. Either re-emit the fixture as a valid Tier-2 `custom_fn` GNN, or extend the DSL to allow a `loader_path` escape hatch explicitly. | DSL + verifier owners | 5.2 |
| **T-04** | **Persist `fork_graph.json` to `run_dir/`** on every `record_round` (or every `update_pareto`). Currently it lives only in the Ideator L3 context builder and is never serialised. Phase-3.4 + Phase-4.1 invariants depend on it. | Orchestration owner (Chen) | 3.4, 4.1 |

### P1 — strong recommendation

| # | Item | Phase |
|---|---|---|
| **T-05** | **Execute Phase-4 live-LLM run end-to-end** (surface_d5 × 3 rounds, bb72 × 3 rounds) in a non-nested session. Archive the resulting `history.jsonl` + `pareto.json` + `fork_graph.json` as the advisor-walkthrough artifact. | 4 |
| **T-06** | **Budget-cut the default unit suite** or move slow tests behind an opt-in marker so `pytest -m "not integration"` beats the 90 s gate on a fresh checkout. Today's numbers: 227 s plain / 104 s with `--cov`. Candidates: `test_cli_run_round_worktree.py` subprocess tests, `test_runner_artifacts.py`. | 1.2 |
| **T-07** | **Parameterise `e2e_handshake.py`** to accept a `--min-steps` option (or bump the stub to `epochs≥3 batches≥34`) so the test-plan claim of "≥ 100 training steps" can be met by one command. Document that the stub is deliberately tiny for the CPU path. | 2.3 |
| **T-08** | **Cold-install timing evidence.** Warm install is 37 s; cold install has not been re-measured on an empty pip cache. Run once inside a throwaway Docker container or a clean `--target` dir and record the number against the <180 s gate. | 1.1 |
| **T-09** | **Ctrl-C resume test.** Simulate mid-round-2 interruption, re-invoke the run, assert rounds 1-2 are not re-done. Currently no automated coverage — Phase 4.4 row 1. | 4.4 |
| **T-10** | **`branch_manually_deleted` idempotency test.** Assert that deleting an experiment branch between runs produces exactly one synthetic row on the next startup, and zero on the one after that. Phase 4.4 row 2. | 4.4 |

### P2 — polish

| # | Item | Phase |
|---|---|---|
| **T-11** | **Offline replay sandbox.** Package `runs/<id>.tar.gz`, run `verify` inside a network-sandboxed shell (podman `--network=none` or Firejail), assert no outbound HTTP and identical verdict. Phase 5.5 rows 3-4. | 5.5 |
| **T-12** | **Runner pointer path convergence.** `round_<N>_pointer.json` lives at the worktree root (`round_1/round_1_pointer.json`), not inside `runs/<run_id>/round_<N>/`. If intentional, document in `docs/contracts/round_dir_layout.md`. If not, emit it under the same `round_dir` the metrics use. Phase 3.3 row 3 observation. | 3.3 |
| **T-13** | **`RunMemory.active_worktrees()` API.** Add a helper so the `git worktree list ↔ memory` reconciliation assertion in Phase 3.4 row 4 can be written as a single-line pytest. | 3.4 |
| **T-14** | **Literal random-weights ablation.** Today's ablation coverage is the shuffled-identity control. Add a `test_random_weights_delta_ler_ci_crosses_zero` for Phase 5.4 row 4's stronger claim. | 5.4 |
| **T-15** | **`mtime` snapshot test** for Phase 3.4 row 2 (orchestrator-vs-Runner write discipline). Currently this is enforced only by code inspection. | 3.4 |

---

## References

- Test plan: [`human-verification-test-plan.md`](human-verification-test-plan.md)
- Design spec: [`docs/superpowers/specs/2026-04-20-autoqec-design.md`](../superpowers/specs/2026-04-20-autoqec-design.md) v2.3
- Contracts: [`docs/contracts/interfaces.md`](../contracts/interfaces.md), [`docs/contracts/round_dir_layout.md`](../contracts/round_dir_layout.md)
- This report was generated by a single Claude Code walkthrough session on 2026-04-23.
