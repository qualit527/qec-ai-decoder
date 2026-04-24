# Human Verification Retest — 2026-04-24

Follow-up to [`human-verification-report-2026-04-23.md`](human-verification-report-2026-04-23.md).
Only the **previously-unresolved items** (⚠️ partial, ✗ fail, 🚧 blocked) were
re-tested, in test-plan order. Items that were ✓ on 2026-04-23 are unchanged
and not re-verified in this pass.

## Run metadata

| Field | Value |
|---|---|
| Branch | `main` |
| HEAD | `ba91caf Merge pull request #59 from qualit527/feat/issue-37-bb72-demo` |
| Delta vs previous report | PR #58 (P0 trap + fork_graph fixes) and PR #59 (bb72 demo) now merged |
| Host | Windows 11, Python 3.13.7, torch 2.11, CUDA **unavailable** (CPU-only) |
| Python venv | `.venv/Scripts/python.exe` (editable `-e '.[dev]'` install) |
| Date | 2026-04-24 |

Legend: ✓ pass · ⚠️ partial · ✗ fail · 🚧 blocked (resource / prerequisite)

---

## Re-tested items (in test-plan order)

### Phase 1.2 — Unit suite wall-clock gate `< 90 s`

| Metric | 2026-04-23 | 2026-04-24 | Gate |
|---|---|---|---|
| Tests collected | 271 | **290** (+19) | — |
| Pass / fail | 271 / 0 | **290 / 0** | — |
| Wall-clock | 227.7 s (plain), 104.5 s (--cov) | **257.7 s** (plain) | <90 s |
| Verdict | ⚠️ partial | **⚠️ partial** | **Gate still missed** |

Command: `./.venv/Scripts/python.exe -m pytest tests/ -m "not integration" -q`.
The 19 additional tests land under `test_fork_graph_persist.py`,
`test_demo_bb72_assets.py`, `test_run_bb72_demo.py`, `test_demo4_snapshot.py`.
Runtime grew proportionally — no regressions, no speedups.

**Action**: T-06 (P1) remains open. Either move slow subprocess tests behind
an opt-in marker, or cut the default budget.

---

### Phase 2.3 — `scripts/e2e_handshake.py` ≥ 100 training steps

| Metric | 2026-04-23 | 2026-04-24 | Gate |
|---|---|---|---|
| Artifacts written | ckpt + metrics + train.log | ckpt + metrics + train.log | all three present |
| `wc -l train.log` | 3 | **3** | ≥100 |
| Verdict | ⚠️ partial | **⚠️ partial** | **Gate still missed** |

Command: `./.venv/Scripts/python.exe scripts/e2e_handshake.py`;
log at `runs/handshake/round_0/train.log`.

The stub uses `handshake_stub.yaml` (`profile=dev, epochs=1`) — intentionally
minimal for the CPU path. The artifact contract is satisfied but the step-count
budget is not. **Action**: T-07 (P1) remains open; accept the stub as "handshake
smoke" and write a separate long-form test that bumps `epochs` / `batches_per_epoch`
on the CPU path.

---

### Phase 3.4 — `fork_graph.json` round_1.parent == "baseline"

| Metric | 2026-04-23 | 2026-04-24 |
|---|---|---|
| `fork_graph.json` written on disk after `run_quick` / `run --no-llm` | ✗ **never written** | **✓ written** at `runs/<id>/fork_graph.json` |
| Atomic swap (tmp + `os.replace`) | N/A | ✓ `test_update_fork_graph_writes_atomically` + crash-resilience test pass |
| Semantic: round_1 references baseline | N/A | ✓ `test_record_round_persists_fork_graph_after_round_one` asserts `round_1.parent == "baseline"` when `fork_from="baseline"` is set |
| Observed in live no-LLM run (`runs/20260423-231636/fork_graph.json`) | — | Baseline node present; round_1 `parent: null` because no-LLM path does not set `fork_from`. Semantic gate is proven by unit test. |
| Verdict | ✗ fail | **✓\*** |

Evidence:
```
$ ls runs/20260423-231636/
candidate_pareto.json  fork_graph.json  history.json  history.jsonl  round_1/
```
Plus 6/6 pass on `tests/test_fork_graph_persist.py` (including atomic-write,
crash-survival, fork-lineage, and narrow-exception regression tests).

**Action**: T-04 (P0) closed by PR #58. The `parent == "baseline"` wording in
the test-plan row holds when the fork decision is explicit (live-LLM path);
the no-LLM path writes `parent: null` because the ideator is not asked. If the
plan reviewer wants the no-LLM path to self-label round 1's parent as
`"baseline"`, that is a one-line change in the no-LLM CLI path — filed as
T-16 (P2) below.

---

### Phase 4 — Full research loop (live LLM)

**Status unchanged: 🚧 still blocked.**

Reason unchanged from 2026-04-23: nested Claude-Code-inside-Claude-Code
session is unstable for spawning Codex/Claude CLIs. Tracked in GitHub issue
#51 (T-05). Proxy tests (`test_llm_loop.py` + `test_cli_run_round_worktree.py`)
still green — not re-timed this pass.

---

### Phase 5.2 — Reward-hacking traps (A / B / C)

Full re-run: `./.venv/Scripts/python.exe -m pytest tests/test_reward_hacking_traps.py -m "integration" --run-integration -v` — **3 / 3 pass in 6.71 s**.

| Trap | 2026-04-23 | 2026-04-24 | Evidence |
|---|---|---|---|
| **Trap-A** — training-seed leak | ✗ (test was `@pytest.mark.skip`; verifier ignored `train_seeds_claimed`) | **✓** | `test_trap_A_fails_verification` passes. Verifier raises `ValueError("train_seeds_claimed leak: …")` via new `_claimed_seeds_leakage_check`. |
| **Trap-B** — paired-batch mismatch | ✗ (no test existed; fixture orphaned) | **✓** | New `test_trap_B_paired_batch_mismatch` passes. Verifier raises `ValueError("paired_batch_mismatch: …")` via new `_paired_batch_mismatch_check` comparing ckpt-pinned `paired_eval_bundle_id` + `recorded_syndrome_sha256` against freshly-computed values. |
| **Trap-C** — overfit memorizer | ✗ (fixture `type:custom` incompatible with `PredecoderDSL`) | **✓** | `test_trap_C_memorizer_fails_or_ci_crosses_zero` passes. Fixture rebuilt with legacy-model ckpt shape (`model`, `state_dict`, `output_mode`, `dsl_config=None`, `trap_kind="overfit_memorizer"`) so it flows through `_load_predecoder`. |
| `independent_eval.py` has zero `autoqec.runner.*` imports | ✓ | ✓ | `test_independent_eval_does_not_import_runner` |

All three P0 trap gates now close. **T-01 / T-02 / T-03 closed by PR #58.**

---

### Phase 5.5 — Offline replay sandbox

**Status unchanged: 🚧 still blocked.** No tarball packaging + network-sandboxed
shell executed this pass. T-11 (P2) remains open.

---

## Updated phase totals

| Phase | Pass | Partial | Fail | Blocked | Delta vs 2026-04-23 |
|---|---|---|---|---|---|
| 1 | 7 | 1 | 0 | 0 | unchanged |
| 2 | 15 | 1 | 0 | 0 | unchanged |
| 3 | **11** | 0 | **1** | 0 | +1 pass, −1 fail (fork_graph.json) |
| 4 | 4 (proxy) | 0 | 0 | 13 | unchanged |
| 5 | **15** | 0 | **0** | 2 | +3 pass, −3 fail (traps A/B/C) |

**Remaining fails (3.3 only): 1** — "Branch tip message matches conventional-commit
regex." This row is N/A for the no-LLM path (the sole commit on the branch is the
Runner pointer-writer commit, not a Coder commit). The conventional-commit
contract applies to the live-LLM Coder output and will be re-verified during
the Phase-4 walkthrough.

---

## Updated TODO status

### P0 — all closed ✓

| # | Item | Status |
|---|---|---|
| ~~T-01~~ | Trap-A verifier extension | **✓ closed by PR #58** (commit `a5fbfd8`) |
| ~~T-02~~ | Trap-B paired-batch test | **✓ closed by PR #58** (commit `a5fbfd8`) |
| ~~T-03~~ | Trap-C fixture vs DSL schema | **✓ closed by PR #58** (commit `39bbb87`) |
| ~~T-04~~ | Persist `fork_graph.json` | **✓ closed by PR #58** (commits `c68b6ce`, `02041a1`, `f1519c8`) |

### P1 — still open (unchanged)

| # | Item | Status |
|---|---|---|
| T-05 | Live-LLM 3-round run (issue #51) | open — needs non-nested session |
| T-06 | Unit suite <90 s budget (issue #52) | open — now 257.7 s with 290 tests |
| T-07 | `e2e_handshake` ≥100 steps (issue #53) | open — still 3 log lines |
| T-08 | Cold-install timing (issue #54) | open — not retested |
| T-09 | Ctrl-C resume test (issue #55) | open |
| T-10 | `branch_manually_deleted` idempotency (issue #56) | open |

### P2 — one item added

| # | Item | Phase | Status |
|---|---|---|---|
| T-11 | Offline replay sandbox | 5.5 | open |
| T-12 | Runner pointer path convergence | 3.3 | open |
| T-13 | `RunMemory.active_worktrees()` API | 3.4 | open |
| T-14 | Literal random-weights ablation | 5.4 | open |
| T-15 | `mtime` snapshot test | 3.4 | open |
| **T-16** | **No-LLM path should set `fork_from="baseline"` for round 1 so `fork_graph.json` round_1 node has `parent == "baseline"` instead of `null`.** | 3.4 | **new (from this retest)** |

---

## Artifacts produced during this retest

- `runs/20260423-231636/` — `run_quick --rounds 1 --profile dev` output; confirms `fork_graph.json` now written
- `runs/handshake/round_0/` — e2e_handshake output (log still 3 lines)

Both directories are gitignored — not checked in.

---

## References

- Previous report: [`human-verification-report-2026-04-23.md`](human-verification-report-2026-04-23.md)
- Test plan: [`human-verification-test-plan.md`](human-verification-test-plan.md)
- PR #58 (P0 fixes): https://github.com/qualit527/qec-ai-decoder/pull/58
- PR #59 (bb72 demo, incidental): https://github.com/qualit527/qec-ai-decoder/pull/59
- Design spec: [`docs/superpowers/specs/2026-04-20-autoqec-design.md`](../superpowers/specs/2026-04-20-autoqec-design.md) v2.3
