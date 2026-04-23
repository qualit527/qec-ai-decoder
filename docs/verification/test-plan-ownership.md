# Test Plan Ownership — 2 : 2 : 1

**Plan:** [`docs/verification/human-verification-test-plan.md`](./human-verification-test-plan.md) (106 checkboxes)
**Design rationale:** [`docs/superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md`](../superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md)

## Split

| Owner | Share | Target checkboxes | Primary subtree |
|---|---|---|---|
| **Chen Jiahan** | 2 | ~42 | `autoqec/orchestration/`, `autoqec/agents/`, `autoqec/tools/`, surface_d5 env + MWPM baseline, Phase 1 infra |
| **Xie Jingu** | 2 | ~42 | `autoqec/eval/`, `autoqec/decoders/baselines/bposd*`, bb72 env + OSD baseline, reward-hacking fixtures, `/verify-decoder` / `/review-log` / `/diagnose-failure` |
| **Lin Tengxiang** | 1 | ~22 | `autoqec/decoders/` (DSL + modules), `autoqec/runner/`, `cli/`, `autoqec/example_db/` |

The 2 : 2 : 1 ratio follows naturally from the subtree split in
`CLAUDE.md §12.3`: orchestration and eval each own broader acceptance
surfaces than runner + DSL. Lin's share is narrower by design, not by
under-loading. The assignment below keeps each owner inside their own
subtree so checkbox execution does not require coordination across
owners (see §5 Cross-owner dependencies for the unavoidable overlaps).

## Decoupling principle

Each owner can execute their checklist end-to-end **without editing
another owner's subtree**. When a checkbox inspects a cross-subtree
artifact (e.g. Phase 4 full loop runs every owner's code together), the
split assigns that checkbox to whoever owns the **observation** —
the runner-side producer tests are Lin's, the orchestration-side
invariants are Chen's, the verifier-side interpretation is Xie's.

---

## 1. Chen Jiahan — ~42 items

Owns: Phase 1 (infra) + Phase 3.3 – 3.4 (worktree & orchestration
invariants) + Phase 4.1 (surface_d5 live loop execution) + Phase 4.3 – 4.4
(LLM dispatch, loop invariants, containment) + partial Phase 4.5
(dirty-worktree detector) + Phase 5.5 (Pareto atomicity) + most of the
Phase 2 orchestration / subagent coverage.

### Phase 1 — Infrastructure (8)
- [ ] 1.1 – 1.2 clean install and dependency sweep
- [ ] 1.3 – 1.4 import sanity + ruff
- [ ] 1.5 – 1.6 unit-test sweep + coverage
- [ ] 1.7 – 1.10 CI conclusion + directory structure + EnvSpec round-trip

### Phase 2 — Orchestration + agent glue (8)
- [ ] 2.1.1 (agent schemas), 2.1.3 (interfaces.md cross-check)
- [ ] 2.3.3 (`test_worktree.py`), 2.3.4 (`test_reconcile.py`)
- [ ] 2.4.1 – 2.4.2 subagent prompt cross-checks (fork_graph / coder commit_message)
- [ ] 2.4.3 backend env-switch; machine_state CUDA fallback (from 2.4)

### Phase 3 — Worktree round + post-round invariants (8)
- [ ] 3.3.1 – 3.3.4 all worktree round items (`run-round` → branch + pointer + cleanup)
- [ ] 3.4.1 – 3.4.4 post-round invariants (append-only, writer boundaries, fork_graph, worktree reconciliation)

### Phase 4 — Live loop (surface_d5), dispatch, containment (14)
- [ ] 4.1.1 – 4.1.4 surface_d5 live 3-round loop execution + structural checks
- [ ] 4.3.1 – 4.3.4 LLM dispatch: prompt contents, fork_from evidence, machine_state tool use, compose-round status
- [ ] 4.4.1 – 4.4.4 resume/idempotency, reconcile trip, git-ignore checks, LLM containment diff
- [ ] 4.5.1 dirty-worktree precondition + dirty SHAs in manifest (the detector, not the writer)

### Phase 5 — Pareto atomicity (2)
- [ ] 5.5.1 – 5.5.2 Pareto written via tmp + atomic rename; mid-write kill leaves no truncated JSON

### Test Data (2)
- [ ] `demos/demo-1-surface-d5/expected_output/baseline_benchmark.json` (LER 0.01394 anchor)
- [ ] `autoqec/envs/builtin/surface_d5_depol.yaml`

### Success Criteria (8)
- [ ] Reproducibility (3 items: baseline LER ±5e-4 / pytest green / bootstrap CI)
- [ ] Correctness aggregate (2 items: surface_d5 3-round completion; pydantic round-trip)
- [ ] Performance (2 items: pytest < 90 s; install < 180 s)
- [ ] Safety & Containment atomicity (1 item: Pareto atomic-rename path)

### Open Questions (3)
- [ ] O-1 CI runner CPU vs self-hosted GPU
- [ ] O-2 Phase 4 wall-clock & token budgets (Chen owns the calibration dry-run because he drives the surface_d5 loop)
- [ ] O-4 advisor live vs recorded replay (Chen is the demo MC)

**Chen total: ~43**

---

## 2. Xie Jingu — ~42 items

Owns: All of Phase 5 (verifier + traps + statistical correctness + offline
replay) + Phase 4.2 (bb72 live loop execution) + Phase 2.2 OSD backend +
Phase 2.4 reward-hacking test + all reward-hacking + diagnostic fixtures.

### Phase 2 — Classical backend (OSD) + reward-hacking test (3)
- [ ] 2.2.2 OSD single-shot latency < 200 ms on bb72 DEM
- [ ] 2.2.3 (shared w/ Chen — Xie owns the `test_bposd_baseline.py` side) pytest on OSD baseline path
- [ ] 2.4.4 (the reward-hacking synthetic test item — `test_reward_hacking.py` must trip the safety layer)

### Phase 4 — bb72 full loop (4)
- [ ] 4.2.1 – 4.2.4 bb72 live 3-round loop execution + mirror of 4.1.2–4.1.6 structural checks + `classical_backend == "osd"` grep

### Phase 5 — Verifier, traps, diagnostics, stats, offline replay (18)
- [ ] 5.1.1 – 5.1.4 `autoqec verify` positive case → verification_report.{md,json}, VerifyReport parses, VERIFIED → pareto
- [ ] 5.2.1 – 5.2.4 Trap-A (seed leak) / Trap-B (paired-batch mismatch) / Trap-C (overfit memorizer) + `independent_eval.py` import-guard
- [ ] 5.3.1 – 5.3.4 `review-log` JSON keys + `diagnose` on OOM / NaN / degenerate + no-autofix contract
- [ ] 5.4.1 – 5.4.4 bootstrap-CI width, holdout isolation, paired byte-equal syndrome tensors, ablation-sanity random-weights
- [ ] 5.5.3 – 5.5.4 advisor offline replay: from tarball + SHA, no outbound HTTP

### Test Data (6)
- [ ] `tests/fixtures/reward_hacking/trap_{A,B,C}.pt` (three Phase-5.2 fixtures)
- [ ] `autoqec/envs/builtin/bb72_depol.yaml`
- [ ] Synthetic stuck-run fixture (Phase 5.3 — three consecutive |Δ_LER| < 0.002)
- [ ] Three diagnose-failure fixtures (OOM, NaN, p = 0)
- [ ] BP + OSD baseline (Phase 2.2 reference impl)
- [ ] Untrained-weights predecoder (ablation-sanity baseline for 5.4.4)

### Success Criteria (6)
- [ ] Research Integrity (4 items: three traps reject + independent_eval import guard + seed disjoint + ablation sanity)
- [ ] Performance OSD (1 item: OSD < 200 ms)
- [ ] Safety & Containment offline replay (1 item: offline verify succeeds)

### Open Questions (2)
- [ ] O-3 trap fixture deadline (Xie is the author)
- [ ] O-5 diagnose-failure fixture ownership

**Xie total: ~41**

---

## 3. Lin Tengxiang — ~22 items

Owns: `autoqec/decoders/` DSL + compiler + custom_fn validator +
`autoqec/runner/` train/eval + `cli/autoqec.py` glue + Tier-1 seed
templates. Phase 2.1 (DSL), Phase 2.3 (runner guard), Phase 3.1 – 3.2
(`run_quick` + `run --no-llm`), Phase 4.5 writer side
(`artifact_manifest.json`).

### Phase 2 — DSL + runner internals (8)
- [ ] 2.1.2 `parse_response` wraps `ValidationError` (Lin owns agents/schemas.py-side validation cross-check with Chen's author role)
- [ ] 2.2.1 MWPM single-shot latency < 50 ms
- [ ] 2.2.4 `scripts/benchmark_surface_baseline.py` diff < 5e-4 vs baseline_benchmark.json (Lin owns runner + backend adapter)
- [ ] 2.2.3-partial pytest on MWPM baseline (cross-owner with Chen — MWPM is Chen's Day-1 baseline but the adapter is Lin's decoders/backend_adapter.py)
- [ ] 2.3.1 – 2.3.2 `e2e_handshake.py` handshake + RoundMetrics field validation
- [ ] 2.3.3 `RunnerCallPathError` in-process guard
- [ ] 2.4.3 (runner safety test — `test_runner_safety.py` must reject training-set leakage at eval time; Lin owns `autoqec/runner/safety.py`)

### Phase 3 — No-LLM single round paths (8)
- [ ] 3.1.1 – 3.1.4 `scripts/run_quick.py` 1-round smoke: exit, artifacts, Δ_LER sanity
- [ ] 3.2.1 – 3.2.4 `python -m cli.autoqec run --no-llm` 1-round: `AUTOQEC_RESULT_JSON` capture, history.jsonl validation, log.md, candidate_pareto.json

### Phase 4 — Artifact manifest writer (2)
- [ ] 4.5.2 – 4.5.3 `artifact_manifest.json` writer (repo SHA, env / DSL SHA-256, versions, command line) + command-line-replay reproducibility

### Test Data (1)
- [ ] Tier-1 seed templates: `autoqec/example_db/gnn_{small,medium,gated}.yaml`, `neural_bp_{min,attn,per_check}.yaml`, `handshake_stub.yaml`

### Success Criteria (2)
- [ ] Performance MWPM (1 item: MWPM < 50 ms)
- [ ] Safety runner-safety layer triggers on NaN / wall-clock / training-set leak

### Open Questions (1)
- [ ] O-6 `autoqec cleanup-worktree` CLI wrapper — Lin owns CLI

**Lin total: ~22**

---

## 4. Rollup

| Owner | Checkboxes | Ratio |
|---|---|---|
| Chen | 43 | 2.0 |
| Xie | 41 | 1.9 |
| Lin | 22 | 1.0 |
| **Total** | **106** | — |

Ratio ≈ 2 : 2 : 1 as requested.

---

## 5. Cross-owner dependencies (unavoidable)

Three places where one owner cannot execute without another owner's code:

### 5.1 Phase 4 full loop (4.1 + 4.2)

Runs every owner's subtree together. Assignment splits by **environment**,
not by subtree: Chen executes `surface_d5_depol` (his env), Xie executes
`bb72_depol` (his env). Both need Lin's runner + DSL and each other's
classical baseline, but sign-off is single-owner per env. Unblock order:
Lin's runner must be stable → Chen lights up surface_d5 → Xie lights up
bb72 using the same orchestration glue.

### 5.2 Phase 4.5 artifact manifest (writer vs consumer)

Lin implements `artifact_manifest.json` in the runner (4.5.2 – 4.5.3).
Chen consumes it in the dirty-worktree detector (4.5.1). Lin's writer
lands first; Chen adds the detector check on top.

### 5.3 Phase 5.2 reward-hacking traps (fixtures vs harness)

Xie authors the three trap checkpoints (`trap_{A,B,C}.pt`). Lin's runner
must load them via the normal predecoder path for the traps to be tested
end-to-end. Contract: Xie hands Lin a DSL config that produces the trap;
Lin confirms the runner ingestion path works; Xie's verifier then rejects
them. No code changes in Lin's subtree once the ingestion path works.

---

## 6. Suggested sequencing

Because Chen and Xie each have ~42 items and Lin has ~22, the gating
path is **Lin's Phase 2.3 + Phase 3 must land first** so Chen and Xie
can then run Phase 4 in parallel. Suggested order:

1. **Day 3 morning (all three in parallel):**
   - Lin: Phase 2.1, 2.3, 3.1, 3.2 (runner + no-LLM CLI paths)
   - Chen: Phase 1 (infra sweep) + Phase 2.3/2.4 orchestration items + Phase 3.3/3.4 (worktree round)
   - Xie: Phase 2.2 OSD + Phase 5.1/5.2/5.4 (verifier + traps + statistical correctness) on a Phase-3 candidate from Lin's smoke run

2. **Day 3 afternoon (pending live-LLM wiring):**
   - Chen: Phase 4.1 surface_d5 live loop + Phase 4.3/4.4 dispatch/containment
   - Xie: Phase 4.2 bb72 live loop + Phase 5.3/5.5 (review-log / diagnose / offline replay)
   - Lin: Phase 4.5 artifact manifest writer + Phase 2 runner-safety final pass

3. **Before advisor walkthrough:**
   - Each owner confirms their Success Criteria rollup independently.
   - Open Questions closed by the respective owner (Chen O-1/O-2/O-4, Xie O-3/O-5, Lin O-6).
   - Joint 30-min dry run on Phase 1 + Phase 3 + one of Phase 4.1 / 4.2 to catch integration surprises.

If Phase 4 live wiring does not land in time, step (2) collapses: Chen's
Phase 4 checklist is marked `blocked` per the plan's prerequisite rule,
Xie's Phase 4.2 is blocked identically, and the walkthrough narrative
shifts to Phase 3 candidates. Lin is unaffected — his items do not
depend on live LLMs.

---

## 7. Ownership change protocol

If a checkbox needs to move between owners after this assignment is
committed:

- Open a comment on the GitHub issue that tracks this plan
- Note the checkbox ID (e.g. `4.3.2`), the reason, and the new owner
- The new owner edits this file to reflect the move
- Do not silently move items — the 2 : 2 : 1 target is a shared accounting rule
