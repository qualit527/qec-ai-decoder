# Test Plan Ownership

**Plan:** [`docs/verification/human-verification-test-plan.md`](./human-verification-test-plan.md)
**Design rationale:** [`docs/superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md`](../superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md)

## Decoupling principle

Each owner executes their checklist end-to-end without editing another owner's subtree. When a checkbox inspects a cross-subtree artifact (e.g. Phase 4 full-loop runs every owner's code together), the split assigns that checkbox to whoever owns the observation side — the runner-side producer checks are Lin's, the orchestration-side invariants are Chen's, the verifier-side interpretation is Xie's. The three unavoidable overlaps are enumerated in §5.

## Owner summary

| Owner | Primary subtree |
|---|---|
| Chen Jiahan | `autoqec/orchestration/`, `autoqec/agents/`, `autoqec/tools/`, surface_d5 env + MWPM baseline, Phase 1 infra |
| Lin Tengxiang | `autoqec/decoders/` (DSL + modules), `autoqec/runner/`, `cli/`, `autoqec/example_db/`, bb72 Demo 2 templates |
| Xie Jingu | `autoqec/eval/`, `autoqec/decoders/baselines/bposd*`, reward-hacking fixtures, statistical correctness, offline replay |

---

## 1. Chen Jiahan

Phase 1 infrastructure, orchestration-side invariants inside Phase 2 and Phase 3, surface_d5 live loop, LLM dispatch and containment, Pareto atomicity.

### Phase 1 — Infrastructure
- [ ] 1.1 – 1.2 clean install and dependency sweep
- [ ] 1.3 – 1.4 import sanity and `ruff`
- [ ] 1.5 – 1.6 unit-test sweep and coverage
- [ ] 1.7 – 1.10 CI conclusion, directory structure, and `EnvSpec` round-trip

### Phase 2 — Orchestration and agent glue
- [ ] 2.1.1 agent-schema pytest sweep
- [ ] 2.1.3 `interfaces.md` pydantic-signature cross-check
- [ ] 2.3.3 `test_worktree.py` path coverage
- [ ] 2.3.4 `test_reconcile.py` orphan / branch-deleted / both-missing
- [ ] 2.4.1 `build_prompt('ideator', …)` fork_graph presence and `last_5_hypotheses` regression guard
- [ ] 2.4.2 `AUTOQEC_IDEATOR_BACKEND` switch
- [ ] 2.4.4 `tools.machine_state` CUDA-unavailable fallback

### Phase 4 — surface_d5 live loop, dispatch, containment
- [ ] 4.1.1 – 4.1.4 surface_d5 3-round live loop execution and structural checks (`history.jsonl`, `pareto.json`, `fork_graph.json`, branch conventional-commit prefix)
- [ ] 4.3.1 – 4.3.4 LLM dispatch: prompt contents, non-trivial `fork_from` evidence, `machine_state` tool use, compose-round status
- [ ] 4.4.1 – 4.4.4 resume / idempotency, reconcile trip on manual branch delete, git-ignore of `.worktrees/` and `runs/`, LLM containment diff

### Phase 5 — Pareto atomicity
- [ ] 5.5.1 – 5.5.2 `pareto.json` written via tmp-file + atomic rename; mid-write kill leaves no truncated JSON

### Test data
- [ ] `demos/demo-1-surface-d5/expected_output/baseline_benchmark.json` (LER 0.01394 anchor)
- [ ] `autoqec/envs/builtin/surface_d5_depol.yaml`

### Success criteria
- [ ] Reproducibility — baseline LER reproduction within ±5 × 10⁻⁴; `pytest -m "not integration"` green with ≥ 33 tests in < 90 s; bootstrap 95% CI on `delta_ler_holdout` does not cross 0 for VERIFIED entries
- [ ] Correctness — surface_d5 3-round completion; `pareto.json` non-dominated invariant; all 7 pydantic schemas round-trip
- [ ] Safety — Pareto atomic-rename path; no experiment branch introduces diffs under `runs/` / `tests/` / `autoqec/envs/builtin/` / `autoqec/eval/` / `docs/contracts/`

### Open questions
- [ ] O-1 CI runner (CPU vs self-hosted GPU)
- [ ] O-2 Phase 4 wall-clock and token budgets — owner of the calibration dry-run since surface_d5 is Chen's env
- [ ] O-4 advisor live run vs recorded replay — Chen is the demo MC

---

## 2. Lin Tengxiang

Runner + DSL + CLI surface, no-LLM single-round paths, worktree-round execution, bb72 Demo 2 live loop, artifact-manifest writer, `review-log` and `diagnose` CLI surfaces.

### Phase 2 — DSL compiler, backend adapter, runner internals
- [ ] 2.1.2 `parse_response` wraps pydantic `ValidationError` as `ValueError` for a truly required field (`rationale`, not the default-valued `fork_from`)
- [ ] 2.2.1 MWPM single-shot latency < 50 ms (median of 100 runs)
- [ ] 2.2.3 `pytest tests/test_backend_adapter.py tests/test_pymatching_baseline.py tests/test_bposd_baseline.py -q` all green
- [ ] 2.2.4 `scripts/benchmark_surface_baseline.py` emits LER within [0.01344, 0.01444] and absolute diff vs `baseline_benchmark.json` < 5 × 10⁻⁴
- [ ] 2.3.1 `scripts/e2e_handshake.py` produces `checkpoint.pt` + `metrics.json` + `train.log` with ≥ 100 training steps
- [ ] 2.3.2 `run_round(cfg_with_code_cwd=X)` in-process call raises `RunnerCallPathError`
- [ ] 2.4.3 `test_reward_hacking.py` synthetic memorizer predecoder trips `autoqec/runner/safety.py`

### Phase 3 — No-LLM single-round paths and worktree round
- [ ] 3.1.1 – 3.1.4 `scripts/run_quick.py --rounds 1 --profile dev` smoke: exit + 15-min budget + GPU VRAM ceiling + `round_1/` artifacts + `delta_ler` sanity
- [ ] 3.2.1 – 3.2.4 `python -m cli.autoqec run <env> --rounds 1 --profile dev --no-llm`: exit, `AUTOQEC_RESULT_JSON.run_dir` capture, append-only `history.jsonl`, `candidate_pareto.json` non-dominated
- [ ] 3.3.1 – 3.3.4 worktree round via `run-round --code-cwd … --branch … --fork-from baseline --round-attempt-id …`: exit, conventional-commit prefix, `metrics.json.branch` / `commit_sha`, `round_1_pointer.json` round-trip, `cleanup_round_worktree(...)` removes directory while retaining branch
- [ ] 3.4.1 – 3.4.4 post-round invariants: `history.jsonl` strictly grows; runner stays inside `round_<N>/` while orchestrator stays at run-root (mtime snapshot); `fork_graph.json` has `round_1.parent == "baseline"`; `git worktree list` reconciles with `RunMemory`

### Phase 4 — bb72 live loop, artifact manifest
- [ ] 4.2.1 – 4.2.4 bb72 3-round live loop execution with 120-min budget, mirrors 4.1.2 – 4.1.6 structural checks, and `round_1/config.yaml` records `classical_backend == "osd"`
- [ ] 4.5.1 dirty-worktree precondition warning path wired from runner: dirty-file SHAs recorded into the per-round manifest
- [ ] 4.5.2 – 4.5.3 `artifact_manifest.json` writer (repo SHA, env YAML SHA-256, DSL SHA-256, Python / torch / CUDA / stim / pymatching / ldpc versions, full command line); re-run from the recorded command line on the same repo SHA reproduces `delta_ler` within 2× the 95% CI half-width

### Phase 5 — `review-log` and `diagnose` CLI surfaces
- [ ] 5.3.1 `python -m cli.autoqec review-log <run_dir>` emits a JSON blob containing `n_rounds`, `n_pareto`, `n_killed_by_safety`, `mean_wallclock_s`, `top_hypotheses`
- [ ] 5.3.2 `n_rounds` matches `wc -l RUN_DIR/history.jsonl`
- [ ] 5.3.3 `python -m cli.autoqec diagnose <run_dir>` exits 0 and identifies the injected OOM / NaN / degenerate-p failure signatures
- [ ] 5.3.4 `diagnose` output never autonomously applies a fix (skill contract)

### Test data
- [ ] Tier-1 seed templates in `autoqec/example_db/`: `gnn_{small,medium,gated}.yaml`, `neural_bp_{min,attn,per_check}.yaml`, `handshake_stub.yaml`
- [ ] `autoqec/envs/builtin/bb72_depol.yaml`
- [ ] Three diagnose-failure fixtures (OOM, NaN loss, degenerate p = 0)

### Success criteria
- [ ] Correctness — bb72 3-round completion (once live LLM DAG lands)
- [ ] Performance — `install -e '.[dev]'` wall-clock < 180 s; MWPM < 50 ms
- [ ] Safety — `autoqec/runner/safety.py` triggers `RunnerCallPathError` or `killed_by_safety` on NaN loss / wall-clock budget / training-set leak at eval time

### Open questions
- [ ] O-5 diagnose-failure fixture ownership and scheduling
- [ ] O-6 `autoqec cleanup-worktree` CLI wrapper (or keep Python-API-only)

---

## 3. Xie Jingu

Verifier, reward-hacking traps, statistical correctness, offline advisor replay, OSD-backend latency.

### Phase 2 — OSD backend
- [ ] 2.2.2 OSD single-shot latency < 200 ms on bb72 DEM

### Phase 5 — Verification, traps, statistics, replay
- [ ] 5.1.1 – 5.1.4 `autoqec verify` positive case: exits 0, writes `verification_report.md` + `verification_report.json`, verdict in {VERIFIED, SUSPICIOUS, FAILED} with bootstrap 95% CI, VERIFIED → `pareto.json` entry, SUSPICIOUS / FAILED → no entry
- [ ] 5.2.1 – 5.2.4 Trap-A (training-seed leak), Trap-B (paired-batch mismatch), Trap-C (overfit memorizer) each receive verdict ≠ VERIFIED; `independent_eval.py` imports no symbol from `autoqec.runner.*`
- [ ] 5.4.1 – 5.4.4 bootstrap-CI width matches analytic 2 × 1.96 × √(p(1−p)/N); training and holdout seed sets disjoint; paired byte-equal syndrome tensors under one `paired_eval_bundle_id`; random-weights predecoder `delta_ler` 95% CI crosses 0
- [ ] 5.5.3 – 5.5.4 advisor offline replay from `runs/<id>.tar.gz` + repo SHA with `AUTOQEC_*_BACKEND` unset; no outbound HTTP during replay

### Test data
- [ ] `tests/fixtures/reward_hacking/trap_{A,B,C}.pt`
- [ ] Synthetic stuck-run fixture for Phase 5.3 (three consecutive `|delta_ler| < 0.002` rounds)
- [ ] BP + OSD baseline as the bb72 reference decoder
- [ ] Untrained-weights predecoder as the ablation-sanity baseline for 5.4.4

### Success criteria
- [ ] Research integrity — all three reward-hacking traps receive verdict ≠ VERIFIED; `independent_eval.py` has zero imports from `autoqec.runner.*`; training and holdout seeds disjoint per env YAML; ablation-sanity CI crosses 0
- [ ] Performance — OSD single-shot latency < 200 ms
- [ ] Safety — offline advisor replay succeeds without network access

### Open questions
- [ ] O-3 trap fixture construction deadline (Xie is the author)

---

## 4. Phase 4 execution split

Phase 4 full-loop execution runs every owner's subtree together. Split by environment rather than by subtree:

- Chen executes `surface_d5_depol` (his env).
- Lin executes `bb72_depol` (his Demo 2 deliverable; bb72 seed templates are already in Lin's scope per `CLAUDE.md §12.1`).
- Xie consumes both runs via the verifier and statistical-correctness checks.

Prerequisite: the live Ideator → Coder → Analyst DAG in `cli.autoqec run` must land before any Phase 4 checkbox is executable. If unlanded by demo day, all three owners' Phase 4 items are marked `blocked` (not `FAIL`) per the plan's prerequisite rule, and the walkthrough falls back to Phase 3 candidates.

## 5. Cross-owner dependencies

Three places where one owner cannot execute without another owner's code landing first.

### 5.1 Phase 4 full loops

Chen runs surface_d5; Lin runs bb72. Both runs depend on Lin's runner + DSL + CLI staying stable and on Chen's orchestration + agents producing a well-formed live loop. Xie observes the outputs of both runs via the verifier. Unblock order: Lin's runner lands → Chen's orchestration lights up → Chen runs surface_d5, Lin runs bb72 → Xie verifies both.

### 5.2 Phase 4.5 artifact manifest

Lin implements `artifact_manifest.json` inside the runner (4.5.2 and 4.5.3). Chen consumes it in the dirty-worktree detector (4.5.1). Lin's writer lands first; Chen layers the detector check on top.

### 5.3 Phase 5.2 reward-hacking traps

Xie authors the three trap checkpoints in `tests/fixtures/reward_hacking/`. Lin's runner must load them through the normal predecoder path for the traps to be tested end-to-end. Contract: Xie hands Lin a DSL config that produces each trap; Lin confirms the runner ingestion path works; Xie's verifier then rejects them. Once the ingestion path works, no further changes in Lin's subtree are required.

## 6. Suggested Day-3 sequencing

**Morning — parallel bring-up.** Lin brings up Phase 2.1 / 2.3 and the two no-LLM paths in Phase 3.1 / 3.2 so that a candidate checkpoint exists by midday. Chen runs Phase 1 infra sweep and the Phase 2 orchestration items (worktree, reconcile, subagent prompts, machine_state). Xie sets up Phase 2.2 OSD latency and starts Phase 5.1 / 5.2 / 5.4 against the no-LLM candidate from Lin.

**Afternoon — live loop (gated on LLM-wiring prerequisite).** Chen runs the surface_d5 live loop and Phase 4.3 / 4.4 dispatch and containment checks. Lin runs the bb72 live loop and layers Phase 4.5 artifact-manifest writer + Phase 5.3 `review-log` / `diagnose` CLI surfaces on top. Xie runs Phase 5.3 and 5.5.3 / 5.5.4 offline replay against a live-loop candidate.

**Before advisor walkthrough.** Each owner signs off their own Success Criteria independently. Open Questions are closed by their respective owners. A joint 30-minute dry run on Phase 1 + Phase 3 + one Phase 4 full loop catches integration surprises. If Phase 4 live wiring has not landed in time, Chen's and Lin's Phase 4 items are `blocked` per the prerequisite rule, and the demo narrative shifts to Phase 3 candidates; Xie's Phase 5 items still run against those candidates.

## 7. Ownership change protocol

If a checkbox needs to move between owners after this assignment is committed, open a comment on the GitHub issue that tracks this plan noting the checkbox ID (e.g. `4.3.2`), the reason, and the new owner. The new owner edits this file to reflect the move. Do not silently move items — the split is a shared accounting rule.
