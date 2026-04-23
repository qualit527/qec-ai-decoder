# Human Verification Test Plan for AutoQEC Research Harness

## Overview

This document tracks the human verification test plan for the AutoQEC
research harness (qec-ai-decoder). The goal is to validate that, on a cold
checkout, the harness reproduces its claimed Δ LER numbers and that the
numbers are resistant to reward hacking. This is the acceptance gate
between Day-3 sprint close and the advisor walkthrough.

For the full design rationale behind every gate, see
[docs/superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md](../superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md).

## Test Objectives

1. **Reproducibility**: Clone → install → single command reproduces the surface_d5 baseline LER within bootstrap CI
2. **Correctness**: Reference environments (`surface_d5_depol`, `bb72_depol`) complete a research loop with structurally valid `history.jsonl` / `pareto.json`
3. **Research integrity**: `independent_eval.py` fair-baseline guards (holdout isolation, paired eval, tool whitelist) trip on deliberately malicious checkpoints
4. **Observability**: `verify`, `review-log`, `diagnose` CLI subcommands each emit structured output over a real run directory
5. **Contract hygiene**: Every pydantic schema in `docs/contracts/interfaces.md` round-trips; `pytest -m "not integration"` is fully green

## Test Plan

### Phase 1: Infrastructure & Cold-Start Reproducibility

#### Test 1.1: Clean Install
- [ ] `git clone` succeeds with no missing submodules
- [ ] `pip install -e '.[dev]'` completes in under 180 s
- [ ] `python -c "import autoqec"` prints a non-empty version string
- [ ] `ruff check autoqec cli tests scripts` passes with zero warnings

#### Test 1.2: Baseline Tests & CI
- [ ] `pytest tests/ -m "not integration" -q` is all green (≥ 33 tests, < 90 s)
- [ ] Line coverage ≥ 70% (`pytest --cov=autoqec`)
- [ ] Most recent `main` CI run concluded `success`
- [ ] Both builtin env YAMLs load via `EnvSpec.model_validate`

### Phase 2: Component-Level Isolation

#### Test 2.1: Schemas & DSL
- [ ] All 7 pydantic models in `docs/contracts/interfaces.md` round-trip against code
- [ ] Tier-1 GNN + Neural-BP seed configs compile and run forward without NaN
- [ ] Tier-2 hostile `custom_fn` (contains `import os` / `eval(`) is rejected by the AST validator
- [ ] `pytest tests/test_dsl_*.py tests/test_*_module.py tests/test_custom_fn_validator.py` is all green

#### Test 2.2: Classical Backends
- [ ] MWPM single-shot latency on surface_d5 DEM < 50 ms (median of 100 runs)
- [ ] OSD single-shot latency on bb72 DEM < 200 ms
- [ ] `scripts/benchmark_surface_baseline.py` reports LER in [0.01344, 0.01444] at p = 5 × 10⁻³, 1 M shots, seed 42
- [ ] Absolute diff vs `demos/demo-1-surface-d5/expected_output/baseline_benchmark.json` < 5 × 10⁻⁴

#### Test 2.3: Runner & Orchestration
- [ ] `scripts/e2e_handshake.py` produces a valid `checkpoint.pt` + `metrics.json` + `train.log` with ≥ 100 training steps
- [ ] `run_round(cfg_with_code_cwd=X)` called in-process raises `RunnerCallPathError` (physical guard against cwd-cache bug)
- [ ] `tests/test_worktree.py` covers create / compose_pure / compose_conflict / cleanup paths
- [ ] `tests/test_reconcile.py` covers orphan-branch recovery with preserved `round_attempt_id`

#### Test 2.4: Subagents & Safety
- [ ] `build_prompt('ideator', ctx)` contains `fork_graph` and does NOT contain legacy `last_5_hypotheses`
- [ ] `AUTOQEC_IDEATOR_BACKEND=claude-cli` actually switches backend (env-dispatch test)
- [ ] `test_reward_hacking.py` synthetic memorizer predecoder triggers the safety layer
- [ ] `tools.machine_state` returns `{}` (not exception) when CUDA is unavailable

### Phase 3: Single Research Round (No-LLM Path)

#### Test 3.1: `run_quick` Smoke
- [ ] `scripts/run_quick.py --rounds 1 --profile dev` exits 0 within 15 minutes
- [ ] GPU VRAM peak during training stays under 80% of available
- [ ] Captured `RUN_DIR/round_1/` contains `config.yaml`, `train.log`, `checkpoint.pt`, `metrics.json`
- [ ] `metrics.json.delta_ler` within [-0.02, 0.02] under fixed seed

#### Test 3.2: CLI `run --no-llm`
- [ ] `python -m cli.autoqec run <env.yaml> --rounds 1 --profile dev --no-llm` exits 0
- [ ] Authoritative `run_dir` captured from `AUTOQEC_RESULT_JSON=...` stdout line
- [ ] `RUN_DIR/history.jsonl` is append-only, one-line-per-round JSONL, each row parses as `RoundMetrics`
- [ ] `RUN_DIR/candidate_pareto.json` exists and follows the non-dominated-set invariant (not a top-5 truncation)

#### Test 3.3: Worktree Round
- [ ] `python -m cli.autoqec run-round <env> <cfg> <round_dir> --code-cwd <wt> --branch exp/test/1-smoke --fork-from baseline --round-attempt-id <uuid>` exits 0
- [ ] Branch tip commit message matches `^(feat|fix|test|docs|chore|refactor|perf|ci)` (Coder commit_message contract)
- [ ] `metrics.json.branch` and `commit_sha` are populated; `round_1_pointer.json` round-trips via `git show`
- [ ] `cleanup_round_worktree(...)` removes the directory but retains the branch (branches-as-Pareto)

#### Test 3.4: Post-Round Invariants
- [ ] `history.jsonl` line count strictly increases across rounds; no deletions
- [ ] Runner writes only inside `round_<N>/`; orchestrator writes only at run-root (verified by mtime snapshot)
- [ ] `fork_graph.json` has `round_1.parent == "baseline"`
- [ ] `git worktree list` reconciles with `RunMemory`'s active-worktree count

### Phase 4: Full Research Loop (Live LLM)

**Prerequisite:** the live Ideator → Coder → Analyst DAG must be wired
into `cli.autoqec run` (currently the command raises unless `--no-llm`).
If unlanded by demo day, Phase 4 is blocked and the walkthrough falls
back to a Phase 3 candidate.

#### Test 4.1: surface_d5 Full Loop
- [ ] `python -m cli.autoqec run <surface_d5_depol.yaml> --rounds 3 --profile dev` exits 0
- [ ] `history.jsonl` has exactly 3 rows; every `round_attempt_id` is unique
- [ ] `pareto.json` obeys the non-dominated invariant on `(delta_ler, flops_per_syndrome, n_params)`
- [ ] `fork_graph.json` has ≥ 4 nodes and at least one round with `parent != "baseline"`

#### Test 4.2: bb72 Full Loop
- [ ] Same CLI run on `bb72_depol.yaml` exits 0
- [ ] All 4.1 structural checks hold
- [ ] `round_1/config.yaml` records `classical_backend == "osd"`
- [ ] Branch tips all conform to the conventional-commit prefix regex

#### Test 4.3: LLM Dispatch & Tool Use
- [ ] Ideator prompts contain `fork_graph`, `machine_state`, `pareto_front` keys; do NOT contain `last_5_hypotheses`
- [ ] At least one round has non-trivial `fork_from` (string ≠ `"baseline"` or a list for compose); otherwise a warning block appears in `log.md`
- [ ] `machine_state` tool is recorded as called ≥ 1 time across 3 rounds
- [ ] Any compose round's `metrics.json.status` ∈ {`ok`, `compose_conflict`, `killed_by_safety`}

#### Test 4.4: Loop Invariants & Containment
- [ ] Ctrl-C during round 2, then re-run — resumes at round 3 without redoing rounds 1-2
- [ ] Manually `git branch -D` an experiment branch — next startup writes an idempotent `branch_manually_deleted` synthetic row
- [ ] `.worktrees/` and `runs/` are git-ignored (`git ls-files` empty under both)
- [ ] No experiment branch has diffs under `runs/`, `tests/`, `autoqec/envs/builtin/`, `autoqec/eval/`, `docs/contracts/` (LLM containment)

#### Test 4.5: Dirty-Worktree & Artifact Manifest
- [ ] Starting a run with uncommitted changes in the base checkout emits a warning and records dirty-file SHAs in the per-round manifest
- [ ] Every `round_N/` contains an `artifact_manifest.json` with repo SHA, branch, env YAML SHA-256, DSL SHA-256, Python / torch / CUDA / stim / pymatching / ldpc versions, full command line
- [ ] Re-running from the recorded command line on the same repo SHA reproduces `delta_ler` within 2× the 95% CI half-width

### Phase 5: Verification & Diagnostic Layer

#### Test 5.1: `autoqec verify` Positive Case
- [ ] `python -m cli.autoqec verify <round_dir> --env <yaml> --n-seeds 50` exits 0
- [ ] Writes both `verification_report.md` and `verification_report.json` into the round dir
- [ ] `verdict` ∈ {`VERIFIED`, `SUSPICIOUS`, `FAILED`}; `delta_ler_holdout` carries a bootstrap 95% CI
- [ ] VERIFIED → round appears in `pareto.json`; SUSPICIOUS / FAILED → does not (prerequisite: verify ↔ pareto wiring)

#### Test 5.2: Reward-Hacking Traps
- [ ] Trap-A (training-seed leak): `verify` returns verdict ≠ `VERIFIED`; report mentions `holdout` + (`mismatch` | `leak` | `isolation`)
- [ ] Trap-B (paired batch mismatch): `verify` returns verdict ≠ `VERIFIED`; `paired_eval_bundle_id` must be byte-equal across the two compared evaluations
- [ ] Trap-C (overfit 100-shot memorizer): 200 K holdout evaluation → verdict `FAILED` OR 95% CI of `delta_ler_holdout` crosses 0
- [ ] `independent_eval.py` imports no symbol from `autoqec.runner.*` (§10 CI guard)

#### Test 5.3: `review-log` & `diagnose`
- [ ] `python -m cli.autoqec review-log <run_dir>` emits a JSON blob with `n_rounds`, `n_pareto`, `n_killed_by_safety`, `mean_wallclock_s`, `top_hypotheses`
- [ ] `n_rounds` matches `wc -l RUN_DIR/history.jsonl`
- [ ] `python -m cli.autoqec diagnose <run_dir>` exits 0 and identifies injected failure modes (OOM, NaN, p = 0 degenerate)
- [ ] Diagnose output never autonomously applies a fix (skill contract)

#### Test 5.4: Statistical Correctness
- [ ] Bootstrap CI width at LER = 0.01, N = 200 K, 1000 resamples is < 0.002 (matches analytic 2 × 1.96 × √(p(1-p)/N))
- [ ] Training and holdout seed sets are disjoint (asserted in code via `seed_policy`)
- [ ] Paired eval: syndrome tensors for plain-classical vs predecoder+classical are byte-hash equal under the same `paired_eval_bundle_id`
- [ ] Ablation sanity: random-weights predecoder has `delta_ler` 95% CI crossing 0 (protocol has no positive bias)

#### Test 5.5: Pareto Atomicity & Offline Replay
- [ ] `pareto.json` is written via tmp-file + atomic rename (grep for `os.replace` in `memory.py`)
- [ ] Killing the Pareto writer mid-update leaves either the previous file or a detectable tmp — never a truncated JSON
- [ ] Advisor can replay a packaged run from `runs/<id>.tar.gz` + repo SHA offline: `verify` command succeeds with all `AUTOQEC_*_BACKEND` env vars unset
- [ ] No outbound HTTP during replay (verified via `strace` / network sandbox during smoke test)

## Test Data

### Required Test Vectors
- [ ] `demos/demo-1-surface-d5/expected_output/baseline_benchmark.json` (LER anchor 0.01394)
- [ ] `tests/fixtures/reward_hacking/trap_{A,B,C}.pt` (Phase 5.2 reward-hack fixtures)
- [ ] Tier-1 seeds: `autoqec/example_db/gnn_{small,medium,gated}.yaml`, `neural_bp_{min,attn,per_check}.yaml`, `handshake_stub.yaml`
- [ ] Two reference envs: `autoqec/envs/builtin/{surface_d5_depol,bb72_depol}.yaml`
- [ ] Synthetic stuck-run fixture (three consecutive |Δ_LER| < 0.002 rounds for Phase 5.3)
- [ ] Three diagnose-failure fixtures (OOM / NaN loss / degenerate p = 0)

### Reference Implementations
- [ ] PyMatching (MWPM) as the surface_d5 baseline decoder
- [ ] BP + OSD (via `ldpc`) as the bb72 qLDPC baseline decoder
- [ ] Untrained-weights predecoder as the ablation-sanity baseline

## Success Criteria

### Reproducibility
- [ ] Surface_d5 baseline LER reproduction within ±5 × 10⁻⁴ of the 0.01394 anchor
- [ ] `pytest -m "not integration"` fully green on a clean checkout, ≥ 33 tests, < 90 s
- [ ] Bootstrap 95% CI on `delta_ler_holdout` does NOT cross 0 for every VERIFIED Pareto entry

### Correctness
- [ ] Both `surface_d5_depol` and `bb72_depol` complete 3-round research loops (once live LLM DAG lands; until then, 1-round no-LLM equivalent)
- [ ] `pareto.json` obeys the non-dominated-set invariant (not top-5 truncation) over `(delta_ler, flops_per_syndrome, n_params)`
- [ ] All 7 pydantic schemas (EnvSpec / RunnerConfig / RoundMetrics / VerifyReport / IdeatorResponse / CoderResponse / AnalystResponse) round-trip against the code

### Research Integrity
- [ ] All three reward-hacking traps (training-seed leak, paired batch mismatch, overfit memorizer) receive verdict ≠ `VERIFIED`
- [ ] `independent_eval.py` has zero imports from `autoqec.runner.*`
- [ ] Training and holdout seed sets are disjoint per env YAML
- [ ] Ablation sanity: random-weights predecoder `delta_ler` 95% CI crosses 0

### Performance
- [ ] MWPM single-shot latency < 50 ms (median of 100 runs)
- [ ] OSD single-shot latency < 200 ms
- [ ] `pytest -m "not integration"` wall-clock < 90 s
- [ ] `install -e '.[dev]'` wall-clock < 180 s

### Safety & Containment
- [ ] Pareto writes are atomic (tmp-file + rename) with no truncation failure path
- [ ] No experiment branch introduces diffs under `runs/`, `tests/`, `autoqec/envs/builtin/`, `autoqec/eval/`, or `docs/contracts/`
- [ ] Safety layer triggers `RunnerCallPathError` or `killed_by_safety` on NaN loss / wall-clock budget / training-set leak at eval time
- [ ] Offline advisor replay succeeds without network access

## Open Questions

1. Will Phase 4 live-LLM wall-clock budgets (75 min surface_d5, 120 min bb72) hold in practice? Requires a calibration dry-run.
2. Will per-env token budgets (500 K input / 80 K output per 3 rounds) match the billed usage of the chosen Codex + Claude backends?
3. Who constructs the three Phase-5.2 reward-hacking fixtures, and by when?
4. Will the advisor walkthrough be a live re-run or a recorded replay? (Changes the Phase 1 wall-clock ceiling for the advisor.)
5. Does the `diagnose` skill write a `diagnosis.md` by Day 3, or does the JSON-blob contract remain the gate?
6. Is an `autoqec cleanup-worktree` CLI wrapper required, or is the `cleanup_round_worktree` Python API sufficient?

## Notes

- Every checkbox has a quantitative pass / fail rule; no "within expected range" language.
- All tests that can be automated live under `pytest tests/` and are invoked by the gates above. Any deviation from expected results must be filed as a GitHub issue labelled `blocks-demo` (hard) or `quarantine` (soft).
- Phases marked with a **Prerequisite** note (Phase 4 and parts of Phase 5) depend on features not yet merged as of this plan; skipping their checkboxes is recorded as `blocked`, not `quarantine`, and does not count against the per-phase gate.
- Performance profiling artifacts (GPU VRAM peaks, round wall-clocks, token usage) must be archived in the run directory, not just printed to stdout.
- Save all synthesis-equivalent artifacts (`metrics.json`, `history.jsonl`, `pareto.json`, `fork_graph.json`, `verification_report.*`, `artifact_manifest.json`) alongside the run for reproducibility audit.

---

**References:**

- [AutoQEC Design v2.3](../superpowers/specs/2026-04-20-autoqec-design.md)
- [Test Plan Design Rationale (v2, Codex-reviewed)](../superpowers/specs/2026-04-23-demo-acceptance-test-plan-design.md)
- [Contracts: interfaces.md](../contracts/interfaces.md) and [round_dir_layout.md](../contracts/round_dir_layout.md)
- [CI Coverage Design](../superpowers/specs/2026-04-22-ci-coverage-design.md)
- Structural inspiration: [QuAIR/0420-FPGA-Decoder human-verification-test-plan](https://github.com/QuAIR/0420-FPGA-Decoder/blob/main/docs/verification/human-verification-test-plan.md)

---

**To create this as a GitHub issue:**

```bash
gh issue create \
  --title "Human Verification Test Plan for AutoQEC Research Harness" \
  --body-file docs/verification/human-verification-test-plan.md
```
