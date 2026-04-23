# Demo-Acceptance Test Plan — qec-ai-decoder

**Status:** design, pending owner review
**Authors:** Chen Jiahan (drafting), Xie Jingu (Phase 5 owner), Lin Tengxiang (Phase 2/3 owner)
**Date:** 2026-04-23
**Gate:** whether Day-3 sprint output can be handed to the advisor

## 1. Overview

This test plan is the **human-verification gate** between Day-3 sprint close
and the advisor walkthrough. It does not replace the engineering CI suite
(see `docs/superpowers/specs/2026-04-22-ci-coverage-design.md`). Instead it
answers a single question: on a cold checkout, can someone reproduce the
Δ_LER numbers the team is claiming and trust them?

### 1.1 Objectives

1. **Reproducibility** — clone → install → single command reproduces the
   baseline LER within bootstrap CI.
2. **Correctness** — both reference envs (`surface_d5_depol`, `bb72_depol`)
   complete a live research loop; `history.jsonl` / `pareto.json` are
   structurally valid.
3. **Research integrity** — `independent_eval.py`'s three fair-baseline
   guards (holdout isolation, paired eval, tool whitelist) can be tripped
   by deliberately malicious checkpoints.
4. **Observability** — `/review-log`, `/diagnose-failure`, `/verify-decoder`
   each produce human-readable output on real run directories.
5. **Contract hygiene** — every pydantic schema in `docs/contracts/interfaces.md`
   round-trips; `pytest -m "not integration"` is fully green; the GitHub
   Actions workflow on `main` is green.

### 1.2 Scope

- In scope: the five architectural layers below.
- Out of scope: cross-advisor acceptance (a different document), long-term
  regression / release gating (`ci-coverage-design.md`), research novelty
  evaluation (advisor's own judgement).

### 1.3 Structure

Five phases, each with inline hard-number thresholds in every checkbox. No
separate "Success Criteria" section; every checkbox is its own gate. This
mirrors the QuAIR FPGA decoder test plan (GitHub issue #3) but tightens
the quantification of pass/fail, which was the main weakness of that
reference.

### 1.4 How to execute

Each phase is run top-to-bottom. A FAIL must be filed as a GitHub issue
labelled `blocks-demo` (hard gate) or `quarantine` (soft gate) before the
phase is considered complete. Quarantine issues must carry an owner and a
deadline; demo can proceed only if the per-phase Gate rule admits the
quarantine count.

---

## 2. Phase 1 — Infrastructure / Cold-Start Reproducibility

**Purpose:** validate that a clean machine can get from zero to the first
runnable command. Any FAIL here invalidates every downstream phase.

**Operator:** advisor or new collaborator. **Environment:** clean
Linux/Windows, Python 3.12, no pre-existing `.venv/`.

```
[ ] 1.1 git clone <repo_url> && cd qec-ai-decoder
        # exit 0; no missing submodule
[ ] 1.2 python -m venv .venv && ./.venv/bin/pip install -e '.[dev]'
        # exit 0; install wall-clock < 180s;
        # torch/stim/pymatching/ldpc/pydantic versions printed
[ ] 1.3 ./.venv/bin/python -c "import autoqec; print(autoqec.__version__)"
        # prints non-empty string, no ImportError
[ ] 1.4 ./.venv/bin/ruff check autoqec cli tests scripts
        # exit 0; zero warnings
[ ] 1.5 ./.venv/bin/pytest tests/ -m "not integration" -q
        # all green; test count >= 33; wall-clock < 90s
[ ] 1.6 ./.venv/bin/pytest tests/ -m "not integration" --cov=autoqec --cov-report=term
        # line coverage >= 70% (aligns with ci-coverage-design.md)
[ ] 1.7 most recent main-branch CI run is green
        # gh run list --branch main --limit 1 -> conclusion=success
[ ] 1.8 directory structure matches spec §10:
        autoqec/{envs,agents,decoders,runner,eval,orchestration,tools}/,
        cli/, circuits/, demos/, tests/, docs/{contracts,superpowers}/
[ ] 1.9 docs/contracts/{interfaces.md,round_dir_layout.md} last commit has
        `contract-change` label or 3-of-3 owner sign-off
[ ] 1.10 ./.venv/bin/python -c "from autoqec.envs.schema import EnvSpec; \
         import yaml; EnvSpec.model_validate(
         yaml.safe_load(open('autoqec/envs/builtin/surface_d5_depol.yaml')))"
         # no ValidationError; repeat for bb72_depol.yaml
```

**Gate:** 10/10 green -> proceed to Phase 2. Any FAIL -> open issue labelled
`blocks-demo`.

---

## 3. Phase 2 — Component-Level Isolated Testability

**Purpose:** verify each subsystem can be exercised without the others so a
failure points to a specific module. Split by §10 architecture into seven
subsections.

**Operator:** team (Day-3 pre-flight sweep). **Environment:** CPU is
sufficient for most checks; GPU-gated ones are noted.

### 3.1 Schemas & Contract

```
[ ] 2.1.1 pytest tests/test_dsl_schema.py tests/test_agent_schemas_worktree.py \
          tests/test_runner_schema_worktree.py tests/test_verify_report_worktree.py -v
          # all green; count >= 20
[ ] 2.1.2 hand-crafted malformed IdeatorResponse (fork_from missing) -> parse_response
          raises pydantic ValidationError; error message names the missing field
[ ] 2.1.3 the 6 pydantic signatures in docs/contracts/interfaces.md match the code
          (grep cross-check: EnvSpec, RunnerConfig, RoundMetrics, VerifyReport,
          IdeatorResponse, CoderResponse, AnalystResponse)
```

### 3.2 DSL Compiler & Predecoder

```
[ ] 2.2.1 Tier-1 GNN seed: compile_dsl(gnn_small.yaml) returns nn.Module;
          forward(dummy_batch) shape == (B, n_detectors); no NaN
[ ] 2.2.2 Tier-1 Neural-BP seed: same (neural_bp_min.yaml)
[ ] 2.2.3 Tier-2 hostile custom_fn (contains `import os` / `eval(`) -> AST reject;
          benign custom_fn -> smoke test passes
[ ] 2.2.4 pytest tests/test_dsl_compiler.py tests/test_gnn_module.py \
          tests/test_neural_bp_module.py tests/test_custom_fn_validator.py -q
          # all green
```

### 3.3 Classical Backend Adapter

```
[ ] 2.3.1 MWPM on surface_d5 DEM: single-shot latency < 50 ms (median of 100 runs)
[ ] 2.3.2 OSD on bb72 DEM: single-shot latency < 200 ms
[ ] 2.3.3 pytest tests/test_backend_adapter.py tests/test_pymatching_baseline.py \
          tests/test_bposd_baseline.py -q
          # all green
[ ] 2.3.4 scripts/benchmark_surface_baseline.py -> LER in [0.01344, 0.01444]
          (1M shots, p=5e-3, seed=42); abs diff vs
          demos/demo-1-surface-d5/expected_output/baseline_benchmark.json < 5e-4
```

### 3.4 Runner (train + eval)

```
[ ] 2.4.1 scripts/e2e_handshake.py --round-dir /tmp/hs1 writes
          checkpoint.pt + metrics.json + train.log; train.log step count >= 100
[ ] 2.4.2 metrics.json parses as RoundMetrics;
          delta_vs_baseline_holdout / flops_per_syndrome / n_params are not None
[ ] 2.4.3 run_round(cfg_with_code_cwd=X) called in-process -> raises
          RunnerCallPathError (tests/test_runner_guard.py)
[ ] 2.4.4 pytest tests/test_runner_*.py tests/test_run_quick.py \
          tests/test_run_single_round.py -q
          # all green
```

### 3.5 Orchestration (Memory / Recorder / Worktree / Fork-graph / Reconcile)

```
[ ] 2.5.1 pytest tests/test_round_recorder.py tests/test_pareto.py -q
          # all green; Pareto algorithm is non-dominated filter (not top-5 cut);
          feeding 5 mutually dominated points -> output length 1
[ ] 2.5.2 pytest tests/test_worktree.py -q covers
          create / compose_pure / compose_conflict / cleanup paths
[ ] 2.5.3 pytest tests/test_fork_graph.py -q; empty-history case -> fork_graph
          contains only the baseline node
[ ] 2.5.4 pytest tests/test_reconcile.py -q covers B\H (orphan), H\B
          (branch deleted), both-missing; recoverable orphan writes a synthetic
          row and preserves the original round_attempt_id
[ ] 2.5.5 pytest tests/test_subprocess_runner.py tests/test_orchestration_stub.py \
          tests/test_loop_helpers.py tests/test_cli_run_paths.py \
          tests/test_cli_run_round_worktree.py -q
          # all green
```

### 3.6 Subagent Prompts

```
[ ] 2.6.1 build_prompt('ideator', ctx) output contains `fork_graph` key and
          does NOT contain `last_5_hypotheses` (regression guard)
[ ] 2.6.2 build_prompt('coder', ctx) contains `commit_message` slot and
          describes Tier-2 custom_fn AST rules
[ ] 2.6.3 build_prompt('analyst', ctx) requires branch + commit_sha echo
[ ] 2.6.4 AUTOQEC_IDEATOR_BACKEND=claude-cli make run actually switches backend
          (tests/test_loop_helpers.py env-dispatch cases)
```

### 3.7 Safety / Tool Whitelist

```
[ ] 2.7.1 pytest tests/test_runner_safety.py tests/test_isolation_rule.py \
          tests/test_reward_hacking.py -q
          # all green; test_reward_hacking.py constructs a DSL that reads
          training-set syndromes at eval time -> safety layer must raise
[ ] 2.7.2 tools.machine_state with CUDA unavailable returns `{}` (does not raise);
          pytest tests/test_machine_state.py -q
```

**Gate:** approx. 28 checkboxes across 7 subsections; >= 95% green (at most
one non-core flaky item may be quarantined with an owner + deadline issue)
-> proceed to Phase 3.

---

## 4. Phase 3 — Single Research Round (No-LLM Handshake)

**Purpose:** prove one research round walks from RunnerConfig to training,
evaluation, and L1 persistence without needing real Ideator/Coder/Analyst
LLM calls. Isolates pure engineering defects from LLM flakiness.

**Operator:** team. **Environment:** GPU (surface_d5 dev profile runs about
5-15 minutes).

### 4.1 `run_quick` flat path

```
[ ] 3.1.1 ./.venv/bin/python scripts/run_quick.py \
          --env autoqec/envs/builtin/surface_d5_depol.yaml \
          --run-dir /tmp/rq1 --rounds 1
          # exit 0; wall-clock < 15 min; GPU VRAM peak < 80% of available
[ ] 3.1.2 /tmp/rq1/history.jsonl has exactly 1 line; JSON valid;
          RoundMetrics.model_validate passes
[ ] 3.1.3 /tmp/rq1/round_01/ contains: config.yaml, train.log,
          checkpoint.pt, metrics.json
[ ] 3.1.4 metrics.json.delta_vs_baseline_holdout in [-0.02, 0.02]
          (sanity: cannot drift far under the same random seed)
```

### 4.2 Handshake script

```
[ ] 3.2.1 ./.venv/bin/python scripts/e2e_handshake.py --round-dir /tmp/hs2
          # exit 0; artifacts same as 3.1.3
[ ] 3.2.2 hs2/metrics.json.flops_per_syndrome > 0 and matches fvcore
          (autoqec.runner.flops)
[ ] 3.2.3 checkpoint.pt is torch.load-replayable; loaded dict has all four
          keys: class_name, state_dict, output_mode, dsl_config
```

### 4.3 No-LLM single round via CLI

```
[ ] 3.3.1 ./.venv/bin/python -m cli.autoqec run \
          autoqec/envs/builtin/surface_d5_depol.yaml \
          --rounds 1 --profile dev --no-llm --run-dir /tmp/cli1
          # exit 0; wall-clock < 15 min
[ ] 3.3.2 /tmp/cli1/history.jsonl row carries a UUID-v4 `round_attempt_id`;
          `reconcile_id` is None (mutual-exclusion invariant)
[ ] 3.3.3 /tmp/cli1/log.md non-empty; contains a "Round 1" heading
[ ] 3.3.4 If VERIFIED: pareto.json contains at least 1 entry;
          pareto_preview.json is the same set sorted by -delta_vs_baseline_holdout.
          If not VERIFIED: pareto.json stays empty or unchanged (no invalid row).
```

### 4.4 Worktree round

```
[ ] 3.4.1 ./.venv/bin/python -m cli.autoqec run-round <env> <cfg.yaml> /tmp/wt1 \
          --code-cwd .worktrees/wt1-test --branch exp/test/01-smoke \
          --fork-from baseline --compose-mode none
          # exit 0; .worktrees/wt1-test/ is a git worktree
          # (visible in `git worktree list`)
[ ] 3.4.2 `git log exp/test/01-smoke --oneline | head -1` has feat/fix/test prefix
          (Coder commit_message was written)
[ ] 3.4.3 metrics.json.branch == "exp/test/01-smoke"; commit_sha is a 40-hex string
[ ] 3.4.4 round_01_pointer.json exists in the worktree's round_01/;
          `git show exp/test/01-smoke:round_01/round_01_pointer.json` round-trips
[ ] 3.4.5 cleanup: invoke `autoqec.orchestration.worktree.cleanup_round_worktree`
          (programmatic; no CLI wrapper yet — see O-6) -> .worktrees/wt1-test/
          removed; branch `exp/test/01-smoke` retained (branches-as-Pareto)
```

### 4.5 Post-round invariants

```
[ ] 3.5.1 history.jsonl is append-only: pre/post wc -l shows only growth;
          no row deletions
[ ] 3.5.2 Runner writes only inside round_<N>/; orchestrator writes only at
          run-root. `find /tmp/cli1 -newer /tmp/cli1/.marker` shows no
          cross-boundary writes
[ ] 3.5.3 fork_graph.json (if present) has round_1.parent == "baseline"
[ ] 3.5.4 no orphan worktrees: `git worktree list | grep -v main` line count
          == active worktree count reported by RunMemory
```

**Gate:** Phase 3's 5 subsections, approx. 18 checkboxes, all green; at most
1 quarantine under 3.5. Any FAIL in 3.1-3.4 -> block demo.

---

## 5. Phase 4 — Full Research Loop (Live LLM)

**Purpose:** end-to-end verification of `cli.autoqec run <env> --rounds N`
with real LLM calls through the full Ideator -> Coder -> Runner -> Analyst
-> (Verifier) DAG, once per reference environment. This is the artifact the
advisor will watch.

**Operator:** team (not advisor; advisor will replay). **Environment:**
GPU + network; backend env vars as in CLAUDE.md.

### 5.1 surface_d5 full loop

```
[ ] 4.1.1 AUTOQEC_IDEATOR_BACKEND=codex-cli AUTOQEC_IDEATOR_MODEL=gpt-5.4 \
          AUTOQEC_CODER_BACKEND=codex-cli AUTOQEC_CODER_MODEL=gpt-5.4-codex \
          AUTOQEC_ANALYST_BACKEND=claude-cli AUTOQEC_ANALYST_MODEL=claude-haiku-4-5 \
          ./.venv/bin/python -m cli.autoqec run \
          autoqec/envs/builtin/surface_d5_depol.yaml --rounds 3 --profile dev
          # exit 0; wall-clock < 75 min
[ ] 4.1.2 runs/<run_id>/history.jsonl has 3 rows; each is a valid RoundMetrics;
          each round_attempt_id is unique
[ ] 4.1.3 runs/<run_id>/pareto.json length >= 1 (baseline or a VERIFIED round);
          non-domination invariant: for all a, b in pareto, neither dominates
          the other on (delta, flops, n_params)
[ ] 4.1.4 runs/<run_id>/log.md contains 3 human-readable narrative blocks,
          each with hypothesis -> result -> verdict
[ ] 4.1.5 runs/<run_id>/fork_graph.json has 4 nodes (baseline + 3 rounds);
          at least one round's parent != baseline (evidence that Ideator
          used fork_graph for a non-trivial decision)
[ ] 4.1.6 `git branch --list 'exp/<run_id>/*'` has 3 entries; every branch tip
          commit has feat/fix/test/docs/test/chore prefix
```

### 5.2 bb72 full loop

```
[ ] 4.2.1 same backend env as 4.1.1, env swapped to bb72_depol.yaml
          # exit 0; wall-clock < 120 min (OSD is slower than MWPM)
[ ] 4.2.2 - 4.2.6 mirror 4.1.2 - 4.1.6
[ ] 4.2.7 classical backend == "osd" (grep runs/<id>/round_01/config.yaml)
```

### 5.3 LLM dispatch / machine_state loop

```
[ ] 4.3.1 the Ideator prompt log (runs/<id>/round_N/ideator_prompt.json)
          contains fork_graph, machine_state, pareto_front keys, and does
          NOT contain last_5_hypotheses
[ ] 4.3.2 at least one round has a non-trivial fork_from (!= "baseline"),
          showing Ideator read the fork_graph; if all 3 rounds have
          fork_from == baseline, emit a warning to log.md (not a block)
[ ] 4.3.3 machine_state tool is called at least once across the 3 rounds
          (look for `tool_call: machine_state` in logs)
[ ] 4.3.4 if compose_mode != none is triggered: compose round's
          metrics.json.status is in {VERIFIED, SUSPICIOUS, FAILED,
          compose_conflict} (never "unknown")
[ ] 4.3.5 per-env, 3-round total token usage <= 500K input + 80K output
          (sum the usage fields from Codex / Claude CLI output)
```

### 5.4 Loop-level invariants

```
[ ] 4.4.1 interrupt + resume is idempotent: run with rounds=3, Ctrl+C during
          round 2, re-run -> continues with round 3, does not redo finished
          rounds; md5 of the first 2 history.jsonl rows is unchanged
[ ] 4.4.2 reconcile trip: manually `git branch -D exp/<run_id>/01-*` -> next
          startup writes an idempotent `branch_manually_deleted` synthetic
          row; starting again does not rewrite it
[ ] 4.4.3 .worktrees/ is not tracked: `git ls-files .worktrees/` is empty
[ ] 4.4.4 runs/ is not tracked: same
[ ] 4.4.5 no runner boundary violation: mtime of history.jsonl + log.md
          aligns with orchestrator process; round_<N>/* aligns with runner
          subprocess (spot check)
```

**Gate:**
- 4.1 AND 4.2 must pass (advisor expects to see both). Any FAIL -> block demo.
- 4.3 tolerates at most 1 warning soft-fail.
- 4.4 tolerates at most 2 quarantines (each with follow-up issue).

---

## 6. Phase 5 — Verification & Diagnostic Skill Layer

**Purpose:** turn every Δ_LER number into something independently
verifiable; demonstrate that reward-hacking attempts are actively detected;
demonstrate that runs can be human-read and diagnosed. This is the
answer to "how do you prove this is not cheating?"

**Operator:** Xie Jingu primary, team cross-checks. **Environment:** CPU
sufficient (independent eval does not train).

### 6.1 `/verify-decoder` positive case

```
[ ] 5.1.1 pick a candidate (Analyst verdict == candidate) from Phase 4;
          ./.venv/bin/python -m cli.autoqec verify \
          runs/<run_id>/round_02 --env autoqec/envs/builtin/surface_d5_depol.yaml \
          --n-seeds 50
          # exit 0; writes verification_report.md AND verification_report.json
          # inside the round directory
[ ] 5.1.2 verification_report.md carries verdict in {VERIFIED, SUSPICIOUS,
          FAILED}; delta_ler_holdout with bootstrap 95% CI [lo, hi];
          paired_eval_bundle_id is a non-empty UUID
[ ] 5.1.3 VerifyReport.model_validate parses verification_report.json
[ ] 5.1.4 if verdict == VERIFIED: this round appears in pareto.json;
          if SUSPICIOUS/FAILED: it does not
```

### 6.2 `/verify-decoder` reward-hacking traps

Three deliberately cheating checkpoints. Verifier must reject each.

```
[ ] 5.2.1 Trap-A "training-syndrome leak": predecoder embeds a lookup
          over training seeds' syndromes -> verdict != VERIFIED;
          verification_report.md contains "holdout mismatch" or "seed leak"
[ ] 5.2.2 Trap-B "paired batch mismatch": predecoder is evaluated on an
          easier batch than the plain-classical baseline -> verdict !=
          VERIFIED; paired_eval_bundle_id must be identical across the two
          evaluations
[ ] 5.2.3 Trap-C "overfit tiny sample": train on 100 shots, checkpoint
          memorizes -> independent eval on 200K holdout -> verdict ==
          FAILED OR 95% CI of delta_vs_baseline_holdout crosses 0
[ ] 5.2.4 trap fixtures live at tests/fixtures/reward_hacking/trap_{A,B,C}.pt;
          pytest tests/test_reward_hacking.py -q -> all green
[ ] 5.2.5 `independent_eval.py` imports no symbol from autoqec.runner.*
          (spec §10 CI guard); cross-check via grep
```

### 6.3 `/review-log`

```
[ ] 5.3.1 run /review-log against Phase-4 produced runs/<run_id>/log.md
          -> produces runs/<run_id>/review.md
[ ] 5.3.2 review.md has four sections (strict names): narrative coherence,
          stuck hypotheses, overfitting signals, next-action recommendation
[ ] 5.3.3 on a synthetic stuck run (three consecutive Δ_LER approx 0 rounds
          injected), review.md must call out the pattern under "stuck
          hypotheses"
[ ] 5.3.4 review.md length >= 300 and <= 2000 words (not padding, not
          an LLM sprawl)
```

### 6.4 `/diagnose-failure`

```
[ ] 5.4.1 inject three failure modes:
          (a) oversized hidden_dim -> OOM
          (b) LR=10.0 -> NaN loss
          (c) env yaml with p=0 -> degenerate eval
          for each, run /diagnose-failure --run-dir <path> --round N
          -> produces diagnosis.md
[ ] 5.4.2 diagnosis.md.root_cause contains (fuzzy match):
          (a) "OOM" or "VRAM" or "memory"
          (b) "NaN" or "learning rate" or "divergence"
          (c) "degenerate" or "p=0" or "no errors"
[ ] 5.4.3 diagnosis.md.recommended_fix is non-empty; the skill does NOT
          autonomously apply a fix (per spec)
```

### 6.5 Statistical correctness

```
[ ] 5.5.1 bootstrap CI unit test: known LER=0.01, N=200K, 1000 resamples
          -> 95% CI width < 0.002 (consistent with analytic
          2 * 1.96 * sqrt(p*(1-p)/N)); in tests/test_independent_eval.py
[ ] 5.5.2 holdout-seed isolation: training_seeds.isdisjoint(holdout_seeds)
          asserted in code
[ ] 5.5.3 paired eval: under a given paired_eval_bundle_id, the syndrome
          tensors used for plain_classical and predecoder+classical are
          byte-hash equal
[ ] 5.5.4 ablation sanity: a random-weights predecoder must have
          Δ_LER 95% CI crossing 0 (no systematic positive bias in the
          evaluation protocol); in tests/test_independent_eval.py
```

**Gate:**
- 5.1, 5.2, 5.5 must all pass (research integrity is non-negotiable).
- 5.3 and 5.4 must each produce at least one live output (visual check OK).
- **Any Trap in 5.2 that gets verdict == VERIFIED blocks the demo** and
  `independent_eval.py` must be fixed before re-running.

---

## 7. Test Data

```
demos/demo-1-surface-d5/expected_output/baseline_benchmark.json  # anchor LER=0.01394
tests/fixtures/reward_hacking/trap_{A,B,C}.pt                    # Phase 5 traps
autoqec/example_db/gnn_*.yaml, neural_bp_*.yaml                  # Tier-1 seeds
autoqec/envs/builtin/{surface_d5_depol,bb72_depol}.yaml          # reference envs
holdout seeds: [10, 20, 30]  (fixed; disjoint from training seeds [0..9])
```

---

## 8. Open Questions (must close before test plan execution)

```
[ ] O-1 CI runner: CPU only, or self-hosted GPU? Decides whether Phase 3/4
        can run in CI. Default assumption: local GPU + CI CPU only.
[ ] O-2 Phase 4 token budget cap (500K input / 80K output) — does this
        match the actual billed usage of AUTOQEC_* backends? A one-shot
        calibration run is needed.
[ ] O-3 Owner of the three Phase-5.2 trap fixtures — proposed: Xie Jingu.
        Deadline: 12 hours before Day-3 demo.
[ ] O-4 Advisor demo day: live run (re-run Phase 4) or replay (recorded
        Phase 4)? Affects the Phase 1 wall-clock gate for the advisor.
[ ] O-5 `/diagnose-failure` injected-failure fixtures: are the three cases
        (OOM / NaN / p=0) already canned, or do we need to build them?
[ ] O-6 `autoqec cleanup-worktree` CLI wrapper does not yet exist; either
        add a thin wrapper over `cleanup_round_worktree` before demo, or
        adjust 3.4.5 to call the Python API directly.
```

---

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Day-3 scope too large (per status §4.5) | High | Phase 4 yellow | Prioritize Phase 1+2+3 and a single-env Phase 4 |
| Reward-hack trap itself is buggy (rejects legitimate checkpoints) | Medium | Phase 5 all fails | Run each trap against the baseline first; confirm verdict == VERIFIED for the clean case |
| LLM backend flaky (Codex CLI drops) | Medium | Phase 4 stuck | Fall back to claude-cli; plan documents both backend combinations as verified |
| `pareto.json` concurrent overwrite | Low | Data loss | Phase 2.5 covers it; Phase 4 adds mtime invariant (4.4.5) |
| `cli/autoqec.py::run` relative `round_dir` bug (status §4.5) unfixed | Medium | Verifier can't load checkpoints | Phase 3.4 must pass; merge Lin's one-line fix before Phase 1 |

---

## 10. Success Criteria Summary

| Gate | Minimum pass | Blocks demo? |
|---|---|---|
| Phase 1 | 10/10 green | Yes |
| Phase 2 | >= 95% green (at most 1 quarantine) | Yes |
| Phase 3 | 3.1-3.4 all green; at most 1 quarantine in 3.5 | Yes |
| Phase 4 | 4.1 AND 4.2 all green; 4.3 at most 1 warning; 4.4 at most 2 quarantines | Yes (at least one env must pass) |
| Phase 5 | 5.1 + 5.2 + 5.5 all green; 5.3 + 5.4 each produce one live output | Yes |

**Overall demo-ready verdict:** all five phase gates met simultaneously ->
ship to advisor; otherwise block and enumerate open items in a GitHub issue.

---

## References

- `docs/superpowers/specs/2026-04-20-autoqec-design.md` — authoritative
  design (v2.3); source for spec §10 architecture, reward-hacking
  defenses, and Pareto-as-non-dominated-set.
- `docs/contracts/interfaces.md` — pydantic schemas exercised by
  Phase 2.1 and Phase 5.1.
- `docs/contracts/round_dir_layout.md` — who writes what inside
  `runs/<run_id>/round_<N>/`; Phase 3.5.2 enforces its boundary rule.
- `docs/superpowers/specs/2026-04-22-ci-coverage-design.md` — engineering
  regression suite that complements this acceptance plan.
- `docs/status/2026-04-22-project-status.md` — Day-2 status snapshot;
  source for Risk list.
- GitHub issue QuAIR/0420-FPGA-Decoder#3 — structural inspiration
  (phased acceptance checklist). This plan tightens the quantification
  that the FPGA plan left soft.
