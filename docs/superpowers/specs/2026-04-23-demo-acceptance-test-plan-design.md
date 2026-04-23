# Demo-Acceptance Test Plan — qec-ai-decoder

**Status:** design, revised after Codex cross-model review (2026-04-23)
**Date:** 2026-04-23
**Gate:** whether Day-3 sprint output can be handed to the advisor
**Revision notes:** v2 aligns every CLI invocation and fixture reference
with actual repo state (as of commit 28a9abf), separates gates that are
executable today from those that depend on unmerged features, and folds in
five missing failure modes from the Codex review.

## 1. Overview

This test plan is the **human-verification gate** between Day-3 sprint close
and the advisor walkthrough. It does not replace the engineering CI suite
(see `docs/superpowers/specs/2026-04-22-ci-coverage-design.md`). Instead it
answers a single question: on a cold checkout, can someone reproduce the
Δ_LER numbers the team is claiming and trust them?

### 1.1 Objectives

1. **Reproducibility** — clone → install → single command reproduces the
   baseline LER within bootstrap CI.
2. **Correctness** — at least one reference env (`surface_d5_depol` or
   `bb72_depol`) completes a full research loop (live-LLM if wired;
   no-LLM if not) with structurally valid `history.jsonl` / `pareto.json`.
3. **Research integrity** — `independent_eval.py`'s fair-baseline guards
   (holdout isolation, paired eval, tool whitelist) can be tripped by
   deliberately malicious checkpoints.
4. **Observability** — `review-log`, `diagnose`, and `verify` CLI
   subcommands each emit structured output over a real run directory.
5. **Contract hygiene** — every pydantic schema in `docs/contracts/interfaces.md`
   round-trips; `pytest -m "not integration"` is fully green; the GitHub
   Actions workflow on `main` is green.

### 1.2 Scope

- In scope: the five phases below.
- Out of scope: long-term regression / release gating (owned by
  `ci-coverage-design.md`) and subjective research novelty (advisor's
  own judgement).

### 1.3 Structure

Five phases, each with inline hard-number thresholds in every checkbox.
Every checkbox is its own gate — there is no separate softer "Success
Criteria" table. This mirrors the QuAIR FPGA decoder test plan
(GitHub issue `QuAIR/0420-FPGA-Decoder#3`) but tightens the quantification
of pass/fail that that reference left soft.

### 1.4 Environment assumption

Commands are written for Linux or macOS (or Windows under WSL). A native
Windows PowerShell checklist is out of scope for Day-3 — commands like
`./.venv/bin/...`, `/tmp`, `wc`, `find`, `grep` do not have direct Windows
equivalents and porting them now is not the bottleneck.

### 1.5 How to execute

Each phase is run top-to-bottom. A FAIL must be filed as a GitHub issue
labelled `blocks-demo` (hard gate) or `quarantine` (soft gate) before the
phase is considered complete. Quarantine issues must carry an owner and
a deadline; demo can proceed only if the per-phase Gate rule admits the
quarantine count.

### 1.6 Prerequisites (must land before the phase can execute)

Some phases depend on features that are not on `main` as of commit
`28a9abf`. The plan marks these with a **`DEPENDS-ON`** callout and the
plan executor is allowed to skip the dependent checkboxes until the
listed prerequisite is merged. Skipping is recorded as `blocked` (not
`quarantine`) and does not count against the per-phase gate denominator
until the prerequisite lands.

| Prerequisite | Required by | Status |
|---|---|---|
| `cli.autoqec run` wires a live Ideator → Coder → Analyst DAG (currently raises unless `--no-llm`) | Phase 4 (§5) | not landed |
| `autoqec verify` persists `paired_eval_bundle_id` in `VerifyReport` and updates the run's `pareto.json` on VERIFIED | Phase 5.1 (§6.1) | schema field exists but writer unset; pareto integration absent |
| `tests/fixtures/reward_hacking/trap_{A,B,C}.pt` fixtures and matching `test_reward_hacking.py` cases | Phase 5.2 (§6.2) | fixtures absent; only synthetic memorizer tests exist |
| `review-log` emits a structured `review.md`; `diagnose` emits a `diagnosis.md` (both currently print JSON to stdout only) | Phase 5.3 + 5.4 (§6.3, §6.4) | not landed |
| `autoqec cleanup-worktree` CLI wrapper (or documented Python API call convention) | Phase 3.4.5 (§4.4.5) | Python API `cleanup_round_worktree` exists; no CLI wrapper |

---

## 2. Phase 1 — Infrastructure / Cold-Start Reproducibility

**Purpose:** validate that a clean machine can get from zero to the first
runnable command. Any FAIL here invalidates every downstream phase.

**Operator:** advisor or new collaborator. **Environment:** clean
Linux / macOS / WSL, Python 3.12, no pre-existing `.venv/`.

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
        # `gh run list --branch main --limit 1 --json conclusion` -> [{"conclusion":"success"}]
[ ] 1.8 directory structure matches spec §10:
        autoqec/{envs,agents,decoders,runner,eval,orchestration,tools}/,
        cli/, circuits/, demos/, tests/, docs/{contracts,superpowers}/
[ ] 1.9 docs/contracts/{interfaces.md,round_dir_layout.md} last commit has
        `contract-change` label or 3-of-3 owner sign-off
[ ] 1.10 ./.venv/bin/python -c "from autoqec.envs.schema import load_env_yaml; \
         load_env_yaml('autoqec/envs/builtin/surface_d5_depol.yaml')"
         # exits 0; repeat for autoqec/envs/builtin/bb72_depol.yaml
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
[ ] 2.1.2 hand-crafted IdeatorResponse with `rationale` field removed
          (rationale IS a required field; fork_from defaults to "baseline")
          -> `parse_response` wraps the underlying pydantic ValidationError
          as a ValueError whose message names the missing field
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
[ ] 2.4.1 ./.venv/bin/python scripts/e2e_handshake.py --round-dir /tmp/hs1
          # exit 0; writes checkpoint.pt + metrics.json + train.log;
          # train.log step count >= 100
[ ] 2.4.2 metrics.json parses as RoundMetrics; delta_ler / flops_per_syndrome
          / n_params are not None
[ ] 2.4.3 run_round(cfg_with_code_cwd=X) called in-process -> raises
          RunnerCallPathError (tests/test_runner_guard.py)
[ ] 2.4.4 pytest tests/test_runner_smoke.py tests/test_runner_safety.py \
          tests/test_runner_guard.py tests/test_runner_schema_worktree.py \
          tests/test_run_quick.py tests/test_run_single_round.py -q
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
          # all green; test_reward_hacking.py constructs a predecoder that
          reads training-set syndromes at eval time -> safety layer must raise
[ ] 2.7.2 tools.machine_state with CUDA unavailable returns `{}` (does not raise);
          pytest tests/test_machine_state.py -q
```

**Checkbox count:** 26 total across 2.1 .. 2.7.
**Gate:** >= 25/26 green (at most one non-core flaky item may be
quarantined with an owner + deadline issue) -> proceed to Phase 3.

---

## 4. Phase 3 — Single Research Round (No-LLM Handshake)

**Purpose:** prove one research round walks from RunnerConfig to training,
evaluation, and L1 persistence without depending on real Ideator /
Coder / Analyst LLM calls. Isolates pure engineering defects from LLM
flakiness.

**Operator:** team. **Environment:** GPU (surface_d5 dev profile runs
about 5–15 minutes).

### 4.1 `run_quick` flat path

```
[ ] 3.1.1 ./.venv/bin/python scripts/run_quick.py --rounds 1 --profile dev \
          --env-yaml autoqec/envs/builtin/surface_d5_depol.yaml
          # exit 0; wall-clock < 15 min; GPU VRAM peak < 80% of available.
          # The script picks the run dir itself under `runs/<timestamp>/`
          # and prints the final JSON to stdout including `run_dir`.
[ ] 3.1.2 capture RUN_DIR from the script's stdout; RUN_DIR/history.json
          exists; `json.loads(RUN_DIR/'history.json')` returns a list of
          length 1; entry[0] passes RoundMetrics.model_validate
[ ] 3.1.3 RUN_DIR/round_1/ contains: config.yaml, train.log, checkpoint.pt,
          metrics.json
[ ] 3.1.4 metrics.json.delta_ler in [-0.02, 0.02]
          (sanity: under the fixed seed this cannot drift far)
```

### 4.2 Handshake script

```
[ ] 3.2.1 ./.venv/bin/python scripts/e2e_handshake.py --round-dir /tmp/hs2
          # exit 0; artifacts same as 3.1.3
[ ] 3.2.2 /tmp/hs2/metrics.json.flops_per_syndrome > 0 and matches fvcore
          (autoqec.runner.flops)
[ ] 3.2.3 /tmp/hs2/checkpoint.pt is torch.load-replayable; loaded dict has
          all four keys: class_name, state_dict, output_mode, dsl_config
```

### 4.3 No-LLM single round via CLI

```
[ ] 3.3.1 ./.venv/bin/python -m cli.autoqec run \
          autoqec/envs/builtin/surface_d5_depol.yaml \
          --rounds 1 --profile dev --no-llm
          # exit 0; wall-clock < 15 min.
          # The CLI writes under runs/<YYYYMMDD-HHMMSS>/ (not user-controlled).
          # The final stdout line begins with AUTOQEC_RESULT_JSON= and contains
          # a JSON blob whose `run_dir` key is the authoritative path.
[ ] 3.3.2 parse AUTOQEC_RESULT_JSON and name the resulting path CLI_RUN_DIR.
          CLI_RUN_DIR/history.jsonl exists (append-only jsonl written by
          RunMemory); every line passes RoundMetrics.model_validate.
          Note: the `run --no-llm` command also writes a separate
          `history.json` (single JSON list) for convenience — both files are
          expected.
[ ] 3.3.3 CLI_RUN_DIR/log.md non-empty; contains a "Round 1" heading
[ ] 3.3.4 CLI_RUN_DIR/candidate_pareto.json exists (no-LLM path writes to
          this filename; live-LLM path writes pareto.json via RunMemory
          default). Contents are a list sorted by the non-dominated filter.
          If the round was not VERIFIED: list may be empty.
```

### 4.4 Worktree round

```
[ ] 3.4.1 ./.venv/bin/python -m cli.autoqec run-round <env.yaml> \
          <cfg.yaml> <round_dir> \
          --code-cwd .worktrees/wt1-test --branch exp/test/1-smoke \
          --fork-from baseline \
          --round-attempt-id <new-uuid-v4>
          # single-parent rounds OMIT --compose-mode (valid choices are
          # pure | with_edit; "none" is not a valid value).
          # exit 0; .worktrees/wt1-test/ is a git worktree
          # (visible in `git worktree list`)
[ ] 3.4.2 `git log exp/test/1-smoke --oneline | head -1` has feat/fix/test
          prefix (Coder commit_message was written by the runner / subprocess)
[ ] 3.4.3 metrics.json.branch == "exp/test/1-smoke"; commit_sha is a 40-hex string
[ ] 3.4.4 round_1_pointer.json exists in the worktree's round_1/ directory;
          `git show exp/test/1-smoke:round_1/round_1_pointer.json` round-trips
[ ] 3.4.5 cleanup via Python API (no CLI wrapper exists — see prerequisite
          in §1.6): `python -c "from autoqec.orchestration.worktree import
          cleanup_round_worktree; cleanup_round_worktree('.worktrees/wt1-test')"`
          -> .worktrees/wt1-test/ removed; branch `exp/test/1-smoke` retained
          (branches-as-Pareto)
```

### 4.5 Post-round invariants

```
[ ] 3.5.1 history.jsonl (CLI live path) is append-only: pre/post wc -l shows
          only growth; no row deletions. history.json (no-LLM path) is
          rewritten atomically each round — its size may not grow monotonically
          but its length MUST strictly increase.
[ ] 3.5.2 Runner writes only inside round_<N>/; orchestrator writes only at
          run-root. Verified by snapshotting directory mtimes before and
          after the round and asserting no orchestrator-owned file under
          round_<N>/ was modified by the runner (spot-check via
          `find $RUN_DIR -newer $RUN_DIR/.marker -not -path '*/round_*/*'`
          returning only orchestrator-owned filenames: history.jsonl,
          history.json, log.md, candidate_pareto.json, pareto.json,
          fork_graph.json).
[ ] 3.5.3 fork_graph.json (if present) has round_1.parent == "baseline"
[ ] 3.5.4 no orphan worktrees: `git worktree list | grep -v main` line count
          == active worktree count reported by RunMemory
```

**Gate:** Phase 3's 5 subsections, 18 checkboxes total, all green;
at most 1 quarantine under 3.5. Any FAIL in 3.1–3.4 -> block demo.

---

## 5. Phase 4 — Full Research Loop (Live LLM)

**`DEPENDS-ON`:** `cli.autoqec run` currently raises
`ClickException("LLM mode is not wired in this branch yet; use --no-llm")`
unless `--no-llm` is passed. The Ideator → Coder → Analyst DAG is not
wired into the `run` command as of commit `28a9abf`. **The entire
Phase 4 is blocked until that wiring lands.** Phase 3 (no-LLM path)
is the highest executable gate until then.

If live wiring does not land before Day-3 demo, the plan regresses to:
advisor-ready = Phase 1 + 2 + 3 + Phase 5 on a **no-LLM-produced**
candidate (Phase 3.3 `candidate_pareto.json` entry), with the gap
explicitly called out in the walkthrough.

**Purpose (once prerequisite lands):** end-to-end verification of
`cli.autoqec run <env> --rounds N` with real LLM calls through the full
Ideator → Coder → Runner → Analyst → (Verifier) DAG.

**Operator:** team (not advisor; advisor will replay).
**Environment:** GPU + network; backend env vars per CLAUDE.md.

### 5.1 surface_d5 full loop

```
[ ] 4.1.1 AUTOQEC_IDEATOR_BACKEND=codex-cli AUTOQEC_IDEATOR_MODEL=gpt-5.4 \
          AUTOQEC_CODER_BACKEND=codex-cli AUTOQEC_CODER_MODEL=gpt-5.4-codex \
          AUTOQEC_ANALYST_BACKEND=claude-cli AUTOQEC_ANALYST_MODEL=claude-haiku-4-5 \
          ./.venv/bin/python -m cli.autoqec run \
          autoqec/envs/builtin/surface_d5_depol.yaml --rounds 3 --profile dev
          # exit 0; wall-clock < 75 min (provisional — no production evidence yet;
          # re-calibrate after first live dry-run, see Open Question O-2).
          # Capture run_dir from AUTOQEC_RESULT_JSON on stdout.
[ ] 4.1.2 $RUN_DIR/history.jsonl has 3 rows; each is a valid RoundMetrics;
          each round_attempt_id is unique
[ ] 4.1.3 $RUN_DIR/pareto.json length >= 1 (baseline or a VERIFIED round);
          non-domination invariant: for all a, b in pareto, neither dominates
          the other on (delta_ler, flops_per_syndrome, n_params)
[ ] 4.1.4 $RUN_DIR/log.md contains 3 narrative blocks, each containing at
          least one of the literal substrings: "Hypothesis", "Result",
          "Verdict" (string-match gate, not subjective readability)
[ ] 4.1.5 $RUN_DIR/fork_graph.json has 4 nodes (baseline + 3 rounds);
          at least one round's parent != "baseline" (evidence that Ideator
          used fork_graph). If all 3 parents are "baseline": emit a warning
          row to log.md under a "## Warnings" header — not a block.
[ ] 4.1.6 `git branch --list 'exp/<run_id>/*'` has 3 entries; every branch tip
          commit message first token is one of: feat, fix, test, docs, chore,
          refactor, perf, ci (conventional-commits prefix check via regex
          `^(feat|fix|test|docs|chore|refactor|perf|ci)(\(.+\))?:\s`)
```

### 5.2 bb72 full loop

```
[ ] 4.2.1 same backend env as 4.1.1, env swapped to bb72_depol.yaml
          # exit 0; wall-clock < 120 min (provisional; see O-2).
[ ] 4.2.2 – 4.2.6 mirror 4.1.2 – 4.1.6
[ ] 4.2.7 classical backend == "osd" in runs/<id>/round_1/config.yaml
          (parse YAML; `env.classical_backend` key)
```

### 5.3 LLM dispatch / machine_state loop

```
[ ] 4.3.1 the Ideator prompt log (runs/<id>/round_N/ideator_prompt.json, if
          the role persists its prompt; otherwise check the generated context
          object) contains `fork_graph`, `machine_state`, `pareto_front` keys,
          and does NOT contain `last_5_hypotheses`
[ ] 4.3.2 at least one round has a non-trivial `fork_from` — either a
          string other than "baseline" or a list (compose round). If all 3
          rounds have `fork_from == "baseline"`: emit a warning to log.md
          (not a block)
[ ] 4.3.3 machine_state tool is recorded as called in the orchestration
          history at least once across the 3 rounds. Exact match:
          `grep -c '"tool_call":\s*"machine_state"' $RUN_DIR/log.md` >= 1.
          If the tool-call log shape is different in the live wiring,
          update this gate as part of the prerequisite landing.
[ ] 4.3.4 if compose_mode != None is triggered by any round: that round's
          metrics.json.status is in {"ok", "compose_conflict",
          "killed_by_safety"} (known valid RoundMetrics.status values;
          confirm against runner/schema.py when live wiring lands)
[ ] 4.3.5 per-env, 3-round total token usage <= 500K input + 80K output
          (provisional budget — no production evidence yet; re-calibrate
          after first live dry-run, see O-2). Sum the usage fields from
          Codex / Claude CLI output.
```

### 5.4 Loop-level invariants

```
[ ] 4.4.1 interrupt + resume is idempotent: run with rounds=3, Ctrl+C during
          round 2, re-run the same command -> continues with round 3, does
          not redo finished rounds. Verified by diffing
          `head -2 $RUN_DIR/history.jsonl` pre and post.
[ ] 4.4.2 reconcile trip: manually `git branch -D exp/<run_id>/1-*` ->
          next startup writes an idempotent `branch_manually_deleted`
          synthetic row; starting again does not rewrite it
[ ] 4.4.3 .worktrees/ is not tracked: `git ls-files .worktrees/` is empty
[ ] 4.4.4 runs/ is not tracked: same
[ ] 4.4.5 **Dirty-worktree precondition** (new, §9 missing failure mode 1):
          starting a run with uncommitted changes in the main checkout
          emits a warning and records the working-tree SHA of the dirty
          files into the per-round artifact manifest (see §6.6) so the
          run is not mistaken for a clean reproduction.
```

### 5.5 Artifact manifest (new; §9 missing failure mode 2)

```
[ ] 4.5.1 each round_N/ contains an artifact_manifest.json with at least:
          repo_sha (HEAD), branch, env_yaml_sha256, dsl_config_sha256,
          python_version, torch_version, cuda_version (or "none"),
          stim_version, pymatching_version, ldpc_version, full command line
          invoked. If the runner does not yet write this, the plan writes
          an issue `blocks-demo:artifact-manifest` and 4.5.1 is marked
          DEPENDS-ON.
[ ] 4.5.2 re-running from the recorded command line on the same repo_sha
          reproduces the round within the bootstrap-CI tolerance (spot
          check: re-run round 1, assert |new_delta_ler - old_delta_ler|
          within 2× the 95% CI half-width).
```

### 5.6 LLM containment (new; §9 missing failure mode 3)

```
[ ] 4.6.1 after a live round completes, verify the experiment branch
          introduced no changes outside the worktree's allowed scope:
          `git diff baseline..exp/<run_id>/N-* -- 'runs/' 'tests/'
          'autoqec/envs/builtin/' 'autoqec/eval/'` returns empty.
          Any diff under those paths = containment breach = block demo.
[ ] 4.6.2 `git diff baseline..exp/<run_id>/N-* -- 'docs/contracts/'` is empty
          (contracts freeze).
```

**Gate:**
- If the prerequisite in §1.6 has not landed: all of Phase 4 is
  blocked; this is NOT a failure (see §1.5), but demo must call out
  the reduced scope.
- Else: 4.1 AND 4.2 both green (both envs must pass — this overrides
  the earlier draft's "at least one env" language, per Codex M4).
  4.3 tolerates at most 1 warning soft-fail. 4.4 + 4.5 + 4.6 tolerate
  at most 2 quarantines.

---

## 6. Phase 5 — Verification & Diagnostic Skill Layer

**Purpose:** turn every Δ_LER number into something independently
verifiable; demonstrate reward-hacking attempts are actively detected;
demonstrate runs can be human-read and diagnosed. This is the answer to
"how do you prove this is not cheating?"

**Operator:** Xie Jingu primary, team cross-checks. **Environment:** CPU
sufficient (independent eval does not train).

### 6.1 `autoqec verify` positive case

**`DEPENDS-ON`:** `independent_verify` must set
`VerifyReport.paired_eval_bundle_id` (the schema field exists but is
never written). VERIFIED reports must be wired to update the run's
`pareto.json`. Both are unmerged as of commit `28a9abf`.

```
[ ] 5.1.1 pick a candidate (Analyst verdict == candidate) from Phase 4 or
          Phase 3's no-LLM path;
          ./.venv/bin/python -m cli.autoqec verify \
          runs/<run_id>/round_2 --env autoqec/envs/builtin/surface_d5_depol.yaml \
          --n-seeds 50
          # exit 0; writes verification_report.md AND verification_report.json
          # inside the round directory
[ ] 5.1.2 verification_report.json parses as VerifyReport; verdict in
          {VERIFIED, SUSPICIOUS, FAILED}; delta_ler_holdout carries a
          bootstrap 95% CI [lo, hi]; paired_eval_bundle_id is a non-empty
          UUID (DEPENDS-ON field to be wired)
[ ] 5.1.3 if verdict == VERIFIED: the round appears in the run's
          pareto.json (DEPENDS-ON verify↔pareto integration to land)
          if SUSPICIOUS/FAILED: the round does NOT appear in pareto.json
```

### 6.2 `autoqec verify` reward-hacking traps

**`DEPENDS-ON`:** the trap fixtures
`tests/fixtures/reward_hacking/trap_{A,B,C}.pt` do not currently exist.
Existing `tests/test_reward_hacking.py` builds synthetic memorizer
checkpoints in a tmp dir. Until the explicit fixtures land, 5.2 falls
back to verifying the existing synthetic tests behave as expected.

```
[ ] 5.2.1 Trap-A "training-syndrome leak": predecoder embeds a lookup
          over training seeds' syndromes -> verify verdict != VERIFIED;
          verification_report.md contains literal substrings
          "holdout" AND ("mismatch" OR "leak" OR "isolation")
[ ] 5.2.2 Trap-B "paired batch mismatch": predecoder is evaluated on an
          easier batch than the plain-classical baseline -> verdict !=
          VERIFIED; paired_eval_bundle_id values of the two compared
          evaluations are identical (byte-equal string comparison)
[ ] 5.2.3 Trap-C "overfit tiny sample": train on 100 shots, predecoder
          memorizes -> independent eval on >= 200K holdout shots ->
          verdict == FAILED OR the 95% CI of delta_ler_holdout crosses 0
[ ] 5.2.4 trap fixtures live at tests/fixtures/reward_hacking/trap_{A,B,C}.pt
          AND pytest tests/test_reward_hacking.py -q -> all green.
          Until fixtures land: this checkbox is DEPENDS-ON; existing
          test_reward_hacking.py must still pass as the baseline for
          5.2.1–5.2.3 behavior.
[ ] 5.2.5 `independent_eval.py` imports no symbol from autoqec.runner.*
          (spec §10 CI guard); cross-check:
          `grep -E "^(from|import)\s+autoqec\.runner" \
          autoqec/eval/independent_eval.py` returns empty
```

### 6.3 `autoqec review-log`

**`DEPENDS-ON`:** the current `review-log` subcommand prints a JSON stats
blob to stdout and does not write a `review.md`. Either the command is
extended to also write `review.md`, or the plan adapts to the JSON
contract below.

```
[ ] 5.3.1 ./.venv/bin/python -m cli.autoqec review-log $RUN_DIR
          # exit 0; prints a JSON blob to stdout
[ ] 5.3.2 the JSON has exactly these keys: n_rounds, n_pareto,
          n_killed_by_safety, mean_wallclock_s, top_hypotheses.
          n_rounds matches `wc -l $RUN_DIR/history.jsonl`.
[ ] 5.3.3 on a synthetic stuck run (three consecutive delta_ler in
          [-0.002, 0.002] rounds injected into history.jsonl), a future
          `review.md`-writing version of the skill must call out the
          pattern. DEPENDS-ON review-log emitting a written report;
          until then, the stats JSON alone is the gate.
```

### 6.4 `autoqec diagnose`

**`DEPENDS-ON`:** the current `diagnose` subcommand prints a JSON blob
(path, has_config_yaml, has_metrics_json, has_train_log, optional
metrics snapshot). It does not output a `diagnosis.md`. Either the
command is extended, or the plan adapts.

```
[ ] 5.4.1 inject three failure modes into separate run dirs:
          (a) oversized hidden_dim -> train.log ends with OOM
          (b) LR=10.0 -> train.log contains NaN
          (c) env yaml with p=0 -> metrics.json.delta_ler == 0.0 exactly
          for each, run ./.venv/bin/python -m cli.autoqec diagnose $RUN_DIR
          -> exits 0; prints JSON blob.
[ ] 5.4.2 the JSON's metrics snapshot agrees with the injected failure:
          (a) metrics.json absent OR status == "killed_by_safety"
          (b) metrics.json has nan in train.log, status == "killed_by_safety"
          (c) metrics.json.delta_ler == 0.0
[ ] 5.4.3 DEPENDS-ON enhancement: once `diagnose` writes diagnosis.md,
          add a `root_cause` field with fuzzy-match gates (OOM / NaN /
          degenerate) and a non-empty `recommended_fix`. Skill must not
          autonomously apply the fix (per spec §7.3).
```

### 6.5 Statistical correctness

```
[ ] 5.5.1 bootstrap CI unit test: known LER=0.01, N=200K, 1000 resamples
          -> 95% CI width < 0.002 (consistent with analytic
          2 * 1.96 * sqrt(p*(1-p)/N)); asserted in
          tests/test_independent_eval.py
[ ] 5.5.2 holdout-seed isolation: `training_seeds.isdisjoint(holdout_seeds)`
          asserted in code (env schema enforces this via the seed_policy
          model)
[ ] 5.5.3 paired eval: under a given paired_eval_bundle_id, the syndrome
          tensors used for plain_classical and predecoder+classical are
          byte-hash equal (DEPENDS-ON bundle_id being written)
[ ] 5.5.4 ablation sanity: a random-weights predecoder has delta_ler
          95% CI crossing 0 (no systematic positive bias in the evaluation
          protocol); asserted in tests/test_independent_eval.py
```

### 6.6 Pareto write atomicity (new; §9 missing failure mode 4)

```
[ ] 5.6.1 pareto.json is written via a tmp-file + rename (atomic on POSIX).
          Verified by grep of `os.replace` or `Path(...).replace` in
          `autoqec/orchestration/memory.py` around the pareto writer.
[ ] 5.6.2 pytest test that kills the writer mid-update (e.g. via
          monkeypatched `tmp.write` raising) leaves either the old
          pareto.json intact OR a detectable-tmp file, never a truncated
          JSON. If no such test exists: open issue
          `blocks-demo:pareto-atomicity`; this is a required gate.
[ ] 5.6.3 concurrent-writer guard: documentation in memory.py must
          explicitly state whether `RunMemory` is single-writer or uses
          an advisory lock. If single-writer, a startup check fails fast
          on a stale lock / running peer.
```

### 6.7 Advisor replay mode (new; §9 missing failure mode 5)

```
[ ] 5.7.1 advisor can replay a packaged run without network / LLM:
          `./.venv/bin/python -m cli.autoqec verify <round_dir> --env <yaml>
          --n-seeds 50` runs to completion with `AUTOQEC_*_BACKEND=offline`
          (or the env vars simply unset). No outbound HTTP.
[ ] 5.7.2 package format: the run_dir tarball (`runs/<run_id>.tar.gz`)
          plus the repo SHA in artifact_manifest.json is sufficient to
          replay. Verified by un-tarring to a fresh dir and re-running 5.7.1.
```

**Gate:**
- 5.1, 5.2, 5.5, 5.6, 5.7 must pass (research integrity is
  non-negotiable); 5.2's DEPENDS-ON paths fall back to the existing
  synthetic tests.
- 5.3 and 5.4 must each produce at least one live output (JSON blob
  until DEPENDS-ON enhancements land).
- Any trap in 5.2 that gets verdict == VERIFIED blocks the demo and
  `independent_eval.py` must be fixed before re-running.

---

## 7. Test Data

```
demos/demo-1-surface-d5/expected_output/baseline_benchmark.json   # LER=0.01394 anchor
tests/fixtures/reward_hacking/trap_{A,B,C}.pt                     # DEPENDS-ON fixtures
autoqec/example_db/gnn_{small,medium,gated}.yaml,
autoqec/example_db/neural_bp_{min,attn,per_check}.yaml,
autoqec/example_db/handshake_stub.yaml                            # Tier-1 seeds
autoqec/envs/builtin/{surface_d5_depol,bb72_depol}.yaml           # reference envs
holdout seeds: per env_yaml.noise.seed_policy.holdout (disjoint from training)
```

---

## 8. Open Questions (close before test plan execution)

```
[ ] O-1 CI runner: CPU only, or self-hosted GPU? Decides whether Phase 3/4
        can run in CI. Default assumption: local GPU + CI CPU only.
[ ] O-2 Phase 4 wall-clock budgets (75 min surface_d5 / 120 min bb72) and
        token budget (500K input / 80K output per env) are placeholders
        with no production evidence. Owner of the live-LLM DAG wiring
        (§1.6) should re-calibrate after first end-to-end dry-run.
[ ] O-3 Phase-5.2 trap fixture construction and Phase 5 owner
        assignments — deferred per current project consensus; to be
        picked up once fixtures are scheduled.
[ ] O-4 Advisor demo day: live run (re-run Phase 4) or replay (recorded
        Phase 4)? Affects the Phase 1 wall-clock gate for the advisor.
[ ] O-5 `diagnose` injected-failure fixtures (OOM / NaN / p=0): build now
        or defer to post-Day-3 regression suite?
[ ] O-6 `autoqec cleanup-worktree` CLI wrapper or keep Python-API-only?
        Affects 3.4.5.
```

---

## 9. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Live-LLM DAG wiring (§1.6 prereq) does not land before Day-3 demo | High | Phase 4 blocked | Advisor-ready narrative falls back to Phase 1+2+3 + Phase 5 on a Phase-3 candidate; gap is called out in the walkthrough, not hidden |
| Day-3 scope too large (per status §4.5) | High | Phase 4 yellow | Prioritize Phase 1+2+3 and single-env Phase 4 when live wiring lands |
| Reward-hack trap itself is buggy (rejects legitimate checkpoints) | Medium | Phase 5 all fails | Run each trap against the baseline checkpoint first; confirm verdict == VERIFIED for the clean case |
| LLM backend flaky (Codex CLI drops) | Medium | Phase 4 stuck | Fall back to claude-cli; plan documents both backend combinations as verified |
| Artifact manifest not written (§5.5) | Medium | Reproduction impossible | Open `blocks-demo:artifact-manifest`; manifest writer is a prerequisite |
| LLM containment not enforced (§5.6) | Medium | Silent contract drift | §4.6 git-diff check is the enforcement; runs that breach get rolled back |
| Pareto concurrent overwrite / crash-truncate | Low | Data loss | §6.6 atomicity check + single-writer assertion |
| `cli/autoqec.py::run` writes runs under `runs/<timestamp>/` with no user control | Medium | Tests that hard-code a path fail silently | Spec now reads `AUTOQEC_RESULT_JSON.run_dir` from stdout instead of setting `--run-dir` |

---

## 10. Success Criteria Summary

Aligned with §2–§6 gates exactly (previous draft's "at least one env
must pass" language is REMOVED — both envs must pass once Phase 4
is unblocked).

| Gate | Minimum pass | Blocks demo? |
|---|---|---|
| Phase 1 | 10/10 green | Yes |
| Phase 2 | >= 25/26 green (at most 1 quarantine) | Yes |
| Phase 3 | 3.1–3.4 all green; at most 1 quarantine in 3.5 | Yes |
| Phase 4 | If live wiring landed: 4.1 AND 4.2 both green; 4.3 at most 1 warning; 4.4+4.5+4.6 at most 2 quarantines. If not landed: blocked (not failed) — narrative shifts to Phase 3 candidate | Conditionally |
| Phase 5 | 5.1 + 5.2 + 5.5 + 5.6 + 5.7 must pass (5.2 DEPENDS-ON falls back to synthetic tests); 5.3 + 5.4 each produce one live output | Yes |

**Overall demo-ready verdict:** all five phase gates met
(with Phase 4's conditional rule) -> ship to advisor; otherwise block and
enumerate open items in a GitHub issue.

---

## 11. References

- `docs/superpowers/specs/2026-04-20-autoqec-design.md` — authoritative
  design (v2.3); source for §10 architecture, reward-hacking defenses,
  Pareto-as-non-dominated-set.
- `docs/contracts/interfaces.md` — pydantic schemas exercised by
  Phase 2.1 and Phase 5.1.
- `docs/contracts/round_dir_layout.md` — who writes what inside
  `runs/<run_id>/round_<N>/`; Phase 3.5.2 enforces its boundary rule.
- `docs/superpowers/specs/2026-04-22-ci-coverage-design.md` — engineering
  regression suite that complements this acceptance plan.
- `docs/status/2026-04-22-project-status.md` — Day-2 status snapshot;
  source for Risk list.
- GitHub issue `QuAIR/0420-FPGA-Decoder#3` — structural inspiration
  (phased acceptance checklist). This plan tightens the quantification
  that the FPGA plan left soft.
- Codex cross-model review (2026-04-23, GPT-5.4 via `codex exec`) —
  source of the v2 revision: factual corrections (H1–H6), schema
  alignment (M1–M6), and the 5 missing failure modes now woven into
  §4.4.5 / §4.5 / §4.6 / §6.6 / §6.7.
