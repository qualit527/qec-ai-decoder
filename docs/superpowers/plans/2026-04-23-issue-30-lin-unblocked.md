# Issue 30 Lin Unblocked Subset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a reviewable first PR for issue #30 that covers the unblocked Lin-owned checks around response validation, no-LLM CLI behavior, and `review-log` / `diagnose` observability without taking on live-LLM or artifact-manifest scope.

**Architecture:** Extend the existing unit and CLI tests first, prove the gaps with failing tests, then make minimal code changes in `autoqec/agents/dispatch.py` and `cli/autoqec.py`. Keep the scope limited to behaviors already exposed by the current CLI and schema contracts, and add only the smallest fixture data needed for diagnose-failure detection.

**Tech Stack:** Python 3.12, pytest, click, pydantic, GitHub CLI

---

### Task 1: Plan Baseline and Response-Validation Guard

**Files:**
- Modify: `tests/test_orchestration_stub.py`
- Modify: `autoqec/agents/dispatch.py`
- Test: `tests/test_orchestration_stub.py`

- [ ] **Step 1: Write the failing test for the required-field case**

Add a test beside the existing `parse_response` schema checks that removes the required `rationale` field from an Ideator payload and asserts `ValueError` is raised with the missing-field name in the message.

- [ ] **Step 2: Run the targeted test to verify it fails or proves the exact current behavior**

Run: `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_orchestration_stub.py -k rationale -v`

Expected: either the new test fails because the message is too weak, or it passes and confirms no production change is required for this checkbox.

- [ ] **Step 3: Make the minimal implementation change if needed**

If the message does not clearly name the missing field, adjust `autoqec/agents/dispatch.py::parse_response` so it still wraps `ValidationError` as `ValueError` but preserves a readable missing-field signal for `rationale`.

- [ ] **Step 4: Re-run the focused test**

Run: `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_orchestration_stub.py -k rationale -v`

Expected: PASS.


### Task 2: No-LLM CLI Contract Coverage

**Files:**
- Modify: `tests/test_cli_run_paths.py`
- Modify: `tests/test_cli_helpers.py`
- Modify: `cli/autoqec.py`
- Test: `tests/test_cli_run_paths.py`
- Test: `tests/test_cli_helpers.py`

- [ ] **Step 1: Write failing tests for the no-LLM run contract**

Add or tighten tests that prove:
- `AUTOQEC_RESULT_JSON=` output is parseable and the `run_dir` exists
- `history.jsonl` grows append-only by one line per round in the `--no-llm` path
- `candidate_pareto.json` stays non-dominated and matches the path reported in the machine-readable payload

- [ ] **Step 2: Run the no-LLM CLI tests and observe failures**

Run: `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_cli_run_paths.py tests/test_cli_helpers.py -k 'result_prefix or candidate_pareto or history or no_llm' -v`

Expected: one or more failures describing the contract gap.

- [ ] **Step 3: Implement the minimal CLI changes**

Update `cli/autoqec.py` only as needed to make the machine-readable payload stable, preserve append-only `history.jsonl` behavior, and keep `candidate_pareto.json` aligned with the non-dominated-set helper.

- [ ] **Step 4: Re-run the focused CLI tests**

Run: `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_cli_run_paths.py tests/test_cli_helpers.py -k 'result_prefix or candidate_pareto or history or no_llm' -v`

Expected: PASS.


### Task 3: `review-log` and `diagnose` Structured Observability

**Files:**
- Modify: `tests/test_cli_helpers.py`
- Create: `tests/fixtures/diagnose/oom/metrics.json`
- Create: `tests/fixtures/diagnose/oom/train.log`
- Create: `tests/fixtures/diagnose/nan/metrics.json`
- Create: `tests/fixtures/diagnose/nan/train.log`
- Create: `tests/fixtures/diagnose/degenerate_p_zero/metrics.json`
- Create: `tests/fixtures/diagnose/degenerate_p_zero/train.log`
- Modify: `cli/autoqec.py`
- Test: `tests/test_cli_helpers.py`

- [ ] **Step 1: Add failing tests for `review-log` JSON shape**

Extend `tests/test_cli_helpers.py` to assert that `review-log` returns:
- `n_rounds`
- `n_pareto`
- `n_killed_by_safety`
- `mean_wallclock_s`
- `top_hypotheses`

Also assert `n_rounds == len(history.jsonl lines)`.

- [ ] **Step 2: Add failing fixture-driven tests for `diagnose`**

Create three lightweight diagnose fixtures for:
- OOM signature
- NaN-loss signature
- degenerate `p = 0` signature

Add tests asserting `python -m cli.autoqec diagnose <run_dir>` exits 0, identifies the signature, and does not claim to apply a fix.

- [ ] **Step 3: Run the focused observability tests to verify red**

Run: `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_cli_helpers.py -k 'review_log or diagnose' -v`

Expected: FAIL on missing fields or incomplete diagnosis output.

- [ ] **Step 4: Implement the minimal CLI changes**

Update `cli/autoqec.py` so:
- `review-log` emits the required JSON keys with deterministic counts
- `diagnose` emits structured signal text/fields for OOM, NaN, and degenerate-`p` fixtures
- `diagnose` output remains read-only and never claims to have applied a fix

- [ ] **Step 5: Re-run the focused observability tests**

Run: `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_cli_helpers.py -k 'review_log or diagnose' -v`

Expected: PASS.


### Task 4: Final Verification, Commit, and PR

**Files:**
- Modify: `docs/superpowers/plans/2026-04-23-issue-30-lin-unblocked.md`
- Test: `tests/test_orchestration_stub.py`
- Test: `tests/test_cli_run_paths.py`
- Test: `tests/test_cli_helpers.py`

- [ ] **Step 1: Run the complete verification set for this PR**

Run:
- `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_orchestration_stub.py tests/test_cli_run_paths.py tests/test_cli_helpers.py -v`
- `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_backend_adapter.py tests/test_pymatching_baseline.py tests/test_bposd_baseline.py -q`
- `/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_e2e_handshake.py tests/test_runner_guard.py -v`
- `make lint`
- `make test`

Expected: all commands exit 0.

- [ ] **Step 2: Commit the branch**

Run:
- `git add autoqec/agents/dispatch.py cli/autoqec.py tests/test_orchestration_stub.py tests/test_cli_run_paths.py tests/test_cli_helpers.py tests/fixtures/diagnose docs/superpowers/plans/2026-04-23-issue-30-lin-unblocked.md`
- `git commit -m "test: cover issue 30 no-llm and diagnose contracts"`

- [ ] **Step 3: Push and open the PR**

Run:
- `git push -u origin test/issue-30-lin-unblocked`
- `gh pr create --base main --head test/issue-30-lin-unblocked --title "test: cover issue 30 no-llm and diagnose contracts" --body "<summary>"`

Expected: PR URL returned.

- [ ] **Step 4: Comment on issue #30 with the PR link and covered checklist IDs**

Run:
- `gh issue comment 30 --repo qualit527/qec-ai-decoder --body "<PR link and covered checklist IDs>"`

Expected: issue comment URL returned.
