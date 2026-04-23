# Surface D5 Demo Issue 36 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the Demo 1 `surface_d5_depol` no-LLM path documentation and script contract in line with the current CLI output so issue #36 has a focused, reviewable PR.

**Architecture:** Add tests that pin the demo script and README to the current machine-readable CLI contract, then update the shell script and README. The no-LLM path remains an unverified candidate path and should report `candidate_pareto.json`, not `pareto.json`.

**Tech Stack:** Bash, Python 3.12, pytest, GitHub CLI

---

### Task 1: Demo Contract Tests

**Files:**
- Create: `tests/test_surface_demo_contract.py`
- Modify: `demos/demo-1-surface-d5/run_quick.sh`
- Modify: `demos/demo-1-surface-d5/README.md`

- [ ] Write tests that assert the Demo 1 script parses `AUTOQEC_RESULT_JSON`, prints an authoritative run dir, and reports `candidate_pareto.json`.
- [ ] Write tests that assert the README documents no-LLM and live paths, candidate Pareto, unverified status, expected output snapshots, and CPU/GPU runtime guidance.
- [ ] Run the new tests and confirm they fail against the current script/README.
- [ ] Update the script and README with the minimal changes to satisfy the contract.
- [ ] Re-run the new tests until green.

### Task 2: Verification and PR

**Files:**
- Test: `tests/test_surface_demo_contract.py`
- Test: `tests/test_surface_assets.py`
- Test: `tests/test_cli_run_paths.py`

- [ ] Run `pytest tests/test_surface_demo_contract.py tests/test_surface_assets.py -v`.
- [ ] Run `pytest tests/test_cli_run_paths.py -q`.
- [ ] Run `make lint`.
- [ ] Commit, push branch `demo/surface-d5-36`, open PR, and comment the PR link on issue #36.
