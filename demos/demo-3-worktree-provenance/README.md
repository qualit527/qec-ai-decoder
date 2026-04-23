# Demo 3 — Worktree branches-as-Pareto provenance

Closes [#38](https://github.com/qualit527/qec-ai-decoder/issues/38).

## What this demo shows

Every AutoQEC research round runs inside its own `git worktree` on a
deterministic branch `exp/<run_id>/<NN>-<slug>`, commits a
`round_<N>/round_<N>_pointer.json` provenance blob, and then releases
the checkout while keeping the branch. This is why `pareto.json`
entries are `git rev-parse`-able commits, not just rows under
`runs/<run_id>/`.

The demo exercises the full §15 path end-to-end with **no live LLM**:

```
Ideator fork decision (simulated by a YAML fixture)
   -> autoqec.orchestration.worktree.create_round_worktree
   -> cli.autoqec run-round --code-cwd ... --branch ... --fork-from ... \
                            --round-attempt-id <uuid>
   -> autoqec.orchestration.subprocess_runner.run_round_in_subprocess
        * runs a fresh python in worktree cwd
        * reads metrics from child stdout
        * writes + commits round_<N>/round_<N>_pointer.json on the branch
   -> inspect runs/<run_id>/round_1/metrics.json (branch + commit_sha)
   -> inspect git show <branch>:round_1/round_1_pointer.json
   -> cleanup_round_worktree  -> removes the checkout, keeps the branch
   -> (reconcile_at_startup would auto-heal from the pointer after a crash)
```

## Run

```bash
bash demos/demo-3-worktree-provenance/run.sh
# or with explicit flags:
bash demos/demo-3-worktree-provenance/run.sh --slug hello --keep-worktree
```

Runtime: ~30–90 s on a laptop CPU, depending on how fast surface_d5
syndrome sampling finishes under `profile=dev` (capped at 256 training
shots, 64 val shots, 1 epoch — see `fixture_config.yaml`).

No network, no LLM, no GPU required.

## Acceptance criteria (issue #38)

| Criterion                                                                                                      | Verified by                                                                              |
|----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `run-round` with `--code-cwd`, `--branch`, `--fork-from baseline`, `--round-attempt-id` exits 0                | Step 3 of `run_demo.py`; non-zero exit surfaces as a failing assertion                   |
| `metrics.json.branch` populated, `commit_sha` is a real git commit                                             | Step 4; asserted against `plan["branch"]` and the HEAD written by `_write_and_commit_pointer` |
| `round_1_pointer.json` exists and round-trips via `git show <branch>:...`                                      | Step 5; runs `git show <branch>:round_1/round_1_pointer.json` and validates the payload  |
| `cleanup_round_worktree(...)` removes the checkout but leaves the branch                                       | Step 7; asserts `git branch --list` still returns the branch while `git worktree list` does not |
| `reconcile_at_startup(...)` behavior demonstrated or linked to a passing test                                  | Step 8 text plus `tests/test_reconcile.py::test_orphaned_branch_with_pointer_autoheals`  |

The `tests/test_run_round_pointer_integration.py` integration test
exercises the same path under `pytest -m integration`, so the demo
stays in sync with production code paths.

## After the presentation

```bash
bash demos/demo-3-worktree-provenance/cleanup.sh
```

Removes any `.worktrees/exp-demo38-*` leftover checkouts and deletes
every `exp/demo38-*` local branch. Safe to run anytime.

## Expected output snapshot

`expected_output/` contains a committed sample run so reviewers can
eyeball the provenance without re-executing the demo:

- `metrics.json` — `branch`, `commit_sha`, `round_attempt_id`, `fork_from`, `delta_ler`
- `round_1_pointer.json` — the blob committed onto the experiment branch
- `run_demo.stdout.txt` — the full narrative walkthrough

These are refreshed by passing `--run-id demo38-sample` to `run_demo.py`
and copying the three artifacts in; see the last section of the demo
script for how the walkthrough is emitted.
