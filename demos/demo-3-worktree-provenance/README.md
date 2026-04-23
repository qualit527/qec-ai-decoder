# Demo 3 — Worktree branches-as-Pareto provenance

Closes [#38](https://github.com/qualit527/qec-ai-decoder/issues/38).

## What this demo shows

Every AutoQEC research round runs inside its own `git worktree` on a
deterministic branch `exp/<run_id>/<NN>-<slug>`, commits a
`round_<N>/round_<N>_pointer.json` provenance blob, and then releases
the checkout while keeping the branch. Compose rounds `git merge` two
parent branches as a **first-class scientific probe** — conflict is a
recorded failure mode (`status="compose_conflict"`), not an exception.

The demo walks through the full §15 path end-to-end in one script, with
**no live LLM**:

```
Round 1 — Idea A forks from baseline
Round 2 — Idea B forks from baseline (independent)
Round 3 — create_compose_worktree(parents=[A, B]) merges the ideas
          and trains a compose-round on the merged checkout
Step 5 — a deliberately-conflicting pair of branches, attempted merge,
          status=compose_conflict with conflicting_files reported
Step 6 — git log --graph of all exp/<run_id>/* branches side-by-side
```

Each worktree's full contents are listed before and after `run-round`
so you can see `round_N/round_N_pointer.json` appear on-disk, and the
compose step prints `git log --graph` showing the merge commit as a
diamond.

## Run

```bash
bash demos/demo-3-worktree-provenance/run.sh
# or with a custom run id:
bash demos/demo-3-worktree-provenance/run.sh --run-id demo38-sample
# skip the conflict probe (saves ~2s):
bash demos/demo-3-worktree-provenance/run.sh --skip-conflict-demo
```

Runtime: ~30 s on a laptop CPU (three `run-round` executions @ ~5 s
each, plus the git-merge plumbing). No network, no LLM, no GPU.

## Acceptance criteria (issue #38)

| Criterion | Covered by |
|---|---|
| `run-round` with `--code-cwd`, `--branch`, `--fork-from`, `--round-attempt-id` exits 0 | `_run_round()` in `run_demo.py`; runs three times (Round 1, 2, 3) and asserts status="ok" |
| `metrics.json.branch` populated, `commit_sha` is a real git commit | Every `_run_round` call asserts both fields; the `git log --graph` in Step 4 shows the commit for compose-round |
| `round_1_pointer.json` exists and round-trips via `git show <branch>:...` | Step 4 explicitly lists `round_1/round_1_pointer.json` **and** `round_2/round_2_pointer.json` inside the merged compose worktree, proving both commits landed |
| `cleanup_round_worktree(...)` removes the checkout, keeps the branch | After each round; Step 3 prints the surviving branch list to confirm |
| `reconcile_at_startup(...)` behavior demonstrated or linked to a passing test | Step 7 + `tests/test_reconcile.py::test_orphaned_branch_with_pointer_autoheals` |
| `create_compose_worktree` path exercised | Step 4 (happy merge) + Step 5 (deliberate conflict caught as `compose_conflict`) |

## Expected output snapshot

`expected_output/` holds sanitized artifacts from a reference run so
reviewers can audit the provenance without re-executing:

- `round_1_metrics.json`, `round_2_metrics.json`, `round_3_metrics.json`
  — `RoundMetrics` rows from each stage. Round 3's `fork_from` is a
  two-element list, `compose_mode="pure"` — this is the compose-round
  signature.
- `metrics.json` — alias for `round_1_metrics.json` (kept for continuity).
- `round_1_pointer.json` — blob committed onto `exp/demo38-sample/01-idea-a`,
  the `§15.10` reconcile input.
- `run_demo.stdout.txt` — full walkthrough including the worktree `ls`
  listings and `git log --graph` outputs.

Paths are replaced with `<REPO_ROOT>` / `<VENV>`. Commit SHAs and
UUIDs are kept verbatim to demonstrate the provenance chain.

To refresh (after editing `run_demo.py` or bumping the fixture config):

```bash
bash demos/demo-3-worktree-provenance/run.sh --run-id demo38-sample > walk.txt
python demos/demo-3-worktree-provenance/_refresh_expected.py walk.txt
```

## Cleanup

```bash
bash demos/demo-3-worktree-provenance/cleanup.sh            # default prefix: demo38
bash demos/demo-3-worktree-provenance/cleanup.sh my-run-id  # custom run_id
```

Removes any `.worktrees/exp-<prefix>*` checkouts and deletes every
`exp/<prefix>*` local branch. Safe to run anytime.
