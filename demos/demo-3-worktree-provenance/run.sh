#!/usr/bin/env bash
# Demo 3: worktree branches-as-Pareto provenance (issue #38).
#
# Runs one real research round inside an isolated git worktree / branch,
# shows branch + commit_sha + round_N_pointer.json, then cleans up the
# checkout while retaining the branch. No LLM required.
set -euo pipefail

# Resolve repo root regardless of whether this worktree lives under
# .worktrees/... or is a normal checkout.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# When REPO_ROOT is a git worktree (.worktrees/<id>/), the actual venv
# lives at the main checkout. Resolve it via --git-common-dir so the demo
# works both from the main checkout and from a nested worktree.
MAIN_REPO="$REPO_ROOT"
if COMMON_GIT_DIR="$(git -C "$REPO_ROOT" rev-parse --git-common-dir 2>/dev/null)"; then
    if [ -d "$COMMON_GIT_DIR" ]; then
        MAIN_REPO="$(cd "$COMMON_GIT_DIR/.." && pwd)"
    fi
fi

# Prefer the in-tree venv so user setup stays consistent with CLAUDE.md.
PY=""
for candidate in \
    "$REPO_ROOT/.venv/Scripts/python.exe" \
    "$REPO_ROOT/.venv/bin/python" \
    "$MAIN_REPO/.venv/Scripts/python.exe" \
    "$MAIN_REPO/.venv/bin/python"
do
    if [ -x "$candidate" ]; then
        PY="$candidate"
        break
    fi
done
if [ -z "$PY" ]; then
    PY="${PYTHON:-python}"
fi

cd "$REPO_ROOT"
"$PY" demos/demo-3-worktree-provenance/run_demo.py "$@"
