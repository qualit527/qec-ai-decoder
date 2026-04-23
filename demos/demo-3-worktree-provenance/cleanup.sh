#!/usr/bin/env bash
# Remove any leftover branches / checkouts created by this demo. Idempotent.
#
# Usage:
#   bash demos/demo-3-worktree-provenance/cleanup.sh            # default: demo38
#   bash demos/demo-3-worktree-provenance/cleanup.sh my-run-id  # custom run_id
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

PREFIX="${1:-demo38}"

# Clean up any dangling .worktrees/exp-${PREFIX}* checkouts.
for wt in .worktrees/exp-${PREFIX}*; do
    [ -e "$wt" ] || continue
    git worktree remove --force "$wt" || true
done
git worktree prune

# Delete any local exp/${PREFIX}* branches. `git branch --list` uses a
# fnmatch pattern where `*` crosses `/`, unlike `git for-each-ref`.
git branch --list "exp/${PREFIX}*" \
    | sed 's/^[* ]*//' \
    | xargs -r -n1 git branch -D

echo "demo-3 cleanup done (prefix=${PREFIX})."
