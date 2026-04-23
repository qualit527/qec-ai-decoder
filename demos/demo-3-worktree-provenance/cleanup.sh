#!/usr/bin/env bash
# Remove any leftover branches / checkouts created by this demo. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

# Clean up any dangling .worktrees/exp-demo38-* checkouts.
for wt in .worktrees/exp-demo38-*; do
    [ -e "$wt" ] || continue
    git worktree remove --force "$wt" || true
done
git worktree prune

# Delete any local exp/demo38-* branches. `git branch --list` uses a
# fnmatch pattern where `*` crosses `/`, unlike `git for-each-ref`.
git branch --list 'exp/demo38*' \
    | sed 's/^[* ]*//' \
    | xargs -r -n1 git branch -D

echo "demo-3 cleanup done."
