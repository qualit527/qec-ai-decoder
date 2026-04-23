"""Git worktree helpers for AutoQEC research rounds (§15.6).

Ported from problem-reductions/scripts/pipeline_worktree.py (MIT License) with
field renames (issue -> round) and a new ``create_compose_worktree`` that
handles §15.6 compose rounds with conflict-as-failure semantics.
"""
# Ported from problem-reductions/scripts/pipeline_worktree.py (MIT License)
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")[:40] or "round"


def _run_git(cwd: str | Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True)


def _run_git_checked(cwd: str | Path, *args: str) -> None:
    subprocess.check_output(["git", "-C", str(cwd), *args], stderr=subprocess.STDOUT)


def plan_round_worktree(
    repo_root: str | Path,
    *,
    run_id: str,
    round_idx: int,
    slug: str,
) -> dict:
    """Return the deterministic (branch, worktree_dir) pair without creating anything."""
    slug = _sanitize(slug)
    branch = f"exp/{run_id}/{round_idx:02d}-{slug}"
    worktree_dir = Path(repo_root) / ".worktrees" / f"exp-{run_id}-{round_idx:02d}-{slug}"
    return {
        "run_id": run_id,
        "round_idx": round_idx,
        "slug": slug,
        "branch": branch,
        "worktree_dir": str(worktree_dir),
    }


def create_round_worktree(
    repo_root: str | Path,
    run_id: str,
    round_idx: int,
    slug: str,
    fork_from: str = "main",
) -> dict:
    """Create ``.worktrees/exp-<run_id>-<N>-<slug>/`` checked out on a new branch."""
    plan = plan_round_worktree(repo_root, run_id=run_id, round_idx=round_idx, slug=slug)
    Path(plan["worktree_dir"]).parent.mkdir(parents=True, exist_ok=True)
    _run_git_checked(
        repo_root,
        "worktree",
        "add",
        plan["worktree_dir"],
        "-b",
        plan["branch"],
        fork_from,
    )
    plan["fork_from"] = fork_from
    return plan


def create_compose_worktree(
    repo_root: str | Path,
    run_id: str,
    round_idx: int,
    slug: str,
    parents: list[str],
) -> dict:
    """Create a worktree from ``parents[0]`` then ``git merge parents[1:]``.

    On conflict: abort merge, remove the worktree, delete the branch, and return
    ``status=compose_conflict`` with the list of conflicting files per §15.6.3.
    """
    if len(parents) < 2:
        raise ValueError(f"compose requires >=2 parents, got {parents}")

    plan = plan_round_worktree(repo_root, run_id=run_id, round_idx=round_idx, slug=slug)
    plan["parents"] = list(parents)
    plan["fork_from_ordered"] = list(parents)
    plan["fork_from_canonical"] = "|".join(sorted(parents))

    Path(plan["worktree_dir"]).parent.mkdir(parents=True, exist_ok=True)
    _run_git_checked(
        repo_root,
        "worktree",
        "add",
        plan["worktree_dir"],
        "-b",
        plan["branch"],
        parents[0],
    )

    for parent in parents[1:]:
        proc = subprocess.run(
            ["git", "-C", plan["worktree_dir"], "merge", parent, "--no-edit"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            conflicting = _run_git(
                plan["worktree_dir"], "diff", "--name-only", "--diff-filter=U"
            ).split()
            subprocess.run(
                ["git", "-C", plan["worktree_dir"], "merge", "--abort"],
                capture_output=True,
            )
            _run_git_checked(
                repo_root, "worktree", "remove", "--force", plan["worktree_dir"]
            )
            _run_git_checked(repo_root, "branch", "-D", plan["branch"])
            plan.update(status="compose_conflict", conflicting_files=conflicting)
            return plan

    plan["status"] = "ok"
    return plan


def cleanup_round_worktree(
    repo_root: str | Path,
    worktree_dir: str | Path,
) -> None:
    """Remove the checkout; keep the branch. Idempotent."""
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_dir)],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "prune"],
        capture_output=True,
    )
