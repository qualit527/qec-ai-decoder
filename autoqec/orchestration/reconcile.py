"""§15.10 startup reconciliation.

Auto-heal empty synthetic branches; quarantine branches with real commits but
no history row. Pause only for ambiguous cases (no recoverable round_attempt_id,
or missing Pareto commit).

Policy summary:
  * Let ``B`` = set of ``exp/<run_id>/*`` branches currently in git.
  * Let ``H`` = set of ``branch`` values appearing in ``history.jsonl`` rows.
  * ``B \\ H`` — branches with no history row:
      - If the branch tip equals ``merge-base(branch, main)`` (empty synthetic),
        delete it silently. Emit ``{"kind": "reaped"}``.
      - Otherwise the branch has real commits; rename it to
        ``quarantine/<run_id>/<remainder>``, append an ``orphaned_branch``
        history row, and emit ``{"kind": "quarantined"}``.
  * ``H \\ B`` — history rows whose branch was manually deleted:
      - Append a ``branch_manually_deleted`` follow-up row and emit
        ``{"kind": "follow_up"}``.
  * Pareto-commit reachability check: every ``pareto.json`` entry's
    ``commit_sha`` must resolve via ``git rev-parse``. Otherwise emit
    ``{"kind": "pause", "reason": ...}`` — caller decides whether to halt.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any


def _run_git(cwd: str | Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True)


def _run_git_checked(cwd: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _list_exp_branches(repo_root: Path, run_id: str) -> set[str]:
    """Return the set of local branches matching ``exp/<run_id>/*``."""
    try:
        out = _run_git(repo_root, "branch", "--list", f"exp/{run_id}/*")
    except subprocess.CalledProcessError:
        return set()
    branches: set[str] = set()
    for line in out.splitlines():
        name = line.strip().lstrip("* ").strip()
        if name:
            branches.add(name)
    return branches


def _history_branches(run_dir: Path) -> set[str]:
    """Return the set of non-null ``branch`` values appearing in ``history.jsonl``."""
    path = run_dir / "history.jsonl"
    if not path.exists():
        return set()
    branches: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("branch"):
            branches.add(row["branch"])
    return branches


def _is_empty_synthetic(repo_root: Path, branch: str) -> bool:
    """A branch is empty-synthetic iff its tip equals its merge-base with main."""
    try:
        tip = _run_git(repo_root, "rev-parse", branch).strip()
        base = _run_git(repo_root, "merge-base", branch, "main").strip()
        return tip == base
    except subprocess.CalledProcessError:
        # If something's broken (e.g. main missing), don't treat as empty —
        # err on the side of quarantine so no real commits are discarded.
        return False


def _append_history_row(run_dir: Path, row: dict[str, Any]) -> None:
    with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pareto_entries(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "pareto.json"
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").strip() or "[]"
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return entries if isinstance(entries, list) else []


def reconcile_at_startup(
    repo_root: str | Path,
    run_id: str,
    run_dir: str | Path,
) -> list[dict[str, Any]]:
    """Reconcile git-branch state vs ``history.jsonl`` per §15.10.

    Parameters
    ----------
    repo_root:
        Path to the git repository root (contains ``.git/``).
    run_id:
        Run identifier whose branches live under ``exp/<run_id>/``.
    run_dir:
        Directory containing ``history.jsonl`` and ``pareto.json``.

    Returns
    -------
    list[dict]
        A list of action records. Each has a ``kind`` key — one of
        ``"reaped"``, ``"quarantined"``, ``"follow_up"``, ``"pause"``.
    """
    repo_root = Path(repo_root)
    run_dir = Path(run_dir)
    actions: list[dict[str, Any]] = []

    # Clear stale worktree metadata so branch-prunes see fresh state.
    _run_git_checked(repo_root, "worktree", "prune")

    branches_in_git = _list_exp_branches(repo_root, run_id)
    branches_in_history = _history_branches(run_dir)

    # 3. Branches without history rows.
    for branch in sorted(branches_in_git - branches_in_history):
        if _is_empty_synthetic(repo_root, branch):
            _run_git_checked(repo_root, "branch", "-D", branch)
            actions.append({"kind": "reaped", "branch": branch})
        else:
            remainder = branch[len(f"exp/{run_id}/"):]
            new_branch = f"quarantine/{run_id}/{remainder}"
            _run_git_checked(repo_root, "branch", "-m", branch, new_branch)
            commit_sha = _run_git(repo_root, "rev-parse", new_branch).strip()
            actions.append(
                {
                    "kind": "quarantined",
                    "original_branch": branch,
                    "quarantine_branch": new_branch,
                    "commit_sha": commit_sha,
                }
            )
            row = {
                "status": "orphaned_branch",
                "round_attempt_id": None,
                "reconcile_id": str(uuid.uuid4()),
                "branch": new_branch,
                "commit_sha": commit_sha,
                "status_reason": (
                    f"branch {branch} had real commits but no history row at startup"
                ),
            }
            _append_history_row(run_dir, row)

    # 4. History rows whose branch was manually deleted.
    for branch in sorted(branches_in_history - branches_in_git):
        row = {
            "status": "branch_manually_deleted",
            "branch": branch,
            "round_attempt_id": None,
            # Follow-up marker — references the original row via the branch key
            # plus a deterministic reconcile_id derived from the branch name.
            "reconcile_id": f"followup-{branch.replace('/', '-')}",
            "status_reason": (
                f"branch {branch} was listed in history but not found in git at startup"
            ),
        }
        _append_history_row(run_dir, row)
        actions.append({"kind": "follow_up", "branch": branch})

    # 5. Pareto-commit reachability.
    for entry in _pareto_entries(run_dir):
        sha = entry.get("commit_sha")
        if not sha:
            continue
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{sha}^{{commit}}"],
            capture_output=True,
        )
        if proc.returncode != 0:
            actions.append(
                {
                    "kind": "pause",
                    "reason": f"Pareto commit {sha} unreachable",
                    "branch": entry.get("branch"),
                }
            )

    return actions
