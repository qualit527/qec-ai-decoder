"""§15.10 startup reconciliation.

Auto-heal empty synthetic branches and committed orphans whose
``round_N_pointer.json`` carries a recoverable ``round_attempt_id``;
pause for human review whenever the pointer is missing, malformed, or
when a Pareto commit is unreachable.

Policy summary:
  * Let ``B`` = set of ``exp/<run_id>/*`` branches currently in git.
  * Let ``H`` = set of ``branch`` values appearing in ``history.jsonl`` rows.
  * ``B \\ H`` — branches with no history row:
      - If the branch tip equals ``merge-base(branch, main)`` (empty synthetic),
        delete it silently. Emit ``{"kind": "reaped"}``.
      - Otherwise the branch has real commits. Try to read
        ``round_N_pointer.json`` from the branch tip. If it has a
        recoverable ``round_attempt_id``, auto-heal: write an
        ``orphaned_branch`` history row preserving the UUID, emit
        ``{"kind": "reaped", "source": "pointer"}``. **Do not rename
        the branch** — the pointer commit is canonical provenance.
      - If the pointer is absent or malformed, emit
        ``{"kind": "pause", "reason": "orphan_branch_without_pointer"}``
        and leave the branch untouched. The orchestrator decides.
  * ``H \\ B`` — history rows whose branch was manually deleted:
      - Append a ``branch_manually_deleted`` follow-up row (only once per
        branch across reconcile runs) and emit ``{"kind": "follow_up"}``.
  * Pareto-commit reachability check: every ``pareto.json`` entry's
    ``commit_sha`` must resolve via ``git rev-parse``. Otherwise emit
    ``{"kind": "pause", "reason": ...}`` — caller decides whether to halt.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any, Optional


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
    """Return the set of non-null ``branch`` values appearing in ``history.jsonl``.

    Excludes rows with ``status == "branch_manually_deleted"`` — those are
    reconcile-emitted follow-ups referring to a deleted branch, not evidence
    that the branch still ran a round.
    """
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
        if row.get("status") == "branch_manually_deleted":
            continue
        if row.get("branch"):
            branches.add(row["branch"])
    return branches


def _branches_with_deletion_marker(run_dir: Path) -> set[str]:
    """Return the set of branches that already have a ``branch_manually_deleted`` row."""
    path = run_dir / "history.jsonl"
    if not path.exists():
        return set()
    deleted: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "branch_manually_deleted" and row.get("branch"):
            deleted.add(row["branch"])
    return deleted


def _is_empty_synthetic(repo_root: Path, branch: str) -> bool:
    """A branch is empty-synthetic iff its tip equals its merge-base with main."""
    try:
        tip = _run_git(repo_root, "rev-parse", branch).strip()
        base = _run_git(repo_root, "merge-base", branch, "main").strip()
        return tip == base
    except subprocess.CalledProcessError:
        # If something's broken (e.g. main missing), don't treat as empty —
        # err on the side of pause so no real commits are touched.
        return False


_BRANCH_ROUND_RE = re.compile(r"^exp/[^/]+/(\d+)(?:-|$)")


def _try_read_pointer(repo_root: Path, branch: str) -> Optional[dict[str, Any]]:
    """Try to read and parse a ``round_N_pointer.json`` from the branch tip.

    Strategy:
      1. If the branch name matches ``exp/<run_id>/<NN>-<slug>``, infer
         ``N`` and try ``git show <branch>:round_<NN>/round_<NN>_pointer.json``
         and a zero-padded variant. Return the parsed dict on success.
      2. Fallback: list the branch tip's tree and look for any
         ``round_*_pointer.json`` file. Read and parse the first match.

    Returns
    -------
    dict | None
        The parsed pointer dict, or None on any failure (JSON decode
        error, missing file, git error).
    """
    # Infer round index from the branch name.
    candidates: list[str] = []
    match = _BRANCH_ROUND_RE.match(branch)
    if match:
        raw = match.group(1)
        padded = raw.zfill(2)
        # Try both padded and unpadded (the slug convention uses padded).
        for n in {raw, padded}:
            candidates.append(f"round_{n}/round_{n}_pointer.json")
            candidates.append(f"round_{n}_pointer.json")

    for rel in candidates:
        try:
            raw = _run_git(repo_root, "show", f"{branch}:{rel}")
        except subprocess.CalledProcessError:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data

    # Fallback: scan the whole tree for any round_*_pointer.json.
    try:
        tree = _run_git(repo_root, "ls-tree", "-r", "--name-only", branch)
    except subprocess.CalledProcessError:
        return None
    for path in tree.splitlines():
        name = path.strip()
        if not name:
            continue
        if not name.endswith("_pointer.json"):
            continue
        # Only accept names that look like "round_<digits>_pointer.json".
        leaf = name.rsplit("/", 1)[-1]
        if not re.match(r"^round_\d+_pointer\.json$", leaf):
            continue
        try:
            raw = _run_git(repo_root, "show", f"{branch}:{name}")
        except subprocess.CalledProcessError:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return data
    return None


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
            continue

        # Committed orphan — try to read round_N_pointer.json for auto-heal.
        pointer = _try_read_pointer(repo_root, branch)
        try:
            commit_sha = _run_git(repo_root, "rev-parse", branch).strip()
        except subprocess.CalledProcessError:
            commit_sha = None

        if pointer is not None and pointer.get("round_attempt_id"):
            # Case 3b-with-recoverable-id: auto-heal, no rename.
            row = {
                "status": "orphaned_branch",
                "round_attempt_id": pointer["round_attempt_id"],
                "branch": branch,
                "commit_sha": commit_sha,
                "status_reason": "auto-healed from pointer",
            }
            _append_history_row(run_dir, row)
            actions.append(
                {
                    "kind": "reaped",
                    "branch": branch,
                    "round_attempt_id": pointer["round_attempt_id"],
                    "source": "pointer",
                }
            )
        else:
            # Case 3b-without-recoverable-id OR malformed pointer: pause.
            actions.append(
                {
                    "kind": "pause",
                    "branch": branch,
                    "commit_sha": commit_sha,
                    "reason": "orphan_branch_without_pointer",
                }
            )

    # 4. History rows whose branch was manually deleted.
    # Idempotence: a branch already flagged once must not get a duplicate row
    # on every subsequent reconcile (the H \ B set is stable across restarts).
    already_flagged = _branches_with_deletion_marker(run_dir)
    for branch in sorted(branches_in_history - branches_in_git):
        if branch in already_flagged:
            continue
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
