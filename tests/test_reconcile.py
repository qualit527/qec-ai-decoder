"""§15.10 — startup reconciliation of git-branch state vs `history.jsonl`.

Verifies:
  * empty synthetic exp/ branches (no commits beyond main) are auto-reaped
  * exp/ branches with real commits but no history row are quarantined
    and an `orphaned_branch` history row is appended
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo_with_run(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny git repo + run-dir scaffolding for reconciliation tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "baseline.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    run_dir = repo / "runs" / "t"
    run_dir.mkdir(parents=True)
    (run_dir / "history.jsonl").write_text("")
    (run_dir / "pareto.json").write_text("[]")
    return repo, run_dir


def test_empty_synthetic_branch_is_reaped(repo_with_run: tuple[Path, Path]) -> None:
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Create an empty synthetic branch whose tip == main (no new commits).
    subprocess.run(
        ["git", "worktree", "add", ".worktrees/exp-t-01-x", "-b", "exp/t/01-x", "main"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "worktree", "remove", "--force", ".worktrees/exp-t-01-x"],
        cwd=repo,
        check=True,
    )
    # Branch exists in git; history is empty → B \\ H = {exp/t/01-x} and it is empty.
    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    reaped = [a for a in actions if a["kind"] == "reaped"]
    assert any("exp/t/01-x" in a["branch"] for a in reaped)
    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "exp/t/01-x" not in branches


def test_committed_orphan_is_quarantined(repo_with_run: tuple[Path, Path]) -> None:
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Create a branch with a real commit — simulates a crash after Runner wrote a commit.
    subprocess.run(
        ["git", "worktree", "add", ".worktrees/exp-t-02-y", "-b", "exp/t/02-y", "main"],
        cwd=repo,
        check=True,
    )
    wt = repo / ".worktrees" / "exp-t-02-y"
    (wt / "new.txt").write_text("work in progress\n")
    subprocess.run(["git", "add", "."], cwd=wt, check=True)
    subprocess.run(["git", "commit", "-m", "wip"], cwd=wt, check=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", ".worktrees/exp-t-02-y"],
        cwd=repo,
        check=True,
    )

    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    quarantined = [a for a in actions if a["kind"] == "quarantined"]
    assert any("exp/t/02-y" in a["original_branch"] for a in quarantined)

    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "quarantine/t/02-y" in branches

    # An orphaned_branch history row must be appended.
    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    orphan_rows = [r for r in rows if r.get("status") == "orphaned_branch"]
    assert orphan_rows, "expected an orphaned_branch history row"
    assert orphan_rows[0]["branch"] == "quarantine/t/02-y"
    assert orphan_rows[0].get("commit_sha")


def test_history_row_without_live_branch_emits_followup(repo_with_run: tuple[Path, Path]) -> None:
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Seed history.jsonl with a row whose branch doesn't exist in git.
    row = {
        "status": "ok",
        "branch": "exp/t/03-z",
        "round_attempt_id": "abc-123",
        "commit_sha": "deadbeef",
    }
    (run_dir / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    follow_ups = [a for a in actions if a["kind"] == "follow_up"]
    assert any(a["branch"] == "exp/t/03-z" for a in follow_ups)

    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    deleted_rows = [r for r in rows if r.get("status") == "branch_manually_deleted"]
    assert deleted_rows, "expected a branch_manually_deleted follow-up row"
    assert deleted_rows[0]["branch"] == "exp/t/03-z"


def test_reconcile_no_op_on_clean_repo(repo_with_run: tuple[Path, Path]) -> None:
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    assert actions == []
