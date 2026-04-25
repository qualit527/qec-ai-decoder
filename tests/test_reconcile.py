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


def test_committed_orphan_without_pointer_emits_pause(repo_with_run: tuple[Path, Path]) -> None:
    """Per §15.10 Auto-heal vs pause: branch with real commits but no recoverable
    round_N_pointer.json must PAUSE for human review, not auto-quarantine.

    This was the pre-codex-review policy — the old tests asserted auto-rename
    to `quarantine/<run_id>/<remainder>` plus a synthetic history row. The
    spec §15.10 "Paused for human review" bullet explicitly covers this case.
    """
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Create a branch with a real commit BUT no round_N_pointer.json.
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

    # Must emit a pause action — the orchestrator, not reconcile, decides.
    paused = [a for a in actions if a["kind"] == "pause"]
    assert any(a.get("branch") == "exp/t/02-y" for a in paused), actions
    pause = next(a for a in paused if a.get("branch") == "exp/t/02-y")
    assert pause.get("reason") == "orphan_branch_without_pointer"

    # The branch MUST NOT be renamed to quarantine/*.
    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "exp/t/02-y" in branches
    assert "quarantine/t/02-y" not in branches

    # No history row must be written — the operator decides the disposition.
    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    orphan_rows = [r for r in rows if r.get("status") == "orphaned_branch"]
    assert orphan_rows == []


def test_committed_orphan_with_pointer_auto_heals(repo_with_run: tuple[Path, Path]) -> None:
    """§15.10 case 3b-with-recoverable-id: pointer JSON carries round_attempt_id
    → reconcile writes an auto-healed history row preserving that UUID.

    No branch rename, no fresh reconcile_id — the pointer commit is already
    canonical provenance.
    """
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    subprocess.run(
        ["git", "worktree", "add", ".worktrees/exp-t-04-with-ptr", "-b", "exp/t/04-with-ptr", "main"],
        cwd=repo,
        check=True,
    )
    wt = repo / ".worktrees" / "exp-t-04-with-ptr"
    # Write a proper round_N_pointer.json at round_04/.
    round_dir = wt / "round_04"
    round_dir.mkdir(parents=True)
    pointer = {
        "run_id": "t",
        "round_idx": 4,
        "round_attempt_id": "orig-uuid-abc-123",
        "branch": "exp/t/04-with-ptr",
        "commit_sha": "will-be-replaced",
        "fork_from": "baseline",
    }
    (round_dir / "round_04_pointer.json").write_text(json.dumps(pointer))
    subprocess.run(["git", "add", "."], cwd=wt, check=True)
    subprocess.run(["git", "commit", "-m", "round 4 pointer"], cwd=wt, check=True)
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=wt, text=True
    ).strip()
    subprocess.run(
        ["git", "worktree", "remove", "--force", ".worktrees/exp-t-04-with-ptr"],
        cwd=repo,
        check=True,
    )

    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)

    # Reconcile reaps (auto-heal), does NOT pause.
    paused = [a for a in actions if a["kind"] == "pause"]
    assert paused == []
    reaped = [a for a in actions if a["kind"] == "reaped"]
    assert any(
        a.get("branch") == "exp/t/04-with-ptr"
        and a.get("round_attempt_id") == "orig-uuid-abc-123"
        for a in reaped
    ), actions

    # Branch is NOT renamed.
    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "exp/t/04-with-ptr" in branches
    assert "quarantine/t/04-with-ptr" not in branches

    # A history row is written preserving the original round_attempt_id.
    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    orphan_rows = [r for r in rows if r.get("status") == "orphaned_branch"]
    assert len(orphan_rows) == 1
    assert orphan_rows[0]["round_attempt_id"] == "orig-uuid-abc-123"
    assert orphan_rows[0]["branch"] == "exp/t/04-with-ptr"
    assert orphan_rows[0]["commit_sha"] == commit_sha


def test_committed_orphan_with_malformed_pointer_emits_pause(repo_with_run: tuple[Path, Path]) -> None:
    """§15.10: malformed pointer JSON is equivalent to no pointer — pause."""
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    subprocess.run(
        ["git", "worktree", "add", ".worktrees/exp-t-05-bad-ptr", "-b", "exp/t/05-bad-ptr", "main"],
        cwd=repo,
        check=True,
    )
    wt = repo / ".worktrees" / "exp-t-05-bad-ptr"
    round_dir = wt / "round_05"
    round_dir.mkdir(parents=True)
    (round_dir / "round_05_pointer.json").write_text("{not valid json")
    subprocess.run(["git", "add", "."], cwd=wt, check=True)
    subprocess.run(["git", "commit", "-m", "bad pointer"], cwd=wt, check=True)
    subprocess.run(
        ["git", "worktree", "remove", "--force", ".worktrees/exp-t-05-bad-ptr"],
        cwd=repo,
        check=True,
    )

    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    paused = [a for a in actions if a["kind"] == "pause"]
    assert any(a.get("branch") == "exp/t/05-bad-ptr" for a in paused), actions

    # No rename, no history row.
    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "exp/t/05-bad-ptr" in branches
    assert "quarantine/t/05-bad-ptr" not in branches
    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [r for r in rows if r.get("status") == "orphaned_branch"] == []


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


def test_pareto_entry_with_unreachable_commit_emits_pause(
    repo_with_run: tuple[Path, Path],
) -> None:
    """§15.10 both-missing: a Pareto entry points at a commit that is neither
    on a live exp/ branch nor otherwise reachable — reconcile emits `pause`.
    """
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Write a Pareto entry with a fabricated 40-char SHA that resolves to nothing.
    pareto = [
        {
            "branch": "exp/t/11-missing",
            "commit_sha": "0" * 40,
            "delta_vs_baseline_holdout": 0.01,
            "flops_per_syndrome": 1000,
            "n_params": 2000,
        }
    ]
    (run_dir / "pareto.json").write_text(json.dumps(pareto), encoding="utf-8")

    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    paused = [a for a in actions if a["kind"] == "pause"]
    assert any(
        a.get("branch") == "exp/t/11-missing" and "unreachable" in a.get("reason", "")
        for a in paused
    ), actions


def test_branch_manually_deleted_is_idempotent(repo_with_run: tuple[Path, Path]) -> None:
    """Running reconcile twice must not duplicate the branch_manually_deleted row.

    H \\ B is stable across restarts, so each reconcile run was appending an
    identical duplicate. Guard against that by scanning existing history first.
    """
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    row = {
        "status": "ok",
        "branch": "exp/t/09-gone",
        "round_attempt_id": "abc-999",
        "commit_sha": "deadbeef",
    }
    (run_dir / "history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    # First run — emits follow-up.
    actions1 = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    assert [a for a in actions1 if a["kind"] == "follow_up"]

    # Second run — must NOT emit another follow-up, and must NOT append
    # another branch_manually_deleted row.
    actions2 = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    assert [a for a in actions2 if a["kind"] == "follow_up"] == []

    rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    deleted_rows = [r for r in rows if r.get("status") == "branch_manually_deleted"]
    assert len(deleted_rows) == 1


def test_branch_manually_deleted_is_idempotent_across_restarts(
    repo_with_run: tuple[Path, Path],
) -> None:
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    original_row = {
        "status": "ok",
        "branch": "exp/t/10-deleted-before-restart",
        "round_attempt_id": "abc-restart-123",
        "commit_sha": "deadbeef",
    }
    (run_dir / "history.jsonl").write_text(
        json.dumps(original_row) + "\n",
        encoding="utf-8",
    )

    first_startup_actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    rows_after_first_startup = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    second_startup_actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    rows_after_second_startup = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert [a for a in first_startup_actions if a["kind"] == "follow_up"] == [
        {"kind": "follow_up", "branch": "exp/t/10-deleted-before-restart"}
    ]
    assert [a for a in second_startup_actions if a["kind"] == "follow_up"] == []
    assert rows_after_second_startup == rows_after_first_startup
    deleted_rows = [
        row for row in rows_after_second_startup if row.get("status") == "branch_manually_deleted"
    ]
    assert len(deleted_rows) == 1
