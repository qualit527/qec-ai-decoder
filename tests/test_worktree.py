"""Tests for autoqec.orchestration.worktree helpers (§15.6).

Uses a real temporary git repo so every git subprocess exercised here
stays isolated from the outer repository.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "baseline.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    return repo


def test_create_round_worktree_returns_paths(git_repo):
    from autoqec.orchestration.worktree import create_round_worktree

    plan = create_round_worktree(
        repo_root=git_repo,
        run_id="20260422-140000",
        round_idx=1,
        slug="gated-mlp",
        fork_from="main",
    )
    assert Path(plan["worktree_dir"]).is_dir()
    assert plan["branch"] == "exp/20260422-140000/01-gated-mlp"


def test_create_compose_worktree_detects_conflict(git_repo):
    from autoqec.orchestration.worktree import (
        create_compose_worktree,
        create_round_worktree,
    )

    # Build two divergent branches that both edit the same line of the same file.
    plan_a = create_round_worktree(git_repo, "t", 1, "a", fork_from="main")
    (Path(plan_a["worktree_dir"]) / "baseline.txt").write_text("A\n")
    subprocess.run(["git", "add", "."], cwd=plan_a["worktree_dir"], check=True)
    subprocess.run(["git", "commit", "-m", "a"], cwd=plan_a["worktree_dir"], check=True)

    plan_b = create_round_worktree(git_repo, "t", 2, "b", fork_from="main")
    (Path(plan_b["worktree_dir"]) / "baseline.txt").write_text("B\n")
    subprocess.run(["git", "add", "."], cwd=plan_b["worktree_dir"], check=True)
    subprocess.run(["git", "commit", "-m", "b"], cwd=plan_b["worktree_dir"], check=True)

    result = create_compose_worktree(
        repo_root=git_repo,
        run_id="t",
        round_idx=3,
        slug="compose-a-b",
        parents=[plan_a["branch"], plan_b["branch"]],
    )
    assert result["status"] == "compose_conflict"
    assert "baseline.txt" in result["conflicting_files"]
    # Worktree should be cleaned + branch deleted per §15.6.3
    assert not Path(result["worktree_dir"]).exists()
    branches = subprocess.check_output(
        ["git", "branch", "--list"], cwd=git_repo, text=True,
    )
    assert "compose-a-b" not in branches


def test_cleanup_round_worktree_removes_checkout_keeps_branch(git_repo):
    from autoqec.orchestration.worktree import (
        cleanup_round_worktree,
        create_round_worktree,
    )

    plan = create_round_worktree(git_repo, "t", 1, "a", fork_from="main")
    cleanup_round_worktree(repo_root=git_repo, worktree_dir=plan["worktree_dir"])
    assert not Path(plan["worktree_dir"]).exists()
    branches = subprocess.check_output(
        ["git", "branch", "--list"], cwd=git_repo, text=True,
    )
    assert "exp/t/01-a" in branches
