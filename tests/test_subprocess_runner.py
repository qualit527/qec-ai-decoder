"""Tests for autoqec.orchestration.subprocess_runner (§15.8).

The smoke test is marked ``integration`` because it spins up a real
subprocess that imports torch + stim. Unit CI skips it; GPU runs wire it
in via ``pytest --run-integration``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from autoqec.envs.schema import load_env_yaml
from autoqec.runner.schema import RunnerConfig


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.integration
@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="subprocess runner invokes torch via run-round")
def test_subprocess_runner_smoke(tmp_path):
    """Run one dev-profile round inside a worktree via subprocess."""
    from autoqec.orchestration.subprocess_runner import run_round_in_subprocess
    from autoqec.orchestration.worktree import create_round_worktree

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    repo_root = Path(".").absolute()
    plan = create_round_worktree(repo_root, "smoke", 1, "gnn-small", fork_from="HEAD")

    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config=yaml.safe_load(
            Path("autoqec/example_db/gnn_small.yaml").read_text()
        ),
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
        code_cwd=plan["worktree_dir"],
        branch=plan["branch"],
    )

    metrics = run_round_in_subprocess(cfg, env, round_attempt_id="test-uuid-1")
    assert metrics.status in ("ok", "killed_by_safety")
    assert metrics.round_attempt_id == "test-uuid-1"


def test_subprocess_runner_rejects_missing_code_cwd():
    """Without code_cwd the Runner must not be shelled out."""
    from autoqec.orchestration.subprocess_runner import run_round_in_subprocess

    cfg = RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={},
        training_profile="dev",
        seed=0,
        round_dir="/tmp/x",
    )
    # Synthesize a minimal EnvSpec-like object via load_env_yaml on builtin.
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")

    with pytest.raises(ValueError, match="code_cwd"):
        run_round_in_subprocess(cfg, env)


def test_subprocess_runner_rejects_suspicious_branch_value(tmp_path):
    """Shell-style metacharacters in branch names must be rejected before spawn."""
    from autoqec.orchestration.subprocess_runner import run_round_in_subprocess

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
        code_cwd=str(tmp_path),
        branch="exp/t/01-a;rm -rf /",
    )

    with pytest.raises(ValueError, match="branch"):
        run_round_in_subprocess(cfg, env, round_attempt_id="u1")


def test_subprocess_runner_uses_shell_false_and_resolved_cwd(tmp_path):
    """The child process must always spawn with shell=False and absolute cwd."""
    from autoqec.orchestration import subprocess_runner

    captured = {}

    class _FakeCompletedProcess:
        returncode = 0
        stdout = '{"status": "ok", "commit_sha": "abc123", "round_attempt_id": "u1"}'
        stderr = ""

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeCompletedProcess()

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
        code_cwd=str(tmp_path),
        branch="exp/foo/01-bar",
    )

    with patch.object(subprocess_runner.subprocess, "run", side_effect=_fake_run):
        subprocess_runner.run_round_in_subprocess(cfg, env, round_attempt_id="u1")

    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
