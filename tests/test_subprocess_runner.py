"""Tests for autoqec.orchestration.subprocess_runner (§15.8).

The smoke test is marked ``integration`` because it spins up a real
subprocess that imports torch + stim. Unit CI skips it; GPU runs wire it
in via ``pytest --run-integration``.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

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
    from autoqec.orchestration.worktree import cleanup_round_worktree, create_round_worktree
    import subprocess

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    repo_root = Path(".").absolute()
    plan = None
    try:
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
        assert metrics.branch == plan["branch"]
        assert metrics.commit_sha
    finally:
        if plan is not None:
            cleanup_round_worktree(repo_root, plan["worktree_dir"])
            subprocess.run(
                ["git", "-C", str(repo_root), "branch", "-D", plan["branch"]],
                capture_output=True,
                text=True,
            )


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


def test_subprocess_runner_cleans_tempfiles_and_sets_writable_mplconfig(
    monkeypatch,
    tmp_path: Path,
):
    """The shell-out path should not leak temp YAMLs and must avoid HOME-bound MPL cache writes."""
    from autoqec.orchestration import subprocess_runner

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml").model_copy(
        update={"name": "surface_d5_ephemeral"}
    )
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
        code_cwd=str(tmp_path),
        branch="exp/foo/01-bar",
    )

    created: list[Path] = []
    captured: dict = {}
    real_named_tempfile = tempfile.NamedTemporaryFile

    def _tracking_named_tempfile(*args, **kwargs):
        kwargs.setdefault("dir", tmp_path)
        handle = real_named_tempfile(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    class _FakeCompletedProcess:
        returncode = 0
        stdout = '{"status": "ok", "commit_sha": "abc123", "round_attempt_id": "u1"}'
        stderr = ""

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return _FakeCompletedProcess()

    monkeypatch.setattr(
        subprocess_runner.tempfile,
        "NamedTemporaryFile",
        _tracking_named_tempfile,
    )
    monkeypatch.setattr(subprocess_runner.subprocess, "run", _fake_run)

    subprocess_runner.run_round_in_subprocess(cfg, env, round_attempt_id="u1")

    assert len(created) == 2
    assert all(not path.exists() for path in created)
    assert "MPLCONFIGDIR" in captured["env"]
