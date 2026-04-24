"""Tests for autoqec.orchestration.subprocess_runner (§15.8).

The smoke test is marked ``integration`` because it spins up a real
subprocess that imports torch + stim. Unit CI skips it; GPU runs wire it
in via ``pytest --run-integration``.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from autoqec.envs.schema import load_env_yaml
from autoqec.runner.schema import RunnerConfig


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _init_git_repo(repo_root: Path) -> None:
    """Minimal git init so subprocess_runner can run ``git -C <worktree> commit``."""
    subprocess.check_call(
        ["git", "-C", str(repo_root), "init", "-q", "-b", "main"]
    )
    subprocess.check_call(
        ["git", "-C", str(repo_root), "config", "user.email", "test@example.com"]
    )
    subprocess.check_call(
        ["git", "-C", str(repo_root), "config", "user.name", "Test"]
    )
    (repo_root / "README.md").write_text("seed\n")
    subprocess.check_call(["git", "-C", str(repo_root), "add", "README.md"])
    subprocess.check_call(
        ["git", "-C", str(repo_root), "commit", "-q", "-m", "seed"]
    )


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
        if "run-round-internal" in argv:
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


def test_subprocess_runner_uses_shell_false_and_venv_python(tmp_path):
    """The child process must spawn via a static internal command.

    subprocess_runner now also issues git invocations for the §15.10
    pointer commit, so we capture only the first ``run-round-internal``
    call instead of "whatever was last run".
    """
    from autoqec.orchestration import subprocess_runner

    captured: dict = {}

    class _FakeChild:
        returncode = 0
        stdout = '{"status": "ok", "commit_sha": "abc123", "round_attempt_id": "u1"}'
        stderr = ""

    class _FakeGit:
        returncode = 0
        stdout = "deadbeef\n"
        stderr = ""

    def _fake_run(argv, **kwargs):
        if "run-round-internal" in argv:
            captured["argv"] = list(argv)
            captured["kwargs"] = kwargs
            return _FakeChild()
        # Short-circuit the pointer-commit git invocations so the test
        # stays hermetic and we don't need a real repo on disk.
        return _FakeGit()

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

    assert captured["argv"] == [sys.executable, "-m", "cli.autoqec", "run-round-internal"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
    assert "executable" not in captured["kwargs"]
    child_env = captured["kwargs"]["env"]
    assert child_env["AUTOQEC_CHILD_ROUND_DIR"] == str((tmp_path / "round_1").resolve())
    assert child_env["AUTOQEC_CHILD_BRANCH"] == "exp/foo/01-bar"
    assert child_env["AUTOQEC_CHILD_ROUND_ATTEMPT_ID"] == "u1"


# ─── M2: round_N_pointer.json writer ─────────────────────────────────────


def test_subprocess_runner_writes_and_commits_pointer(tmp_path):
    """After the child returns, parent writes round_N_pointer.json into the
    worktree, git-adds + commits it on the branch, and sets metrics.commit_sha
    to the new HEAD. This is the producer side of §15.10 auto-heal — without
    it reconcile can never recover an orphaned branch.
    """
    from autoqec.orchestration import subprocess_runner

    _init_git_repo(tmp_path)
    subprocess.check_call(
        ["git", "-C", str(tmp_path), "checkout", "-q", "-b", "exp/test/07-pointer"]
    )

    child_stdout = json.dumps({
        "status": "ok",
        "ler_plain_classical": 1e-3,
        "ler_predecoder": 5e-4,
        "delta_ler": 5e-4,
        "round_attempt_id": "attempt-abc",
    })

    class _FakeCompletedProcess:
        returncode = 0
        stdout = child_stdout
        stderr = ""

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "runs" / "r1" / "round_1"),
        code_cwd=str(tmp_path),
        branch="exp/test/07-pointer",
    )

    real_subprocess_run = subprocess.run

    def _stub_child_pass_through_git(argv, **kw):
        # The child invocation is ``python -m cli.autoqec run-round-internal``;
        # stub that. Everything else (git add / git commit / git rev-parse)
        # runs for real so the pointer commit actually lands on the branch.
        if "run-round-internal" in argv:
            return _FakeCompletedProcess()
        return real_subprocess_run(argv, **kw)

    with patch.object(
        subprocess_runner.subprocess, "run", side_effect=_stub_child_pass_through_git
    ):
        metrics = subprocess_runner.run_round_in_subprocess(
            cfg, env, round_attempt_id="attempt-abc"
        )

    pointer_path = tmp_path / "round_1" / "round_1_pointer.json"
    assert pointer_path.exists(), "pointer file was not written into the worktree"
    payload = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert payload["round_attempt_id"] == "attempt-abc"
    assert payload["branch"] == "exp/test/07-pointer"
    assert payload["round_idx"] == 1

    # File must be committed on the branch — reconcile reads it via
    # ``git show <branch>:round_<NN>/round_<NN>_pointer.json``.
    shown = subprocess.check_output(
        ["git", "-C", str(tmp_path), "show",
         "exp/test/07-pointer:round_1/round_1_pointer.json"],
        text=True,
    )
    assert json.loads(shown)["round_attempt_id"] == "attempt-abc"

    # metrics.commit_sha must be set to the new HEAD containing the pointer.
    head_sha = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    assert metrics.commit_sha == head_sha
    tip_subject = subprocess.check_output(
        ["git", "-C", str(tmp_path), "log", "-1", "--pretty=%s"],
        text=True,
    ).strip()
    assert tip_subject.startswith("chore(")


def test_subprocess_runner_writes_artifact_manifest(tmp_path):
    """Worktree rounds should leave a manifest next to metrics.json."""
    from autoqec.orchestration import subprocess_runner

    _init_git_repo(tmp_path)
    subprocess.check_call(
        ["git", "-C", str(tmp_path), "checkout", "-q", "-b", "exp/test/09-manifest"]
    )
    worktree_run_sha = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    child_stdout = json.dumps({
        "status": "ok",
        "ler_plain_classical": 1e-3,
        "ler_predecoder": 5e-4,
        "delta_ler": 5e-4,
        "round_attempt_id": "attempt-manifest",
    })

    class _FakeCompletedProcess:
        returncode = 0
        stdout = child_stdout
        stderr = ""

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "runs" / "r1" / "round_1"),
        code_cwd=str(tmp_path),
        branch="exp/test/09-manifest",
    )

    real_subprocess_run = subprocess.run

    def _stub_child_pass_through_git(argv, **kw):
        if "run-round-internal" in argv:
            return _FakeCompletedProcess()
        return real_subprocess_run(argv, **kw)

    with patch.object(
        subprocess_runner.subprocess, "run", side_effect=_stub_child_pass_through_git
    ):
        subprocess_runner.run_round_in_subprocess(
            cfg, env, round_attempt_id="attempt-manifest"
        )

    manifest_path = Path(cfg.round_dir) / "artifact_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["repo"]["branch"] == "exp/test/09-manifest"
    assert manifest["repo"]["commit_sha"] == worktree_run_sha
    assert manifest["environment"]["env_yaml_sha256"]
    assert manifest["round"]["dsl_config_sha256"]
    assert Path(manifest["round"]["command_line"][0]).name.startswith("python")
    assert manifest["round"]["command_line"][1:] == [
        "-m",
        "cli.autoqec",
        "run-round-internal",
    ]
    assert manifest["artifacts"] == {
        "config_yaml": "config.yaml",
        "checkpoint": "checkpoint.pt",
        "metrics": "metrics.json",
        "train_log": "train.log",
    }


def test_subprocess_runner_skips_pointer_on_compose_conflict(tmp_path):
    """compose_conflict rows carry branch=None — no commit target, no pointer."""
    from autoqec.orchestration import subprocess_runner

    _init_git_repo(tmp_path)

    child_stdout = json.dumps({
        "status": "compose_conflict",
        "status_reason": "merge conflict in x.py",
        "round_attempt_id": "attempt-xyz",
        "conflicting_files": ["x.py"],
    })

    class _FakeCompletedProcess:
        returncode = 0
        stdout = child_stdout
        stderr = ""

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "runs" / "r1" / "round_2"),
        code_cwd=str(tmp_path),
        branch="exp/test/08-compose",
    )

    with patch.object(subprocess_runner.subprocess, "run", return_value=_FakeCompletedProcess()):
        metrics = subprocess_runner.run_round_in_subprocess(
            cfg, env, round_attempt_id="attempt-xyz"
        )

    assert metrics.status == "compose_conflict"
    assert metrics.commit_sha is None
    # No pointer file should have been written — compose_conflict has no branch.
    assert not (tmp_path / "round_2" / "round_2_pointer.json").exists()
