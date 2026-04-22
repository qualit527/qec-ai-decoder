"""Task 5: run-round CLI accepts worktree flags (§15.7)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="legacy runner path needs torch")
def test_run_round_legacy_positional_still_works(tmp_path):
    """Lin's positional form must keep working."""
    env = Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute()
    cfg = Path("autoqec/example_db/gnn_small.yaml").absolute()
    out = tmp_path / "round_1"
    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "run-round",
         str(env), str(cfg), str(out), "--profile", "dev"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    metrics = json.loads(result.stdout)
    assert metrics["status"] in ("ok", "killed_by_safety")


def test_run_round_accepts_worktree_flags_without_running():
    """The --help page lists the new flags so callers can discover them."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "run-round", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    for flag in ("--code-cwd", "--branch", "--fork-from", "--compose-mode", "--round-attempt-id"):
        assert flag in result.stdout, f"missing flag {flag}"


def test_run_round_internal_flag_skips_subprocess_dispatch(tmp_path):
    """With --_internal-execute-locally, the child must run in-process.

    Without the recursion guard, passing --code-cwd on the CLI re-dispatches
    through subprocess_runner forever. The internal flag forces the local
    branch (in-process Runner) even when --code-cwd is set.
    """
    import sys
    import types

    from cli.autoqec import run_round_cmd
    from autoqec.runner.schema import RoundMetrics

    env_yaml = str(Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute())
    cfg_yaml_path = tmp_path / "cfg.yaml"
    cfg_yaml_path.write_text("type: gnn\noutput_mode: soft_priors\n")
    round_dir = str(tmp_path / "round_1")

    fake_metrics = RoundMetrics(status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4, delta_ler=5e-4)

    # Pre-seed a lightweight stub for autoqec.runner.runner so the in-process
    # import in run_round_cmd succeeds even without torch installed.
    fake_runner_mod = types.ModuleType("autoqec.runner.runner")
    fake_runner_mod.run_round = lambda *_a, **_kw: fake_metrics

    def _boom(*_a, **_kw):
        raise AssertionError(
            "subprocess dispatch must not be invoked when --_internal-execute-locally is set"
        )

    original = sys.modules.get("autoqec.runner.runner")
    sys.modules["autoqec.runner.runner"] = fake_runner_mod
    try:
        with patch(
            "autoqec.orchestration.subprocess_runner.run_round_in_subprocess",
            side_effect=_boom,
        ):
            cli_runner = CliRunner()
            result = cli_runner.invoke(
                run_round_cmd,
                [
                    env_yaml,
                    str(cfg_yaml_path),
                    round_dir,
                    "--code-cwd",
                    str(tmp_path),
                    "--branch",
                    "exp/foo/01-bar",
                    "--round-attempt-id",
                    "test-uuid",
                    "--_internal-execute-locally",
                ],
                catch_exceptions=False,
            )
    finally:
        if original is not None:
            sys.modules["autoqec.runner.runner"] = original
        else:
            sys.modules.pop("autoqec.runner.runner", None)

    assert result.exit_code == 0, result.output


def test_subprocess_runner_injects_internal_flag(tmp_path):
    """The argv emitted by run_round_in_subprocess must carry the guard flag."""
    from autoqec.orchestration import subprocess_runner
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.schema import RunnerConfig

    captured = {}

    class _FakeCompletedProcess:
        returncode = 0
        stdout = '{"status": "ok", "commit_sha": "abc123", "round_attempt_id": "u1"}'
        stderr = ""

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
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

    assert "--_internal-execute-locally" in captured["argv"], captured["argv"]
