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

from tests.fixture_utils import load_json_fixture

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

RUN_ROUND_HELP_CONTRACT = load_json_fixture("public_api", "run_round_help_contract.json")


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
        [sys.executable, "-m", "cli.autoqec", *RUN_ROUND_HELP_CONTRACT["argv"]],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    for flag in RUN_ROUND_HELP_CONTRACT["expected_flags"]:
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
    captured = {}

    # Pre-seed a lightweight stub for autoqec.runner.runner so the in-process
    # import in run_round_cmd succeeds even without torch installed.
    fake_runner_mod = types.ModuleType("autoqec.runner.runner")

    def _fake_run_round(cfg, _env):
        captured["cfg"] = cfg
        return fake_metrics

    fake_runner_mod.run_round = _fake_run_round

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
        ), patch("cli.autoqec.subprocess.check_output", return_value="abc123\n"):
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
    payload = json.loads(result.output)
    assert captured["cfg"].code_cwd is None
    assert payload["branch"] == "exp/foo/01-bar"
    assert payload["commit_sha"] == "abc123"
    assert payload["round_attempt_id"] == "test-uuid"
    saved = json.loads((Path(round_dir) / "metrics.json").read_text(encoding="utf-8"))
    assert saved["branch"] == "exp/foo/01-bar"
    assert saved["commit_sha"] == "abc123"
    assert saved["round_attempt_id"] == "test-uuid"


def test_internal_flag_writes_failure_metrics_when_git_head_lookup_fails(tmp_path):
    """A git HEAD lookup failure must still leave behind train_error metrics."""
    import sys
    import types

    from cli.autoqec import run_round_cmd
    from autoqec.runner.schema import RoundMetrics

    env_yaml = str(Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute())
    cfg_yaml_path = tmp_path / "cfg.yaml"
    cfg_yaml_path.write_text("type: gnn\noutput_mode: soft_priors\n")
    round_dir = str(tmp_path / "round_1")

    fake_metrics = RoundMetrics(
        status="ok",
        ler_plain_classical=1e-3,
        ler_predecoder=5e-4,
        delta_ler=5e-4,
    )
    fake_runner_mod = types.ModuleType("autoqec.runner.runner")
    fake_runner_mod.run_round = lambda *_a, **_kw: fake_metrics
    original = sys.modules.get("autoqec.runner.runner")
    sys.modules["autoqec.runner.runner"] = fake_runner_mod

    try:
        with patch(
            "cli.autoqec.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(
                128,
                ["git", "rev-parse", "HEAD"],
            ),
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
                catch_exceptions=True,
            )
    finally:
        if original is not None:
            sys.modules["autoqec.runner.runner"] = original
        else:
            sys.modules.pop("autoqec.runner.runner", None)

    assert result.exit_code != 0
    assert isinstance(result.exception, subprocess.CalledProcessError)
    saved = json.loads((Path(round_dir) / "metrics.json").read_text(encoding="utf-8"))
    assert saved["status"] == "train_error"
    assert saved["round_attempt_id"] == "test-uuid"
    assert saved["branch"] is None
    assert "git rev-parse HEAD failed" in saved["status_reason"]


def test_internal_flag_strips_code_cwd_before_in_process_runner(tmp_path):
    """Regression: the child hop must call run_round with cfg.code_cwd=None.

    In-process ``run_round`` raises ``RunnerCallPathError`` whenever
    ``cfg.code_cwd is not None`` (§15.8). The recursion guard already skips
    the subprocess dispatcher, but if the child CLI forwards ``code_cwd``
    into ``RunnerConfig`` verbatim the local runner call still blows up.
    Worktree metadata (branch/fork_from/compose_mode/round_attempt_id) must
    still flow through the round record.
    """
    import sys
    import types

    from cli.autoqec import run_round_cmd
    from autoqec.runner.schema import RoundMetrics

    env_yaml = str(Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute())
    cfg_yaml_path = tmp_path / "cfg.yaml"
    cfg_yaml_path.write_text("type: gnn\noutput_mode: soft_priors\n")
    round_dir = str(tmp_path / "round_1")

    captured: dict = {}

    def _fake_run_round(cfg, _env, **_kw):
        captured["cfg"] = cfg
        # Emulate the real in-process guard: if code_cwd is set we must crash.
        if cfg.code_cwd is not None:
            from autoqec.runner.runner import RunnerCallPathError

            raise RunnerCallPathError("in-process run_round rejects code_cwd")
        return RoundMetrics(
            status="ok",
            ler_plain_classical=1e-3,
            ler_predecoder=5e-4,
            delta_ler=5e-4,
            branch=cfg.branch,
            commit_sha="deadbeef" if cfg.branch else None,
            fork_from=cfg.fork_from,
            round_attempt_id="test-uuid",
        )

    fake_runner_mod = types.ModuleType("autoqec.runner.runner")

    class _FakeRunnerCallPathError(RuntimeError):
        pass

    fake_runner_mod.run_round = _fake_run_round
    fake_runner_mod.RunnerCallPathError = _FakeRunnerCallPathError

    original = sys.modules.get("autoqec.runner.runner")
    sys.modules["autoqec.runner.runner"] = fake_runner_mod
    try:
        with patch("cli.autoqec.subprocess.check_output", return_value="deadbeef\n"):
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
    cfg = captured.get("cfg")
    assert cfg is not None, "run_round was never called"
    assert cfg.code_cwd is None, (
        "child hop must null out code_cwd before the in-process Runner; "
        f"got code_cwd={cfg.code_cwd!r}"
    )
    assert cfg.branch is None
    assert cfg.fork_from is None
    assert cfg.compose_mode is None
    payload = json.loads(result.output)
    assert payload["branch"] == "exp/foo/01-bar"
    assert payload["commit_sha"] == "deadbeef"


def test_fork_from_malformed_json_is_bad_parameter(tmp_path):
    """Broken JSON in --fork-from must surface as click.BadParameter, not JSONDecodeError."""
    import click
    import sys
    import types

    from cli.autoqec import run_round_cmd
    from autoqec.runner.schema import RoundMetrics

    env_yaml = str(Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute())
    cfg_yaml_path = tmp_path / "cfg.yaml"
    cfg_yaml_path.write_text("type: gnn\noutput_mode: soft_priors\n")
    round_dir = str(tmp_path / "round_1")

    fake_metrics = RoundMetrics(status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4, delta_ler=5e-4)
    fake_runner_mod = types.ModuleType("autoqec.runner.runner")
    fake_runner_mod.run_round = lambda *_a, **_kw: fake_metrics
    original = sys.modules.get("autoqec.runner.runner")
    sys.modules["autoqec.runner.runner"] = fake_runner_mod

    try:
        cli_runner = CliRunner()
        result = cli_runner.invoke(
            run_round_cmd,
            [
                env_yaml, str(cfg_yaml_path), round_dir,
                "--fork-from", "[malformed",
            ],
            catch_exceptions=True,
        )
    finally:
        if original is not None:
            sys.modules["autoqec.runner.runner"] = original
        else:
            sys.modules.pop("autoqec.runner.runner", None)

    assert result.exit_code != 0
    # BadParameter surfaces to the user with a friendly message, not a raw trace.
    assert isinstance(result.exception, (click.BadParameter, SystemExit)), type(result.exception)
    combined = (result.output or "") + (str(result.exception) if result.exception else "")
    assert "fork-from" in combined.lower() or "json" in combined.lower()


def test_fork_from_json_list_non_strings_rejected(tmp_path):
    """--fork-from JSON must be a list of strings."""
    import click
    import sys
    import types

    from cli.autoqec import run_round_cmd
    from autoqec.runner.schema import RoundMetrics

    env_yaml = str(Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute())
    cfg_yaml_path = tmp_path / "cfg.yaml"
    cfg_yaml_path.write_text("type: gnn\noutput_mode: soft_priors\n")
    round_dir = str(tmp_path / "round_1")

    fake_metrics = RoundMetrics(status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4, delta_ler=5e-4)
    fake_runner_mod = types.ModuleType("autoqec.runner.runner")
    fake_runner_mod.run_round = lambda *_a, **_kw: fake_metrics
    original = sys.modules.get("autoqec.runner.runner")
    sys.modules["autoqec.runner.runner"] = fake_runner_mod

    try:
        cli_runner = CliRunner()
        result = cli_runner.invoke(
            run_round_cmd,
            [env_yaml, str(cfg_yaml_path), round_dir, "--fork-from", "[1,2]"],
            catch_exceptions=True,
        )
    finally:
        if original is not None:
            sys.modules["autoqec.runner.runner"] = original
        else:
            sys.modules.pop("autoqec.runner.runner", None)

    assert result.exit_code != 0
    assert isinstance(result.exception, (click.BadParameter, SystemExit))


def test_fork_from_valid_json_list_parses(tmp_path):
    """A well-formed JSON list of strings parses without error."""
    import sys
    import types

    from cli.autoqec import run_round_cmd
    from autoqec.runner.schema import RoundMetrics

    env_yaml = str(Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute())
    cfg_yaml_path = tmp_path / "cfg.yaml"
    cfg_yaml_path.write_text("type: gnn\noutput_mode: soft_priors\n")
    round_dir = str(tmp_path / "round_1")

    fake_metrics = RoundMetrics(
        status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4, delta_ler=5e-4,
    )
    fake_runner_mod = types.ModuleType("autoqec.runner.runner")
    fake_runner_mod.run_round = lambda *_a, **_kw: fake_metrics
    original = sys.modules.get("autoqec.runner.runner")
    sys.modules["autoqec.runner.runner"] = fake_runner_mod

    try:
        cli_runner = CliRunner()
        result = cli_runner.invoke(
            run_round_cmd,
            [
                env_yaml, str(cfg_yaml_path), round_dir,
                "--fork-from", '["exp/a", "exp/b"]',
                "--compose-mode", "pure",
            ],
            catch_exceptions=False,
        )
    finally:
        if original is not None:
            sys.modules["autoqec.runner.runner"] = original
        else:
            sys.modules.pop("autoqec.runner.runner", None)

    assert result.exit_code == 0, result.output


def test_subprocess_runner_uses_internal_command_env_bridge(tmp_path):
    """The subprocess bridge must avoid dynamic CLI argv payloads.

    subprocess_runner now also issues git invocations to commit the
    §15.10 pointer, so capture only the first child CLI call (the one
    carrying ``run-round-internal``) rather than "whatever was last run".
    """
    from autoqec.orchestration import subprocess_runner
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.schema import RunnerConfig

    captured: dict = {}

    class _FakeCompletedProcess:
        returncode = 0
        stdout = '{"status": "ok", "commit_sha": "abc123", "round_attempt_id": "u1"}'
        stderr = ""

    def _fake_run(argv, **kwargs):
        if "run-round-internal" in argv and "argv" not in captured:
            captured["argv"] = list(argv)
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

    assert captured["argv"] == ["python", "-m", "cli.autoqec", "run-round-internal"]
    assert captured["kwargs"]["env"]["AUTOQEC_CHILD_BRANCH"] == "exp/foo/01-bar"
