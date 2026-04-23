from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

import cli.autoqec as cli
from autoqec.envs.schema import load_env_yaml
from autoqec.runner.schema import RoundMetrics


def test_load_example_templates_error_paths(monkeypatch) -> None:
    class BrokenRoot:
        def joinpath(self, _name: str):
            return self

        def iterdir(self):
            raise FileNotFoundError("missing")

    monkeypatch.setattr(cli.resources, "files", lambda _pkg: BrokenRoot())
    with pytest.raises(click.ClickException, match="Example templates are not available"):
        cli.load_example_templates()

    class EmptyRoot:
        def joinpath(self, _name: str):
            return self

        def iterdir(self):
            return []

    monkeypatch.setattr(cli.resources, "files", lambda _pkg: EmptyRoot())
    with pytest.raises(click.ClickException, match="No example templates were found"):
        cli.load_example_templates()


def test_parse_fork_from_option_variants() -> None:
    assert cli._parse_fork_from_option(None) is None
    assert cli._parse_fork_from_option("exp/t/01-a") == "exp/t/01-a"
    assert cli._parse_fork_from_option('["exp/a","exp/b"]') == ["exp/a", "exp/b"]
    with pytest.raises(click.BadParameter):
        cli._parse_fork_from_option("[malformed")
    with pytest.raises(click.BadParameter):
        cli._parse_fork_from_option("[1,2]")


def test_run_round_internal_cmd_reads_env_bridge(monkeypatch, tmp_path) -> None:
    captured = {}

    monkeypatch.setenv(cli.AUTOQEC_CHILD_ENV_YAML, "env.yaml")
    monkeypatch.setenv(cli.AUTOQEC_CHILD_CONFIG_YAML, "config.yaml")
    monkeypatch.setenv(cli.AUTOQEC_CHILD_ROUND_DIR, str(tmp_path / "round_1"))
    monkeypatch.setenv(cli.AUTOQEC_CHILD_PROFILE, "dev")
    monkeypatch.setenv(cli.AUTOQEC_CHILD_CODE_CWD, str(tmp_path))
    monkeypatch.setenv(cli.AUTOQEC_CHILD_BRANCH, "exp/t/01-a")
    monkeypatch.setenv(cli.AUTOQEC_CHILD_FORK_FROM, '["exp/a","exp/b"]')
    monkeypatch.setenv(cli.AUTOQEC_CHILD_COMPOSE_MODE, "pure")
    monkeypatch.setenv(cli.AUTOQEC_CHILD_ROUND_ATTEMPT_ID, "uuid-1")

    monkeypatch.setattr(cli, "_run_round_impl", lambda **kwargs: captured.update(kwargs))
    cli.run_round_internal_cmd.callback()

    assert captured["env_yaml"] == "env.yaml"
    assert captured["_internal_execute_locally"] is True
    assert captured["round_attempt_id"] == "uuid-1"


def test_run_command_rejects_llm_mode(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_env_yaml", lambda _path: env)
    monkeypatch.setattr(cli.time, "strftime", lambda _fmt: "20260423-000000")
    monkeypatch.setattr(cli, "RunMemory", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(cli, "load_example_templates", lambda: [("gnn_small", {"type": "gnn"})])

    runner = CliRunner()
    result = runner.invoke(cli.run, [env.model_dump()["name"]], catch_exceptions=True)
    assert result.exit_code != 0
    assert "LLM mode is not wired" in result.output


def test_add_env_writes_yaml_for_mwpm_and_osd(tmp_path) -> None:
    runner = CliRunner()
    mwpm_out = tmp_path / "mwpm.yaml"
    osd_out = tmp_path / "osd.yaml"

    mwpm = runner.invoke(
        cli.add_env,
        [
            "--out",
            str(mwpm_out),
            "--name",
            "surface_like",
            "--code-source",
            "circuits/code.stim",
            "--noise-p",
            "1e-3,2e-3",
            "--backend",
            "mwpm",
        ],
        catch_exceptions=False,
    )
    osd = runner.invoke(
        cli.add_env,
        [
            "--out",
            str(osd_out),
            "--name",
            "qldpc_like",
            "--code-source",
            "circuits/hx.npy",
            "--noise-p",
            "1e-3,2e-3",
            "--backend",
            "osd",
        ],
        catch_exceptions=False,
    )

    assert mwpm.exit_code == 0
    assert osd.exit_code == 0
    assert "Wrote" in mwpm.output
    assert "Wrote" in osd.output

    mwpm_payload = cli.yaml.safe_load(mwpm_out.read_text())
    osd_payload = cli.yaml.safe_load(osd_out.read_text())
    assert mwpm_payload["code"]["type"] == "stim_circuit"
    assert mwpm_payload["baseline_decoders"] == ["pymatching"]
    assert osd_payload["code"]["type"] == "parity_check_matrix"
    assert osd_payload["baseline_decoders"] == ["bposd"]


def test_run_command_emits_result_prefix(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_env_yaml", lambda _path: env)
    monkeypatch.setattr(cli.time, "strftime", lambda _fmt: "20260423-000001")
    monkeypatch.setattr(cli.random, "choice", lambda items: items[0])
    monkeypatch.setattr(cli, "load_example_templates", lambda: [("gnn_small", {"type": "gnn", "training": {"learning_rate": 1e-3, "batch_size": 1, "epochs": 1}})])

    class MemoryStub:
        def __init__(self, run_dir, pareto_filename):
            self.run_dir = Path(run_dir)
            self.pareto_path = self.run_dir / pareto_filename
            self.pareto_path.write_text("[]", encoding="utf-8")

        def append_round(self, record):
            pass

        def update_pareto(self, pareto):
            self.pareto_path.write_text(json.dumps(pareto), encoding="utf-8")

    monkeypatch.setattr(cli, "RunMemory", MemoryStub)
    monkeypatch.setattr(
        cli,
        "run",
        cli.run,
    )

    def fake_run_round(_cfg, _env):
        return RoundMetrics(status="ok", delta_ler=0.0, flops_per_syndrome=1, n_params=1)

    monkeypatch.setattr("autoqec.runner.runner.run_round", fake_run_round)

    runner_cli = CliRunner()
    result = runner_cli.invoke(
        cli.run,
        [env.model_dump()["name"], "--rounds", "1", "--profile", "dev", "--no-llm"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert cli.RESULT_PREFIX in result.output


def test_candidate_pareto_skips_ok_rows_with_missing_metrics() -> None:
    front = cli._candidate_pareto(
        [
            {
                "round": 1,
                "status": "ok",
                "delta_ler": 0.1,
                "flops_per_syndrome": None,
                "n_params": 10,
            },
            {
                "round": 2,
                "status": "ok",
                "delta_ler": 0.2,
                "flops_per_syndrome": 5,
                "n_params": 10,
            },
        ]
    )
    assert [row["round"] for row in front] == [2]
