from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

import cli.autoqec as cli
from autoqec.envs.schema import load_env_yaml
from autoqec.eval.schema import VerifyReport
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


def test_load_round_metrics_for_verify_returns_none_when_missing(tmp_path) -> None:
    assert cli._load_round_metrics_for_verify(tmp_path / "round_9") is None


def test_load_round_metrics_for_verify_infers_round_from_directory_name(tmp_path) -> None:
    round_dir = tmp_path / "round_7"
    round_dir.mkdir()
    (round_dir / "metrics.json").write_text(
        json.dumps({"status": "ok"}),
        encoding="utf-8",
    )

    metrics = cli._load_round_metrics_for_verify(round_dir)

    assert metrics["status"] == "ok"
    assert metrics["round"] == 7


def test_verify_command_writes_report_artifacts(monkeypatch, tmp_path) -> None:
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    captured = {}

    def fake_verify(ckpt, env_spec, holdout_seeds, n_shots):
        captured["ckpt"] = ckpt
        captured["env"] = env_spec
        captured["holdout_seeds"] = holdout_seeds
        captured["n_shots"] = n_shots
        return VerifyReport(
            verdict="VERIFIED",
            ler_holdout=0.1,
            ler_holdout_ci=(0.08, 0.12),
            delta_ler_holdout=0.02,
            ler_shuffled=0.11,
            ablation_sanity_ok=True,
            holdout_seeds_used=holdout_seeds,
            seed_leakage_check_ok=True,
            notes="ok",
        )

    monkeypatch.setattr("autoqec.eval.independent_eval.independent_verify", fake_verify)

    runner = CliRunner()
    result = runner.invoke(
        cli.verify,
        [str(round_dir), "--env", "autoqec/envs/builtin/surface_d5_depol.yaml", "--n-seeds", "3"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "VERIFIED" in result.output
    assert captured["ckpt"] == round_dir / "checkpoint.pt"
    assert captured["holdout_seeds"] == [9000, 9001, 9002]
    assert json.loads((round_dir / "verification_report.json").read_text())["verdict"] == "VERIFIED"
    report_md = (round_dir / "verification_report.md").read_text()
    assert "Verification Report" in report_md
    assert "Paired eval bundle ID" not in report_md


def test_verify_command_admits_verified_round_into_pareto(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    metrics = RoundMetrics(
        status="ok",
        flops_per_syndrome=123,
        n_params=7,
        checkpoint_path=str(round_dir / "checkpoint.pt"),
        train_wallclock_s=1.5,
    ).model_dump()
    metrics["round"] = 1
    (round_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def fake_verify(_ckpt, _env_spec, holdout_seeds, n_shots=None):
        return VerifyReport(
            verdict="VERIFIED",
            ler_holdout=0.1,
            ler_holdout_ci=(0.08, 0.12),
            delta_ler_holdout=0.02,
            ler_shuffled=0.11,
            ablation_sanity_ok=True,
            holdout_seeds_used=holdout_seeds,
            seed_leakage_check_ok=True,
            notes="ok",
            delta_vs_baseline_holdout=0.02,
            paired_eval_bundle_id="33333333-3333-5333-8333-333333333333",
        )

    monkeypatch.setattr("autoqec.eval.independent_eval.independent_verify", fake_verify)

    runner = CliRunner()
    result = runner.invoke(
        cli.verify,
        [str(round_dir), "--env", "autoqec/envs/builtin/surface_d5_depol.yaml", "--n-seeds", "2"],
        catch_exceptions=False,
    )
    rerun = runner.invoke(
        cli.verify,
        [str(round_dir), "--env", "autoqec/envs/builtin/surface_d5_depol.yaml", "--n-seeds", "2"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert rerun.exit_code == 0
    pareto = json.loads((run_dir / "pareto.json").read_text(encoding="utf-8"))
    assert len(pareto) == 1
    assert pareto[0]["round"] == 1
    assert pareto[0]["delta_vs_baseline_holdout"] == 0.02
    assert pareto[0]["paired_eval_bundle_id"] == "33333333-3333-5333-8333-333333333333"
    assert "Paired eval bundle ID: 33333333-3333-5333-8333-333333333333" in (
        round_dir / "verification_report.md"
    ).read_text(encoding="utf-8")


def test_verify_command_skips_pareto_for_non_verified_report(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    metrics = RoundMetrics(
        status="ok",
        flops_per_syndrome=123,
        n_params=7,
        checkpoint_path=str(round_dir / "checkpoint.pt"),
    ).model_dump()
    metrics["round"] = 1
    (round_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def fake_verify(_ckpt, _env_spec, holdout_seeds, n_shots=None):
        return VerifyReport(
            verdict="SUSPICIOUS",
            ler_holdout=0.1,
            ler_holdout_ci=(0.08, 0.12),
            delta_ler_holdout=0.0,
            ler_shuffled=0.1,
            ablation_sanity_ok=True,
            holdout_seeds_used=holdout_seeds,
            seed_leakage_check_ok=True,
            notes="needs more evidence",
            delta_vs_baseline_holdout=0.0,
            paired_eval_bundle_id="44444444-4444-5444-8444-444444444444",
        )

    monkeypatch.setattr("autoqec.eval.independent_eval.independent_verify", fake_verify)

    runner = CliRunner()
    result = runner.invoke(
        cli.verify,
        [str(round_dir), "--env", "autoqec/envs/builtin/surface_d5_depol.yaml", "--n-seeds", "2"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert not (run_dir / "pareto.json").exists()


def test_verify_command_runs_with_offline_backend_env(monkeypatch, tmp_path) -> None:
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "offline")
    monkeypatch.setenv("AUTOQEC_CODER_BACKEND", "offline")
    monkeypatch.setenv("AUTOQEC_ANALYST_BACKEND", "offline")

    def fake_verify(_ckpt, _env_spec, holdout_seeds, n_shots=None):
        return VerifyReport(
            verdict="SUSPICIOUS",
            ler_holdout=0.1,
            ler_holdout_ci=(0.08, 0.12),
            delta_ler_holdout=0.0,
            ler_shuffled=0.1,
            ablation_sanity_ok=True,
            holdout_seeds_used=holdout_seeds,
            seed_leakage_check_ok=True,
            notes="offline replay ok",
            paired_eval_bundle_id="55555555-5555-5555-8555-555555555555",
        )

    monkeypatch.setattr("autoqec.eval.independent_eval.independent_verify", fake_verify)

    runner = CliRunner()
    result = runner.invoke(
        cli.verify,
        [str(round_dir), "--env", "autoqec/envs/builtin/surface_d5_depol.yaml", "--n-seeds", "2"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "SUSPICIOUS" in result.output
    assert (round_dir / "verification_report.json").exists()


def test_package_run_command_emits_result_prefix(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    (round_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo": {"commit_sha": "abc123", "branch": "topic", "dirty": False},
                "environment": {"env_yaml_path": "autoqec/envs/builtin/surface_d5_depol.yaml", "env_yaml_sha256": "deadbeef"},
                "round": {
                    "run_id": "demo-run",
                    "round_dir": "round_1",
                    "round": 1,
                    "dsl_config_sha256": "cafebabe",
                    "command_line": ["python", "-m", "cli.autoqec", "run", "--no-llm"],
                },
                "artifacts": {
                    "config_yaml": "config.yaml",
                    "checkpoint": "checkpoint.pt",
                    "metrics": "metrics.json",
                    "train_log": "train.log",
                },
                "packages": {
                    "python": "3.12.3",
                    "torch": "2.0",
                    "cuda": "none",
                    "stim": "1.0",
                    "pymatching": "2.0",
                    "ldpc": "1.0",
                },
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli.main, ["package-run", str(run_dir)], catch_exceptions=False)

    assert result.exit_code == 0
    payload_line = next(line for line in result.output.splitlines() if line.startswith(cli.RESULT_PREFIX))
    payload = json.loads(payload_line[len(cli.RESULT_PREFIX) :])
    assert payload["run_dir"] == str(run_dir.resolve())
    assert payload["package_path"].endswith("demo-run.tar.gz")
    assert payload["rounds"] == ["round_1"]
    assert Path(payload["package_path"]).exists()


def test_package_run_command_refuses_to_overwrite_existing_tarball(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    (round_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo": {"commit_sha": "abc123", "branch": "topic", "dirty": False},
                "environment": {"env_yaml_path": "autoqec/envs/builtin/surface_d5_depol.yaml", "env_yaml_sha256": "deadbeef"},
                "round": {
                    "run_id": "demo-run",
                    "round_dir": "round_1",
                    "round": 1,
                    "dsl_config_sha256": "cafebabe",
                    "command_line": ["python", "-m", "cli.autoqec", "run", "--no-llm"],
                },
                "artifacts": {
                    "config_yaml": "config.yaml",
                    "checkpoint": "checkpoint.pt",
                    "metrics": "metrics.json",
                    "train_log": "train.log",
                },
                "packages": {
                    "python": "3.12.3",
                    "torch": "2.0",
                    "cuda": "none",
                    "stim": "1.0",
                    "pymatching": "2.0",
                    "ldpc": "1.0",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir.parent / "demo-run.tar.gz").write_text("already here", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.main, ["package-run", str(run_dir)])

    assert result.exit_code != 0
    assert "already exists" in result.output


def test_review_log_handles_missing_history(tmp_path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.review_log, [str(tmp_path)], catch_exceptions=False)
    assert result.exit_code == 0
    assert "No history.jsonl" in result.output


def test_review_log_summarizes_existing_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "history.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"status": "ok", "train_wallclock_s": 1.0, "hypothesis": "alpha"}),
                json.dumps({"status": "killed_by_safety", "train_wallclock_s": 3.0, "hypothesis": "beta"}),
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "pareto.json").write_text(json.dumps([{"delta_ler": 0.1}]), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.review_log, [str(run_dir)], catch_exceptions=False)
    payload = json.loads(result.output)
    assert payload["n_rounds"] == 2
    assert payload["n_pareto"] == 1
    assert payload["n_killed_by_safety"] == 1
    assert payload["mean_wallclock_s"] == 2.0
    assert payload["top_hypotheses"] == ["alpha", "beta"]


def test_review_log_falls_back_to_candidate_pareto(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "history.jsonl").write_text(json.dumps({"status": "ok"}) + "\n", encoding="utf-8")
    (run_dir / "candidate_pareto.json").write_text(
        json.dumps([{"delta_ler": 0.1}, {"delta_ler": 0.2}]),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli.review_log, [str(run_dir)], catch_exceptions=False)

    payload = json.loads(result.output)
    assert payload["n_pareto"] == 2


def _materialize_diagnose_fixture(tmp_path: Path, fixture_name: str) -> Path:
    src = Path(__file__).parent / "fixtures" / "diagnose" / fixture_name
    dst = tmp_path / fixture_name
    dst.mkdir()
    for src_name, dst_name in (
        ("config.yaml", "config.yaml"),
        ("metrics.json", "metrics.json"),
        ("train_log.txt", "train.log"),
    ):
        (dst / dst_name).write_text((src / src_name).read_text(encoding="utf-8"), encoding="utf-8")
    return dst


def test_diagnose_handles_round_dir_and_missing_rounds(tmp_path) -> None:
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    (round_dir / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")

    runner = CliRunner()
    direct = runner.invoke(cli.diagnose, [str(round_dir)], catch_exceptions=False)
    payload = json.loads(direct.output)
    assert payload["has_metrics.json"] is True
    assert payload["metrics"]["status"] == "ok"

    empty = runner.invoke(cli.diagnose, [str(tmp_path / "empty")], catch_exceptions=False)
    assert "No round dirs found" in empty.output


def test_diagnose_uses_latest_round_from_run_dir(tmp_path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "round_1").mkdir(parents=True)
    (run_dir / "round_2").mkdir(parents=True)
    (run_dir / "round_2" / "metrics.json").write_text(json.dumps({"status": "compile_error"}), encoding="utf-8")
    (run_dir / "round_2" / "train.log").write_text("", encoding="utf-8")
    (run_dir / "round_2" / "config.yaml").write_text("type: gnn\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli.diagnose, [str(run_dir)], catch_exceptions=False)
    payload = json.loads(result.output)
    assert payload["path"].endswith("round_2")


@pytest.mark.parametrize(
    ("fixture_name", "expected_root_cause", "expected_signal"),
    [
        ("oom", "oom", "cuda out of memory"),
        ("nan", "nan_loss", "nan"),
        ("degenerate_p_zero", "degenerate_p_zero", "p = 0"),
    ],
)
def test_diagnose_identifies_known_failure_signatures(
    tmp_path: Path,
    fixture_name: str,
    expected_root_cause: str,
    expected_signal: str,
) -> None:
    round_dir = _materialize_diagnose_fixture(tmp_path, fixture_name)

    runner = CliRunner()
    result = runner.invoke(cli.diagnose, [str(round_dir)], catch_exceptions=False)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["root_cause"] == expected_root_cause
    assert expected_signal in "\n".join(payload["signals"]).lower()


def test_diagnose_never_claims_to_apply_a_fix(tmp_path: Path) -> None:
    round_dir = _materialize_diagnose_fixture(tmp_path, "oom")

    runner = CliRunner()
    result = runner.invoke(cli.diagnose, [str(round_dir)], catch_exceptions=False)

    payload = json.loads(result.output)
    assert payload["read_only"] is True
    assert "applied_fix" not in payload
