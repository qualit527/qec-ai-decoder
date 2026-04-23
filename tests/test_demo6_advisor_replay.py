from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _write_manifest(round_dir: Path, *, checkpoint: str = "checkpoint.pt") -> None:
    (round_dir / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo": {"commit_sha": "abc123", "branch": "topic", "dirty": False},
                "environment": {
                    "env_yaml_path": "autoqec/envs/builtin/surface_d5_depol.yaml",
                    "env_yaml_sha256": "deadbeef",
                },
                "round": {
                    "run_id": "demo-run",
                    "round_dir": round_dir.name,
                    "round": 1,
                    "dsl_config_sha256": "cafebabe",
                    "command_line": ["python", "-m", "cli.autoqec", "run", "--no-llm"],
                },
                "artifacts": {
                    "config_yaml": "config.yaml",
                    "checkpoint": checkpoint,
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


def test_package_and_extract_roundtrip(tmp_path: Path) -> None:
    from autoqec.tools.advisor_replay import extract_run_package, package_run_dir

    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    _write_manifest(round_dir)

    package_path = package_run_dir(run_dir)
    extracted_run_dir = extract_run_package(package_path, tmp_path / "replay")

    assert package_path.exists()
    assert extracted_run_dir == tmp_path / "replay" / "demo-run"
    assert (extracted_run_dir / "round_1" / "metrics.json").exists()
    assert (extracted_run_dir / "round_1" / "artifact_manifest.json").exists()


def test_package_run_dir_requires_artifact_manifest(tmp_path: Path) -> None:
    from autoqec.tools.advisor_replay import package_run_dir

    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing artifact manifest"):
        package_run_dir(run_dir)


def test_package_run_dir_rejects_missing_required_artifact(tmp_path: Path) -> None:
    from autoqec.tools.advisor_replay import package_run_dir

    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    _write_manifest(round_dir)

    with pytest.raises(ValueError, match="missing file"):
        package_run_dir(run_dir)


def test_package_run_dir_refuses_to_overwrite_existing_tarball(tmp_path: Path) -> None:
    from autoqec.tools.advisor_replay import package_run_dir

    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    _write_manifest(round_dir)
    package_path = run_dir.parent / "demo-run.tar.gz"
    package_path.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        package_run_dir(run_dir)


def test_compare_verification_reports_accepts_small_float_drift() -> None:
    from autoqec.tools.advisor_replay import compare_verification_reports

    original = {
        "verdict": "VERIFIED",
        "holdout_seeds_used": [9000, 9001],
        "paired_eval_bundle_id": "bundle-1",
        "ler_holdout": 0.1000000,
        "delta_ler_holdout": 0.0200000,
        "ler_shuffled": 0.1100000,
        "ler_holdout_ci": [0.08, 0.12],
    }
    replay = {
        "verdict": "VERIFIED",
        "holdout_seeds_used": [9000, 9001],
        "paired_eval_bundle_id": "bundle-1",
        "ler_holdout": 0.1000004,
        "delta_ler_holdout": 0.0199997,
        "ler_shuffled": 0.1099999,
        "ler_holdout_ci": [0.0800001, 0.1199999],
    }

    compare_verification_reports(original, replay, float_tol=1e-3)


def test_compare_verification_reports_tolerates_missing_optional_fields() -> None:
    from autoqec.tools.advisor_replay import compare_verification_reports

    original = {
        "verdict": "VERIFIED",
        "holdout_seeds_used": [9000, 9001],
        "ler_holdout": 0.1000000,
        "delta_ler_holdout": 0.0200000,
        "ler_shuffled": 0.1100000,
        "ler_holdout_ci": [0.08, 0.12],
    }
    replay = {
        "verdict": "VERIFIED",
        "holdout_seeds_used": [9000, 9001],
        "ler_holdout": 0.1000004,
        "delta_ler_holdout": 0.0199997,
        "ler_shuffled": 0.1099999,
        "ler_holdout_ci": [0.0800001, 0.1199999],
    }

    compare_verification_reports(original, replay, float_tol=1e-3)


def test_compare_verification_reports_rejects_mismatched_verdict() -> None:
    from autoqec.tools.advisor_replay import compare_verification_reports

    original = {
        "verdict": "VERIFIED",
        "holdout_seeds_used": [9000],
        "paired_eval_bundle_id": "bundle-1",
        "ler_holdout": 0.1,
        "delta_ler_holdout": 0.02,
        "ler_shuffled": 0.11,
        "ler_holdout_ci": [0.08, 0.12],
    }
    replay = {
        **original,
        "verdict": "FAILED",
    }

    with pytest.raises(AssertionError, match="verdict"):
        compare_verification_reports(original, replay, float_tol=1e-3)


def test_run_verify_offline_unsets_backend_env(monkeypatch, tmp_path: Path) -> None:
    from autoqec.tools import advisor_replay

    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    report_path = round_dir / "verification_report.json"

    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_CODER_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_ANALYST_BACKEND", "claude-cli")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example")
    monkeypatch.setenv("http_proxy", "http://proxy.example")
    monkeypatch.setenv("https_proxy", "https://proxy.example")

    captured: dict[str, object] = {}

    def fake_run(cmd, *, env, cwd, capture_output, text, check, encoding):
        del cwd, capture_output, text, check, encoding
        captured["cmd"] = cmd
        captured["env"] = env
        guard_root = Path(str(env["PYTHONPATH"]).split(":", 1)[0])
        captured["offline_guard_path"] = guard_root / "sitecustomize.py"
        report_path.write_text(
            json.dumps(
                {
                    "verdict": "SUSPICIOUS",
                    "holdout_seeds_used": [9000, 9001],
                    "paired_eval_bundle_id": "bundle-1",
                    "ler_holdout": 0.1,
                    "delta_ler_holdout": 0.0,
                    "ler_shuffled": 0.1,
                    "ler_holdout_ci": [0.08, 0.12],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="SUSPICIOUS\n", stderr="")

    monkeypatch.setattr(advisor_replay.subprocess, "run", fake_run)

    report = advisor_replay.run_verify_offline(
        round_dir,
        env_yaml="autoqec/envs/builtin/surface_d5_depol.yaml",
        python_bin="/tmp/fake-python",
        n_shots=64,
        n_seeds=2,
    )

    assert report["verdict"] == "SUSPICIOUS"
    assert captured["cmd"] == [
        "/tmp/fake-python",
        "-m",
        "cli.autoqec",
        "verify",
        str(round_dir),
        "--env",
        "autoqec/envs/builtin/surface_d5_depol.yaml",
        "--n-shots",
        "64",
        "--n-seeds",
        "2",
    ]
    env = captured["env"]
    assert "AUTOQEC_IDEATOR_BACKEND" not in env
    assert "AUTOQEC_CODER_BACKEND" not in env
    assert "AUTOQEC_ANALYST_BACKEND" not in env
    assert "HTTP_PROXY" not in env
    assert "HTTPS_PROXY" not in env
    assert "http_proxy" not in env
    assert "https_proxy" not in env
    assert env["NO_PROXY"] == "*"
    assert env["AUTOQEC_REPLAY_NO_NETWORK"] == "1"
    assert "sitecustomize.py" in str(captured["offline_guard_path"])


def test_replay_packaged_run_succeeds_without_original_report(monkeypatch, tmp_path: Path) -> None:
    from autoqec.tools import advisor_replay

    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    _write_manifest(round_dir)

    def fake_run_verify_offline(round_dir, *, env_yaml, python_bin, n_shots, n_seeds):
        del env_yaml, python_bin, n_shots, n_seeds
        report = {
            "verdict": "SUSPICIOUS",
            "holdout_seeds_used": [9000, 9001],
            "ler_holdout": 0.1,
            "delta_ler_holdout": 0.0,
            "ler_shuffled": 0.1,
            "ler_holdout_ci": [0.08, 0.12],
        }
        (round_dir / "verification_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(advisor_replay, "run_verify_offline", fake_run_verify_offline)

    result = advisor_replay.replay_packaged_run(
        run_dir,
        env_yaml="autoqec/envs/builtin/surface_d5_depol.yaml",
        python_bin=sys.executable,
        n_shots=8,
        n_seeds=2,
        extract_root=tmp_path / "replay",
    )

    assert result["package_path"].endswith("demo-run.tar.gz")
    assert result["original_report_copy_path"] is None
    assert Path(result["replay_report_path"]).exists()


def test_replay_packaged_run_accepts_prebuilt_package(monkeypatch, tmp_path: Path) -> None:
    from autoqec.tools import advisor_replay

    run_dir = tmp_path / "runs" / "demo-run"
    round_dir = run_dir / "round_1"
    round_dir.mkdir(parents=True)
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")
    _write_manifest(round_dir)
    package_path = advisor_replay.package_run_dir(run_dir)

    def fake_run_verify_offline(round_dir, *, env_yaml, python_bin, n_shots, n_seeds):
        del env_yaml, python_bin, n_shots, n_seeds
        report = {
            "verdict": "SUSPICIOUS",
            "holdout_seeds_used": [9000, 9001],
            "ler_holdout": 0.1,
            "delta_ler_holdout": 0.0,
            "ler_shuffled": 0.1,
            "ler_holdout_ci": [0.08, 0.12],
        }
        (round_dir / "verification_report.json").write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(advisor_replay, "run_verify_offline", fake_run_verify_offline)

    result = advisor_replay.replay_packaged_run(
        run_dir,
        package_path=package_path,
        env_yaml="autoqec/envs/builtin/surface_d5_depol.yaml",
        python_bin=sys.executable,
        n_shots=8,
        n_seeds=2,
        extract_root=tmp_path / "replay",
    )

    assert result["package_path"] == str(package_path)
    assert Path(result["replay_report_path"]).exists()


def test_no_network_sitecustomize_blocks_socket_connections(tmp_path: Path) -> None:
    from autoqec.tools.advisor_replay import _write_no_network_sitecustomize

    guard_root = tmp_path / "offline_guard"
    sitecustomize_path = _write_no_network_sitecustomize(guard_root)
    assert sitecustomize_path.exists()

    env = os.environ.copy()
    env["AUTOQEC_REPLAY_NO_NETWORK"] = "1"
    env["PYTHONPATH"] = (
        f"{guard_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(guard_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket; socket.create_connection(('127.0.0.1', 9), timeout=0.1)",
        ],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "blocks network access" in completed.stderr


def test_demo6_readme_documents_no_network_and_existing_demo_packaging() -> None:
    readme = Path("demos/demo-6-advisor-replay/README.md").read_text(encoding="utf-8").lower()

    assert "no-network" in readme
    assert "no-llm" in readme
    assert "demo 1" in readme or "demo 2" in readme
    assert "runs/<run_id>.tar.gz" in readme
    assert "package-run" in readme
    assert "artifact_manifest.json" in readme
    assert "repo sha" in readme
    assert "stochastic" in readme
