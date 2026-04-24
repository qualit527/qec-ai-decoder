from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import cli.autoqec as autoqec_cli

from autoqec.envs.schema import load_env_yaml


def _extract_result_payload(stdout: str) -> dict:
    prefix = getattr(autoqec_cli, "RESULT_PREFIX", None)
    assert isinstance(prefix, str) and prefix, "cli.autoqec must expose a parseable result prefix"
    payload_lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    assert payload_lines, f"no result payload line found in stdout: {stdout!r}"
    return json.loads(payload_lines[-1][len(prefix) :])


def test_candidate_pareto_keeps_only_nondominated_successful_records() -> None:
    front = autoqec_cli._candidate_pareto(
        [
            {
                "round": 1,
                "status": "ok",
                "delta_ler": 0.02,
                "flops_per_syndrome": 1000,
                "n_params": 200,
                "checkpoint_path": "runs/r1/checkpoint.pt",
            },
            {
                "round": 2,
                "status": "ok",
                "delta_ler": 0.02,
                "flops_per_syndrome": 1500,
                "n_params": 200,
                "checkpoint_path": "runs/r2/checkpoint.pt",
            },
            {
                "round": 3,
                "status": "ok",
                "delta_ler": 0.01,
                "flops_per_syndrome": 500,
                "n_params": 100,
                "checkpoint_path": "runs/r3/checkpoint.pt",
            },
            {
                "round": 4,
                "status": "compile_error",
                "delta_ler": None,
                "flops_per_syndrome": None,
                "n_params": None,
                "checkpoint_path": None,
            },
            {
                "round": 5,
                "status": "ok",
                "delta_ler": 0.02,
                "flops_per_syndrome": 1000,
                "n_params": 200,
                "checkpoint_path": "runs/r5/checkpoint.pt",
            },
        ]
    )

    assert [item["round"] for item in front] == [1, 3]
    assert all(item["verified"] is False for item in front)


def test_load_example_templates_includes_dev_safe_templates() -> None:
    loader = getattr(autoqec_cli, "load_example_templates", None)
    assert callable(loader), "cli.autoqec must provide a template loader"
    template_names = {name for name, _ in loader()}
    assert {"gnn_small", "gnn_gated", "neural_bp_min"}.issubset(template_names)


def test_pyproject_declares_example_yaml_as_package_data() -> None:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]["autoqec"]
    assert "example_db/*.yaml" in package_data


def test_load_env_yaml_resolves_code_source_via_repo_root_fallback() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"
    sibling_candidate = env_yaml.parent / "circuits/surface_d5.stim"

    assert not sibling_candidate.exists()

    env = load_env_yaml(env_yaml)

    assert env.code.source == str((repo_root / "circuits/surface_d5.stim").resolve())


@pytest.mark.slow
def test_run_cli_works_from_foreign_cwd_and_writes_candidate_pareto(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )
    # Force UTF-8 on the child process's stdio so locales like Windows-zh-CN
    # (CP936/GBK default) do not emit bytes the parent's utf-8 pipe can't decode.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.autoqec",
            "run",
            str(env_yaml),
            "--rounds",
            "1",
            "--profile",
            "dev",
            "--no-llm",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    payload = _extract_result_payload(completed.stdout)
    run_dir = Path(payload["run_dir"])

    assert run_dir.exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert not (run_dir / "pareto.json").exists()
    assert (run_dir / "candidate_pareto.json").exists()

    pareto = json.loads((run_dir / "candidate_pareto.json").read_text(encoding="utf-8"))
    assert isinstance(pareto, list)
    assert payload["candidate_pareto_path"].endswith("candidate_pareto.json")


@pytest.mark.slow
def test_run_cli_no_llm_appends_one_history_row_per_round(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.autoqec",
            "run",
            str(env_yaml),
            "--rounds",
            "2",
            "--profile",
            "dev",
            "--no-llm",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    payload = _extract_result_payload(completed.stdout)
    run_dir = Path(payload["run_dir"])
    history_lines = (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(history_lines) == 2
    rows = [json.loads(line) for line in history_lines]
    assert [row["round"] for row in rows] == [1, 2]


def test_run_cli_no_llm_accepts_template_name_and_pins_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.autoqec",
            "run",
            str(env_yaml),
            "--rounds",
            "1",
            "--profile",
            "dev",
            "--no-llm",
            "--template-name",
            "gnn_small",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    payload = _extract_result_payload(completed.stdout)
    run_dir = Path(payload["run_dir"])
    assert run_dir.exists()
    round_cfg = json.loads((run_dir / "round_1" / "metrics.json").read_text(encoding="utf-8"))
    assert round_cfg["status"] == "ok"
    saved_yaml = (run_dir / "round_1" / "config.yaml").read_text(encoding="utf-8")
    assert "hidden_dim: 16" in saved_yaml
