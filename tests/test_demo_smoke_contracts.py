from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cli.autoqec as autoqec_cli
from autoqec.runner.schema import RoundMetrics


def _extract_result_payload(stdout: str) -> dict:
    prefix = autoqec_cli.RESULT_PREFIX
    payload_lines = [line for line in stdout.splitlines() if line.startswith(prefix)]
    assert payload_lines, f"no result payload line found in stdout: {stdout!r}"
    return json.loads(payload_lines[-1][len(prefix) :])


@pytest.mark.integration
@pytest.mark.parametrize(
    "env_relpath",
    [
        "autoqec/envs/builtin/surface_d5_depol.yaml",
        "autoqec/envs/builtin/bb72_depol.yaml",
    ],
)
def test_no_llm_demo_smoke_writes_valid_artifact_contract(
    tmp_path: Path,
    env_relpath: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.autoqec",
            "run",
            str(repo_root / env_relpath),
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
        env=env,
    )

    payload = _extract_result_payload(completed.stdout)
    run_dir = Path(payload["run_dir"])
    round_dir = run_dir / "round_1"

    assert run_dir.exists()
    assert not (run_dir / "log.md").exists()
    assert not (run_dir / "pareto.json").exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "candidate_pareto.json").exists()
    assert payload["candidate_pareto_path"] == str(run_dir / "candidate_pareto.json")

    history_json = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    history_jsonl = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(history_json) == 1
    assert history_jsonl == history_json

    history_row = history_jsonl[0]
    assert history_row["round"] == 1
    metrics_from_history = RoundMetrics.model_validate(history_row)
    assert metrics_from_history.status == "ok"

    assert (round_dir / "config.yaml").exists()
    assert (round_dir / "train.log").exists()
    assert (round_dir / "checkpoint.pt").exists()
    assert (round_dir / "metrics.json").exists()
    assert (round_dir / "artifact_manifest.json").exists()

    metrics = RoundMetrics.model_validate_json(
        (round_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics.status == "ok"
    assert metrics.checkpoint_path == str(round_dir / "checkpoint.pt")
    assert metrics.training_log_path == str(round_dir / "train.log")
    assert Path(metrics.checkpoint_path).exists()
    assert Path(metrics.training_log_path).exists()
    assert metrics.n_params is not None and metrics.n_params > 0
    assert metrics.flops_per_syndrome is not None and metrics.flops_per_syndrome > 0

    manifest = json.loads((round_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["repo"]["commit_sha"]
    assert manifest["environment"]["env_yaml_path"] == str(repo_root / env_relpath)
    assert manifest["environment"]["env_yaml_sha256"]
    assert manifest["environment"]["env_yaml_sha256"] != "unavailable"
    assert manifest["round"]["command_line"][0] == sys.executable
    assert manifest["round"]["dsl_config_sha256"]
    assert manifest["artifacts"] == {
        "config_yaml": "config.yaml",
        "checkpoint": "checkpoint.pt",
        "metrics": "metrics.json",
        "train_log": "train.log",
    }

    candidate_pareto = json.loads((run_dir / "candidate_pareto.json").read_text(encoding="utf-8"))
    assert isinstance(candidate_pareto, list)
