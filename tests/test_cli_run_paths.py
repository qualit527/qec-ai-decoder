from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cli.autoqec import _candidate_pareto


def test_candidate_pareto_keeps_only_nondominated_successful_records() -> None:
    front = _candidate_pareto(
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


def test_run_cli_works_from_foreign_cwd_and_writes_pareto(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"

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
    )

    payload = json.loads(completed.stdout[completed.stdout.rfind("{") :])
    run_dir = tmp_path / payload["run_dir"]

    assert run_dir.exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "pareto.json").exists()

    pareto = json.loads((run_dir / "pareto.json").read_text(encoding="utf-8"))
    assert isinstance(pareto, list)
    assert payload["pareto_path"].endswith("pareto.json")
