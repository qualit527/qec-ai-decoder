from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_demo_2_readme_calls_out_osd_and_unverified_candidate_pareto() -> None:
    readme = (REPO_ROOT / "demos/demo-2-bb72/README.md").read_text(encoding="utf-8")

    assert "classical_backend: osd" in readme
    assert "MWPM" in readme
    assert "candidate Pareto" in readme
    assert "unverified" in readme.lower()


def test_demo_2_runtime_doc_calls_out_cpu_gpu_and_osd_cost() -> None:
    runtime_doc = REPO_ROOT / "demos/demo-2-bb72/runtime.md"

    text = runtime_doc.read_text(encoding="utf-8")
    assert "CPU" in text
    assert "GPU" in text
    assert "OSD" in text


def test_demo_2_expected_output_snapshot_contains_successful_round() -> None:
    snapshot_dir = REPO_ROOT / "demos/demo-2-bb72/expected_output/sample_run"

    history = [
        json.loads(line)
        for line in (snapshot_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metrics = json.loads((snapshot_dir / "round_1_metrics.json").read_text(encoding="utf-8"))
    candidate_pareto = json.loads((snapshot_dir / "candidate_pareto.json").read_text(encoding="utf-8"))
    config_text = (snapshot_dir / "round_1_config.yaml").read_text(encoding="utf-8")

    assert history
    assert history[0]["status"] == "ok"
    assert metrics["status"] == "ok"
    assert isinstance(candidate_pareto, list)
    assert candidate_pareto
    assert "type: gnn" in config_text
