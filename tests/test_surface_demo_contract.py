from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "demos/demo-1-surface-d5/README.md"
NO_LLM_SCRIPT_PATH = REPO_ROOT / "demos/demo-1-surface-d5/run_quick.sh"
LIVE_SCRIPT_PATH = REPO_ROOT / "demos/demo-1-surface-d5/run_live.sh"


def test_demo1_no_llm_script_uses_authoritative_result_payload() -> None:
    script = NO_LLM_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "AUTOQEC_RESULT_JSON=" in script
    assert "candidate_pareto.json" in script
    assert "TEMPLATE_NAME" in script
    assert "--template-name" in script
    assert "ls -t runs" not in script


def test_demo1_live_script_sets_github_models_defaults_and_reports_run_dir() -> None:
    script = LIVE_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "AUTOQEC_IDEATOR_BACKEND" in script
    assert "github-models" in script
    assert "AUTOQEC_RESULT_JSON=" in script
    assert "history.jsonl" in script
    assert "torch.version.cuda" in script
    assert "torch.cuda.get_device_name(0)" in script


def test_demo1_readme_describes_live_gpu_validation_contract() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "Path A - live LLM CLI" in readme
    assert "Path B - no-LLM baseline" in readme
    assert "github-models" in readme
    assert "AUTOQEC_IDEATOR_BACKEND" in readme
    assert "torch.version.cuda" in readme
    assert "vram_peak_gb" in readme
    assert "CPU vs GPU" in readme
    assert "expected_output/live_llm_gpu_round3/history.jsonl" in readme
    assert "expected_output/live_llm_gpu_round3/runtime_env.json" in readme


def test_demo1_expected_output_contains_live_gpu_snapshot() -> None:
    expected_root = REPO_ROOT / "demos/demo-1-surface-d5/expected_output/live_llm_gpu_round3"

    required = [
        expected_root / "history.jsonl",
        expected_root / "log.md",
        expected_root / "pareto.json",
        expected_root / "runtime_env.json",
        expected_root / "cpu_gpu_same_config_comparison.json",
        expected_root / "round_1_config.yaml",
        expected_root / "round_1_metrics.json",
    ]
    for path in required:
        assert path.exists(), path

    runtime_env = json.loads((expected_root / "runtime_env.json").read_text(encoding="utf-8"))
    assert runtime_env["backend_matrix"]["ideator"]["backend"] == "github-models"
    assert runtime_env["torch"]["cuda_available"] is True

    comparison = json.loads(
        (expected_root / "cpu_gpu_same_config_comparison.json").read_text(encoding="utf-8")
    )
    assert comparison["gpu"]["vram_peak_gb"] > 0
    assert comparison["gpu"]["train_wallclock_s"] < comparison["cpu"]["train_wallclock_s"]
