from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPO_ROOT / "demos/demo-1-surface-d5/README.md"
RUN_SCRIPT_PATH = REPO_ROOT / "demos/demo-1-surface-d5/run_quick.sh"


def test_demo1_script_uses_authoritative_result_payload() -> None:
    script = RUN_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "AUTOQEC_RESULT_JSON=" in script
    assert "candidate_pareto.json" in script
    assert "TEMPLATE_NAME" in script
    assert "--template-name" in script
    assert "ls -t runs" not in script


def test_demo1_readme_describes_no_llm_candidate_path() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "Path A — LLM loop" in readme
    assert "Path B — no-LLM baseline" in readme
    assert "candidate_pareto.json" in readme
    assert "unverified candidate path" in readme
    assert "without GPU in dev mode" in readme
    assert "TEMPLATE_NAME=gnn_small" in readme
    assert "random dev-safe template" not in readme


def test_demo1_expected_output_mentions_candidate_snapshot() -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    assert "expected_output/no_llm_round1/history.jsonl" in readme
    assert "expected_output/no_llm_round1/round_1_metrics.json" in readme
    assert "candidate_pareto.json" in readme


def test_demo1_expected_output_contains_candidate_snapshot() -> None:
    expected_root = REPO_ROOT / "demos/demo-1-surface-d5/expected_output/no_llm_round1"

    assert (expected_root / "history.jsonl").exists()
    assert (expected_root / "history.json").exists()
    assert (expected_root / "candidate_pareto.json").exists()
    assert (expected_root / "round_1_config.yaml").exists()
    assert (expected_root / "round_1_metrics.json").exists()
