from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demos/demo-7-gpu-positive-ai-loop"


def test_demo7_readme_documents_one_command_ai_loop() -> None:
    text = (DEMO / "README.md").read_text(encoding="utf-8")

    assert "bash demos/demo-7-gpu-positive-ai-loop/run.sh" in text
    assert "without `--no-llm`" in text
    assert "orchestrator_trace.md" in text
    assert "delta_ler" in text


def test_demo7_positive_summary_has_trace_and_positive_delta() -> None:
    summary = json.loads(
        (DEMO / "expected_output/live_loop_positive_summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["metrics"]["status"] == "ok"
    assert summary["metrics"]["delta_ler"] > 0
    assert summary["metrics"]["ler_predecoder"] < summary["metrics"]["ler_plain_classical"]
    assert "runner metrics" in summary["agent_trace"]["sections"]
    assert "run complete" in summary["agent_trace"]["sections"]
