"""Unit tests for B1 loop helpers (autoqec/orchestration/loop.py additions)."""
from __future__ import annotations

import json
from pathlib import Path


def test_build_coder_prompt_contains_hypothesis_and_rules(tmp_path: Path) -> None:
    from autoqec.orchestration.loop import build_coder_prompt
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run")
    hypothesis = {
        "hypothesis": "try gated_mlp message_fn with 3 layers",
        "expected_delta_ler": 5e-5,
        "expected_cost_s": 120,
        "rationale": "Pareto has no entry with gated_mlp yet",
    }
    prompt = build_coder_prompt(
        hypothesis=hypothesis,
        mem=mem,
        dsl_schema_md="## PredecoderDSL\n...",
    )
    assert "CODER" in prompt
    assert "gated_mlp" in prompt
    # tier2_validator_rules must reach the Coder so it can honour its contract
    assert "slot_signatures" in prompt
    assert "forbidden_names" in prompt


def test_build_analyst_prompt_has_absolute_metrics_path(tmp_path: Path) -> None:
    from autoqec.orchestration.loop import build_analyst_prompt
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run")
    # analyst prompt embeds metrics.json path + previous round summary
    round_dir = Path("runs") / "demo" / "round_1"  # deliberately relative
    prompt = build_analyst_prompt(
        mem=mem,
        round_dir=round_dir,
        prev_summary="round 0 was plain-classical baseline",
    )
    assert "ANALYST" in prompt
    # absolute path must appear in prompt
    payload_start = prompt.index("{")
    payload = json.loads(prompt[payload_start : prompt.rindex("}") + 1])
    assert Path(payload["metrics_path"]).is_absolute()
    assert payload["metrics_path"].endswith("metrics.json")
    assert payload["previous_summary"].startswith("round 0")
