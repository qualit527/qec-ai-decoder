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
    # sentinel that a regression dropping dsl_schema_md routing would also wipe
    schema_md = "## PredecoderDSL authoritative source (sentinel-CA7E5F)"
    prompt = build_coder_prompt(
        hypothesis=hypothesis,
        mem=mem,
        dsl_schema_md=schema_md,
    )
    assert "CODER" in prompt
    assert "gated_mlp" in prompt
    # tier2_validator_rules must reach the Coder so it can honour its contract
    assert "slot_signatures" in prompt
    assert "forbidden_names" in prompt
    # dsl_schema_md must actually land in the prompt payload
    assert "sentinel-CA7E5F" in prompt


def test_run_round_plan_threads_fork_from(tmp_path: Path) -> None:
    from autoqec.envs.schema import (
        CodeSpec,
        ConstraintsSpec,
        EnvSpec,
        NoiseSpec,
        SeedPolicy,
    )
    from autoqec.orchestration.loop import run_round_plan

    env = EnvSpec(
        name="test",
        code=CodeSpec(type="stim_circuit", source="circuits/surface_d5.stim"),
        noise=NoiseSpec(type="depolarizing", p=[1e-3], seed_policy=SeedPolicy()),
        constraints=ConstraintsSpec(),
        baseline_decoders=["pymatching"],
        classical_backend="mwpm",
    )
    plan = run_round_plan(
        env_spec=env,
        run_dir=tmp_path,
        round_idx=1,
        machine_state={"gpu": {}},
        kb_excerpt="",
        dsl_schema_md="",
        fork_from="exp/t/02-a",  # NEW param
    )
    assert plan["fork_from"] == "exp/t/02-a"


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
