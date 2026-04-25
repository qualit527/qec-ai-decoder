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


def test_build_coder_prompt_falls_back_to_history_when_pareto_empty(tmp_path: Path) -> None:
    """Regression (2026-04-24): cold-start Pareto trap.

    Before a VERIFIED round exists `pareto.json` is `[]`, so the Coder
    previously saw `best_so_far: []` and had no reference configs to
    mutate from. We fall back to the top-3 `status=ok` history rows by
    `delta_ler` so the Coder always has *something* to steer off of.
    """
    from autoqec.orchestration.loop import build_coder_prompt
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run")
    # history has 3 ok rounds — none verified (pareto stays []).
    history_rows = [
        {"round": 1, "status": "ok", "delta_ler": -1e-4, "flops_per_syndrome": 1000, "n_params": 100,
         "train_loss_final": 0.01, "checkpoint_path": "r1.pt"},
        {"round": 2, "status": "ok", "delta_ler": 3e-5, "flops_per_syndrome": 2000, "n_params": 200,
         "train_loss_final": 0.008, "checkpoint_path": "r2.pt"},
        {"round": 3, "status": "ok", "delta_ler": 1e-5, "flops_per_syndrome": 1500, "n_params": 150,
         "train_loss_final": 0.007, "checkpoint_path": "r3.pt"},
    ]
    with (mem.run_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for row in history_rows:
            f.write(json.dumps(row) + "\n")

    hypothesis = {
        "hypothesis": "try a bigger GNN",
        "expected_delta_ler": 5e-5,
        "expected_cost_s": 120,
        "rationale": "empty pareto — lean on history",
    }
    prompt = build_coder_prompt(hypothesis=hypothesis, mem=mem, dsl_schema_md="")

    # The fallback should surface round 2 (best delta) as the top
    # reference, and the cold-start flag should be visible to the Coder
    # so it knows these rows are NOT VERIFIED.
    payload_start = prompt.index("{")
    payload = json.loads(prompt[payload_start : prompt.rindex("}") + 1])
    best = payload["best_so_far"]
    assert best, "best_so_far must not be empty when ok-history exists"
    assert best[0].get("round") == 2, f"expected round 2 (best delta) first, got {best[0]}"
    assert best[0].get("cold_start_fallback") is True, (
        "each fallback entry must be flagged as not VERIFIED"
    )


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
