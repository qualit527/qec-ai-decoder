"""Unit tests for Chen Jiahan's Day-1 orchestration skeleton (A1.6).

All tests are CPU-only and avoid heavy deps (no torch, no stim).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─── RunMemory ─────────────────────────────────────────────────────────


def test_run_memory_initializes_files(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run1")
    assert (tmp_path / "run1").is_dir()
    # pareto.json seeded with empty list
    assert json.loads(mem.pareto_path.read_text()) == []


def test_run_memory_append_round_and_snapshot(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run2")
    for idx in range(5):
        mem.append_round(
            {
                "round": idx + 1,
                "hypothesis": f"try variant {idx}",
                "status": "ok",
                "delta_ler": 1e-5 * idx,
            }
        )

    snap = mem.l2_snapshot(k_last=3)
    assert snap["rounds_so_far"] == 5
    assert len(snap["last_rounds"]) == 3
    assert snap["last_rounds"][-1]["round"] == 5
    assert snap["pareto"] == []


def test_run_memory_update_pareto_roundtrips(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run3")
    pareto = [
        {"round": 2, "delta_ler": 5e-5, "flops": 1000, "n_params": 500},
        {"round": 4, "delta_ler": 8e-5, "flops": 2000, "n_params": 700},
    ]
    mem.update_pareto(pareto)
    snap = mem.l2_snapshot()
    assert snap["pareto"] == pareto


def test_l3_for_ideator_contains_required_keys(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run4")
    mem.append_round({"round": 1, "hypothesis": "h1", "status": "ok"})
    mem.append_round({"round": 2, "hypothesis": "h2", "status": "killed_by_safety"})

    ctx = mem.l3_for_ideator(
        env_spec={"name": "surface_d5_depol"},
        kb_excerpt="building block catalogue ...",
        machine_state={"gpu": {"vram_free_gb": 20.0}},
    )
    assert set(ctx.keys()) >= {
        "env_spec",
        "pareto_front",
        "last_5_hypotheses",
        "knowledge_excerpts",
        "machine_state_hint",
    }
    # last_5_hypotheses must carry status so Ideator can avoid re-proposing killed ones
    assert ctx["last_5_hypotheses"][-1]["status"] == "killed_by_safety"


def test_l3_for_coder_and_analyst(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run5")
    coder_ctx = mem.l3_for_coder(
        hypothesis={"hypothesis": "try GNN"}, schema_md="...", best_so_far=[]
    )
    assert "hypothesis" in coder_ctx and "dsl_schema" in coder_ctx

    analyst_ctx = mem.l3_for_analyst(
        round_dir=tmp_path / "run5" / "round_1",
        prev_summary="prior summary",
        pareto=[{"round": 1}],
    )
    assert analyst_ctx["metrics_path"].endswith("metrics.json")
    assert analyst_ctx["previous_summary"] == "prior summary"


# ─── Dispatcher prompt ↔ response roundtrip ────────────────────────────


def test_build_prompt_mentions_role_and_context() -> None:
    from autoqec.agents.dispatch import build_prompt

    prompt = build_prompt("ideator", {"env_spec": {"name": "surface_d5_depol"}})
    assert "IDEATOR" in prompt
    assert "surface_d5_depol" in prompt
    assert "json" in prompt.lower()  # reminds the subagent to emit fenced JSON


def test_parse_response_extracts_first_json_block() -> None:
    from autoqec.agents.dispatch import parse_response

    text = (
        "Here is my proposal.\n\n"
        "```json\n"
        '{"hypothesis": "h", "expected_delta_ler": 1e-4, '
        '"expected_cost_s": 10, "rationale": "r"}\n'
        "```\n"
        "(trailing chatter)"
    )
    parsed = parse_response("ideator", text)
    assert parsed["hypothesis"] == "h"
    assert parsed["expected_delta_ler"] == 1e-4


def test_parse_response_errors_when_no_block() -> None:
    from autoqec.agents.dispatch import parse_response

    with pytest.raises(ValueError, match="No JSON block"):
        parse_response("coder", "sorry, no fenced block here")


# ─── Loop skeleton ────────────────────────────────────────────────────


def test_run_round_plan_emits_ideator_prompt(tmp_path: Path) -> None:
    yaml_text = """
name: test_env
code:
  type: stim_circuit
  source: circuits/surface_d5.stim
noise:
  type: depolarizing
  p: [1.0e-3]
constraints:
  latency_flops_budget: 1000000
  param_budget: 10000
  target_ler: 1.0e-4
  target_p: 1.0e-3
baseline_decoders: [pymatching]
classical_backend: mwpm
"""
    yaml_path = tmp_path / "env.yaml"
    yaml_path.write_text(yaml_text)

    from autoqec.envs.schema import load_env_yaml
    from autoqec.orchestration.loop import run_round_plan

    env = load_env_yaml(yaml_path)
    plan = run_round_plan(
        env_spec=env,
        run_dir=tmp_path / "run",
        round_idx=1,
        machine_state={
            "gpu": {"vram_free_gb": 20.0},
            "budget": {"total_wallclock_s_remaining": 3600},
        },
        kb_excerpt="catalogue",
        dsl_schema_md="schema",
    )
    assert plan["round_idx"] == 1
    assert plan["round_dir"].endswith("round_1")
    assert "IDEATOR" in plan["ideator_prompt"]
    assert "test_env" in plan["ideator_prompt"]
