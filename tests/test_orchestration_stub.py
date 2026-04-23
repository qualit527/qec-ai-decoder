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


def test_l3_for_ideator_returns_fork_graph(tmp_path: Path) -> None:
    """The Ideator context exposes fork_graph per §15.4, not last_5_hypotheses."""
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run4")
    mem.append_round(
        {
            "round": 1,
            "status": "ok",
            "delta_ler": 4e-4,
            "flops_per_syndrome": 180_000,
            "n_params": 42_000,
            "branch": "exp/t/01-a",
            "commit_sha": "sha1",
            "round_attempt_id": "u1",
            "fork_from": "baseline",
            "hypothesis": "test",
        }
    )

    ctx = mem.l3_for_ideator(
        env_spec={"name": "surface_d5"},
        kb_excerpt="(excerpt)",
        machine_state={"gpu": {}},
        run_id="t",
    )
    assert "fork_graph" in ctx
    assert "nodes" in ctx["fork_graph"]
    assert any(n.get("branch") == "exp/t/01-a" for n in ctx["fork_graph"]["nodes"])
    assert "baseline" in {n.get("branch") for n in ctx["fork_graph"]["nodes"]}
    # last_5_hypotheses is gone.
    assert "last_5_hypotheses" not in ctx


def test_l3_for_coder_includes_tier2_validator_rules(tmp_path: Path) -> None:
    """Codex review (medium): coder.md declares `tier2_validator_rules` as input,
    so l3_for_coder must provide it or the subagent cannot honour its own contract."""
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run5")
    coder_ctx = mem.l3_for_coder(
        hypothesis={"hypothesis": "try GNN"}, schema_md="...", best_so_far=[]
    )
    assert {"hypothesis", "dsl_schema", "best_so_far", "tier2_validator_rules"} <= set(coder_ctx)
    rules = coder_ctx["tier2_validator_rules"]
    # must tell the Coder the exact slot signatures and the CustomFn object shape
    assert "slot_signatures" in rules
    assert rules["slot_signatures"]["message_fn"] == ["x_src", "x_dst", "e_ij", "params"]
    assert "forbidden_names" in rules and "os" in rules["forbidden_names"]
    assert "output_shape" in rules and rules["output_shape"]["type"] == "custom"


def test_l3_for_analyst_metrics_path_is_absolute(tmp_path: Path) -> None:
    """Codex review (medium): analyst.md says `metrics_path` is absolute;
    l3_for_analyst must resolve before serialising, even when given a
    relative round_dir."""
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run5b")
    # deliberately pass a relative path — this is what triggered the finding
    relative_round_dir = Path("runs") / "any" / "round_1"
    analyst_ctx = mem.l3_for_analyst(
        round_dir=relative_round_dir,
        prev_summary="prior summary",
        pareto=[{"round": 1}],
    )
    assert analyst_ctx["metrics_path"].endswith("metrics.json")
    assert Path(analyst_ctx["metrics_path"]).is_absolute()
    assert analyst_ctx["previous_summary"] == "prior summary"


# ─── Dispatcher prompt ↔ response roundtrip ────────────────────────────


def test_build_prompt_mentions_role_and_context() -> None:
    from autoqec.agents.dispatch import build_prompt

    prompt = build_prompt("ideator", {"env_spec": {"name": "surface_d5_depol"}})
    assert "IDEATOR" in prompt
    assert "surface_d5_depol" in prompt
    assert "json" in prompt.lower()  # reminds the subagent to emit fenced JSON


def test_ideator_prompt_contains_fork_graph_and_not_legacy_hypotheses() -> None:
    """Phase 2.4.1 — the rendered ideator prompt string must carry fork_graph
    (per §15.4) and must not carry the pre-§15 last_5_hypotheses key even if
    a caller accidentally passes it in, the plain JSON serialisation will
    surface any regression."""
    from autoqec.agents.dispatch import build_prompt

    ctx = {
        "env_spec": {"name": "surface_d5_depol"},
        "fork_graph": {
            "nodes": [
                {"branch": "baseline", "parent": None},
                {"branch": "exp/t/01-a", "parent": "baseline", "on_pareto": True},
            ]
        },
        "pareto": [{"branch": "exp/t/01-a", "delta_ler": 1e-4}],
        "machine_state": {"gpu": {}},
    }
    prompt = build_prompt("ideator", ctx)
    assert "fork_graph" in prompt
    assert "exp/t/01-a" in prompt
    assert "last_5_hypotheses" not in prompt


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


def test_parse_response_enforces_ideator_schema() -> None:
    """Codex open-question: validate §2.5 shape, not just JSON well-formedness."""
    from autoqec.agents.dispatch import parse_response

    good = (
        "```json\n"
        '{"hypothesis": "h", "expected_delta_ler": 1e-4, '
        '"expected_cost_s": 10, "rationale": "r"}\n'
        "```"
    )
    parsed = parse_response("ideator", good)
    assert parsed["hypothesis"] == "h"

    missing_field = '```json\n{"hypothesis": "h"}\n```'
    with pytest.raises(ValueError, match="ideator response"):
        parse_response("ideator", missing_field)


def test_parse_response_names_missing_required_rationale() -> None:
    from autoqec.agents.dispatch import parse_response

    missing_rationale = (
        "```json\n"
        '{"hypothesis": "h", "expected_delta_ler": 1e-4, "expected_cost_s": 10}\n'
        "```"
    )

    with pytest.raises(ValueError, match=r"rationale"):
        parse_response("ideator", missing_rationale)


def test_parse_response_enforces_coder_tier_value() -> None:
    from autoqec.agents.dispatch import parse_response

    bad_tier = (
        '```json\n{"tier": "3", "dsl_config": {"type": "gnn"}, "rationale": "r"}\n```'
    )
    with pytest.raises(ValueError, match="coder response"):
        parse_response("coder", bad_tier)


def test_parse_response_enforces_analyst_verdict() -> None:
    from autoqec.agents.dispatch import parse_response

    bad_verdict = (
        '```json\n{"summary_1line": "s", "verdict": "maybe", '
        '"next_hypothesis_seed": "x"}\n```'
    )
    with pytest.raises(ValueError, match="analyst response"):
        parse_response("analyst", bad_verdict)


# ─── UTF-8 encoding ───────────────────────────────────────────────────


def test_run_memory_append_round_rejects_schema_violation(tmp_path: Path) -> None:
    """M3: history.jsonl must not grow rows that violate §15.2 invariants.

    Before the guard, ``append_round`` happily json.dumps'd any dict, so a
    caller could drop a row with both ``round_attempt_id`` and ``reconcile_id``
    set, or a ``branch`` with no ``commit_sha`` on an ok-status row. The
    schema exists, but it only fired when a caller voluntarily constructed
    a ``RoundMetrics`` instance — which many paths didn't.
    """
    import pytest

    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run_m3")

    # Both round_attempt_id and reconcile_id set — mutual-exclusion violation.
    with pytest.raises(ValueError, match="mutually exclusive"):
        mem.append_round(
            {
                "status": "ok",
                "branch": "exp/t/01-a",
                "commit_sha": "abc",
                "round_attempt_id": "u1",
                "reconcile_id": "r1",
            }
        )
    # History file is untouched after the failed append.
    assert not mem.history_path.exists() or mem.history_path.read_text() == ""

    # Branch set but commit_sha missing on an ok row — also rejected.
    with pytest.raises(ValueError, match="commit_sha"):
        mem.append_round(
            {
                "status": "ok",
                "branch": "exp/t/01-a",
                "round_attempt_id": "u1",
            }
        )

    # A valid row still writes.
    mem.append_round(
        {
            "status": "ok",
            "round": 1,
            "branch": "exp/t/01-a",
            "commit_sha": "abc",
            "round_attempt_id": "u1",
        }
    )
    assert mem.history_path.read_text(encoding="utf-8").strip() != ""


def test_run_memory_append_log_roundtrips_utf8(tmp_path: Path) -> None:
    """Codex review (medium): log.md / jsonl must be UTF-8 on Windows too."""
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run_utf8")
    chinese = "第 1 轮：尝试 GNN 带门控消息 — Δ LER ≈ 5e-5"
    mem.append_log(chinese)
    mem.append_round({"round": 1, "status": "ok", "note": chinese})

    assert mem.log_path.read_text(encoding="utf-8").strip() == chinese
    snap = mem.l2_snapshot()
    assert snap["last_rounds"][-1]["note"] == chinese


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
