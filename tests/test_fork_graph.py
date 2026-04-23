"""Tests for autoqec.orchestration.fork_graph (§15.4).

Covers the richer serialization implemented in Task 10 — baseline root,
committed rounds, compose-conflict markers, Pareto flags, and the
non-dominated helper.
"""
from __future__ import annotations

from autoqec.orchestration.fork_graph import build_fork_graph, non_dominated


def test_empty_run_returns_baseline_only():
    g = build_fork_graph(history=[], pareto=[], run_id="t")
    assert len(g["nodes"]) == 1
    assert g["nodes"][0]["branch"] == "baseline"


def test_compose_conflict_node_included():
    history = [
        {
            "round": 12,
            "status": "compose_conflict",
            "fork_from": ["exp/t/02-a", "exp/t/04-b"],
            "fork_from_canonical": "exp/t/02-a|exp/t/04-b",
            "round_attempt_id": "u12",
            "conflicting_files": ["autoqec/decoders/modules/gnn.py"],
        },
    ]
    g = build_fork_graph(history=history, pareto=[], run_id="t")
    compose_nodes = [n for n in g["nodes"] if n.get("status") == "FAILED_COMPOSE"]
    assert len(compose_nodes) == 1
    assert compose_nodes[0].get("parents") == ["exp/t/02-a", "exp/t/04-b"]


def test_on_pareto_flag():
    history = [
        {
            "round": 1,
            "status": "ok",
            "branch": "exp/t/01-a",
            "commit_sha": "s1",
            "round_attempt_id": "u1",
            "fork_from": "baseline",
            "delta_ler": 1e-4,
            "hypothesis": "test",
        },
    ]
    pareto = [{"branch": "exp/t/01-a", "commit_sha": "s1", "delta_ler": 1e-4}]
    g = build_fork_graph(history=history, pareto=pareto, run_id="t")
    node = next(n for n in g["nodes"] if n.get("branch") == "exp/t/01-a")
    assert node["on_pareto"] is True


def test_non_dominated_filter():
    points = [
        {"delta_ler": 4e-4, "flops_per_syndrome": 200_000, "n_params": 40_000, "id": "a"},
        {"delta_ler": 2e-4, "flops_per_syndrome": 50_000, "n_params": 20_000, "id": "b"},
        {"delta_ler": 1e-4, "flops_per_syndrome": 300_000, "n_params": 60_000, "id": "c"},
    ]
    out = non_dominated(points)
    ids = {p["id"] for p in out}
    assert ids == {"a", "b"}
