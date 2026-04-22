"""Fork graph assembly for the Ideator context (§15.4).

The Ideator sees the full branching history (not just last-5 hypotheses) so
it can steer forks deliberately — pick a parent, see what worked on each
branch, avoid dead ends. This module owns the shape of that graph.

Full implementation lands in Task 10 / Phase E; this file ships the minimal
stub that satisfies `l3_for_ideator` in Task 6.
"""
from __future__ import annotations

from typing import Any


def build_fork_graph(
    history: list[dict],
    pareto: list[dict],
    run_id: str,
) -> dict[str, Any]:
    """Assemble the §15.4 fork_graph payload from disk L1.

    Nodes: one per branch in history + a synthetic "baseline" root.
    Edges: implicit via each node's `parent` field.
    Pareto front: list of branch names currently on the non-dominated front.
    """
    nodes: list[dict[str, Any]] = [
        {"branch": "baseline", "delta_vs_baseline": 0.0, "status": "baseline"}
    ]
    for row in history:
        if row.get("branch"):
            nodes.append(
                {
                    "branch": row["branch"],
                    "parent": row.get("fork_from", "baseline"),
                    "delta_vs_parent": row.get("delta_ler"),
                    "status": (row.get("status") or "ok").upper(),
                    "hypothesis_1line": (row.get("hypothesis") or "")[:80],
                    "on_pareto": any(
                        p.get("branch") == row["branch"] for p in pareto
                    ),
                }
            )
    return {
        "run_id": run_id,
        "nodes": nodes,
        "pareto_front": [p.get("branch") for p in pareto if p.get("branch")],
    }
