"""Fork-graph assembly for the Ideator's L3 context (§15.4).

Nodes cover the synthetic ``baseline`` root, every committed round, and
every ``compose_conflict`` dead-end. Each node carries enough metadata
for the Ideator to steer forks deliberately — parent linkage, Pareto
membership, delta vs parent/baseline, and a one-line hypothesis summary.
"""
from __future__ import annotations

from typing import Any


def _history_node_branch(row: dict[str, Any]) -> str | None:
    """Return the node key used in fork_graph for one history row.

    Worktree-backed rounds already have a git branch name. The no-LLM path does
    not, so synthesize a stable node id from the round index to keep
    ``fork_graph.json`` structurally useful for artifact review.
    """
    branch = row.get("branch")
    if branch:
        return str(branch)
    round_idx = row.get("round")
    if isinstance(round_idx, int):
        return f"round_{round_idx}"
    return None


def non_dominated(points: list[dict]) -> list[dict]:
    """Return the non-dominated subset over (+delta_ler, -flops, -n_params).

    A point ``p`` dominates ``q`` if it is at least as good on every axis
    and strictly better on at least one. Missing values are treated as 0.
    """

    def dominates(a: dict, b: dict) -> bool:
        ad, bd = float(a.get("delta_ler") or 0), float(b.get("delta_ler") or 0)
        af, bf = (
            int(a.get("flops_per_syndrome") or 0),
            int(b.get("flops_per_syndrome") or 0),
        )
        ap, bp = int(a.get("n_params") or 0), int(b.get("n_params") or 0)
        return (ad >= bd and af <= bf and ap <= bp) and (
            ad > bd or af < bf or ap < bp
        )

    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def build_fork_graph(
    history: list[dict],
    pareto: list[dict],
    run_id: str,
) -> dict[str, Any]:
    """Serialize the fork graph (§15.4) for the Ideator's L3 context.

    Includes the synthetic ``baseline`` root, every committed round, and
    every ``compose_conflict`` dead-end. Each committed node carries its
    parent, delta-vs-parent/baseline, Pareto flag, and a one-line
    hypothesis summary; conflict nodes carry the parent set and the list
    of conflicting files for downstream pruning.
    """
    pareto_branches = {p.get("branch") for p in pareto if p.get("branch")}
    nodes: list[dict[str, Any]] = [
        {
            "branch": "baseline",
            "delta_vs_baseline": 0.0,
            "ler": None,
            "flops": 0,
            "n_params": 0,
            "status": "baseline",
        }
    ]

    for row in history:
        status = row.get("status") or ""
        if status == "compose_conflict":
            parents = row.get("fork_from") if isinstance(row.get("fork_from"), list) else []
            conflicting = row.get("conflicting_files") or []
            nodes.append(
                {
                    "branch": None,
                    "round_attempt_id": row.get("round_attempt_id"),
                    "parents": list(parents),
                    "fork_from_canonical": row.get("fork_from_canonical"),
                    "status": "FAILED_COMPOSE",
                    "failure_reason": (
                        f"git merge conflict in {', '.join(conflicting)}"
                        if conflicting
                        else "git merge conflict"
                    ),
                    "hypothesis_1line": (row.get("hypothesis") or "")[:80],
                }
            )
        else:
            node_branch = _history_node_branch(row)
            if node_branch is None:
                continue
            nodes.append(
                {
                    "branch": node_branch,
                    "commit_sha": row.get("commit_sha"),
                    "parent": row.get("fork_from", "baseline"),
                    "delta_vs_parent": row.get("delta_vs_parent") or row.get("delta_ler"),
                    "delta_vs_baseline": row.get("delta_vs_baseline") or row.get("delta_ler"),
                    "ler": row.get("ler_predecoder"),
                    "flops": row.get("flops_per_syndrome"),
                    "params": row.get("n_params"),
                    "status": status.upper() or "OK",
                    "on_pareto": node_branch in pareto_branches,
                    "hypothesis_1line": (row.get("hypothesis") or "")[:80],
                    "failure_reason": row.get("status_reason"),
                }
            )

    return {
        "run_id": run_id,
        "nodes": nodes,
        "pareto_front": [p.get("branch") for p in pareto if p.get("branch")],
    }
