from __future__ import annotations


def is_pareto_dominated(cand: dict, front: list[dict]) -> bool:
    for other in front:
        if (other["delta_ler"] >= cand["delta_ler"]
            and other["flops"] <= cand["flops"]
            and other["n_params"] <= cand["n_params"]
            and (other["delta_ler"] > cand["delta_ler"]
                 or other["flops"] < cand["flops"]
                 or other["n_params"] < cand["n_params"])):
            return True
    return False


def _dedup(front: list[dict]) -> list[dict]:
    """Remove entries with identical (delta_ler, flops, n_params)."""
    seen: set[tuple] = set()
    out: list[dict] = []
    for p in front:
        key = (p["delta_ler"], p["flops"], p["n_params"])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def update_front(front: list[dict], cand: dict) -> list[dict]:
    if cand.get("verdict") != "VERIFIED":
        return front
    # Remove existing entries dominated by cand
    pruned = [p for p in front if not is_pareto_dominated(p, [cand])]
    if is_pareto_dominated(cand, pruned):
        return pruned
    result = pruned + [cand]
    result = _dedup(result)
    # Sort by -delta_ler (best improvement first)
    result.sort(key=lambda x: -x["delta_ler"])
    return result
