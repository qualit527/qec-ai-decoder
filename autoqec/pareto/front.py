from typing import Any


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


def update_front(front: list[dict], cand: dict) -> list[dict]:
    if cand.get("verdict") != "VERIFIED":
        return front
    # Remove existing entries dominated by cand
    pruned = [p for p in front if not is_pareto_dominated(p, [cand])]
    if is_pareto_dominated(cand, pruned):
        return pruned
    return pruned + [cand]
