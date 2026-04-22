"""End-of-round bookkeeping helper (Task B2).

Chains: `metrics.json` + Analyst verdict →
  (1) superset row appended to `history.jsonl`
  (2) `pareto.json` refreshed if the round is a candidate
  (3) one-liner appended to `log.md`.

Lives in its own module so `/autoqec-run` can call a single function
after each round instead of threading files through the orchestrator by
hand. Matches the contract in `docs/contracts/round_dir_layout.md`.
"""
from __future__ import annotations

import json

from autoqec.orchestration.memory import RunMemory

_PARETO_CAP = 5

_PARETO_FIELDS = (
    "round",
    "delta_ler",
    "flops_per_syndrome",
    "n_params",
    "checkpoint_path",
    "hypothesis",
)


def _pareto_key(entry: dict) -> tuple:
    """Sort key: maximise delta_ler, then minimise flops, then minimise n_params."""
    return (
        -float(entry.get("delta_ler") or 0),
        int(entry.get("flops_per_syndrome") or 0),
        int(entry.get("n_params") or 0),
    )


def _refresh_pareto(mem: RunMemory, row: dict) -> None:
    pareto = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
    entry = {k: row.get(k) for k in _PARETO_FIELDS}
    # dedupe by (round, delta_ler) — the round number is a natural id
    pareto = [e for e in pareto if e.get("round") != entry["round"]]
    pareto.append(entry)
    pareto.sort(key=_pareto_key)
    mem.update_pareto(pareto[:_PARETO_CAP])


def record_round(
    mem: RunMemory,
    round_idx: int,
    hypothesis: str,
    dsl_config: dict,
    metrics: dict,
    verdict: str,
    summary_1line: str,
) -> dict:
    """Persist one completed round and return the row written to `history.jsonl`."""
    row: dict = {
        "round": round_idx,
        "hypothesis": hypothesis,
        "dsl_config": dsl_config,
        "verdict": verdict,
        "summary_1line": summary_1line,
        **metrics,
    }
    mem.append_round(row)
    mem.append_log(f"### round {round_idx} — {summary_1line}")

    # pareto only tracks verified-or-candidate rounds with a positive delta
    if verdict == "candidate" and (metrics.get("delta_ler") or 0) > 0:
        _refresh_pareto(mem, row)

    return row
