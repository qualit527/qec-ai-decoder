"""End-of-round bookkeeping helper (Task B2 + §15.2 non-dominated Pareto).

Chains: `RoundMetrics` superset row + Analyst/Verify verdict →
  (1) row appended to `history.jsonl`
  (2) `pareto.json` updated via non-dominated dominance filter when the
      round is VERIFIED (no size cap — the full non-dominated archive)
  (3) `pareto_preview.json` refreshed with the top-5-by-delta slice
  (4) one-liner appended to `log.md`.

Lives in its own module so `/autoqec-run` can call a single function
after each round instead of threading files through the orchestrator by
hand. Matches the contract in `docs/contracts/round_dir_layout.md`.

### §15.2 dominance semantics

A candidate `a` dominates `b` iff `a` is at least as good as `b` on
every axis (+delta_ler, -flops_per_syndrome, -n_params) AND strictly
better on at least one. The archive retains every non-dominated row —
it is never truncated. `pareto_preview.json` is a separate top-5-by-
delta slice intended for compact prompt contexts.
"""
from __future__ import annotations

import json
from pathlib import Path

from autoqec.orchestration.memory import RunMemory

_PARETO_PREVIEW_CAP = 5

_PARETO_FIELDS = (
    "round",
    "round_attempt_id",
    "branch",
    "commit_sha",
    "fork_from",
    "fork_from_canonical",
    "delta_ler",
    "flops_per_syndrome",
    "n_params",
    "checkpoint_path",
    "hypothesis",
)


def _dominates(a: dict, b: dict) -> bool:
    """Return True iff candidate `a` dominates `b` on (+delta_ler, -flops, -n_params).

    `a` dominates `b` iff `a` is at least as good as `b` on every axis AND
    strictly better on at least one. Missing numeric fields coerce to 0.
    """
    a_d, b_d = float(a.get("delta_ler") or 0), float(b.get("delta_ler") or 0)
    a_f, b_f = int(a.get("flops_per_syndrome") or 0), int(b.get("flops_per_syndrome") or 0)
    a_p, b_p = int(a.get("n_params") or 0), int(b.get("n_params") or 0)
    at_least_as_good = (a_d >= b_d) and (a_f <= b_f) and (a_p <= b_p)
    strictly_better = (a_d > b_d) or (a_f < b_f) or (a_p < b_p)
    return at_least_as_good and strictly_better


def _non_dominated_merge(front: list[dict], candidate: dict) -> list[dict]:
    """Admit `candidate` to `front` using Pareto dominance.

    - Reject `candidate` if any existing member dominates it.
    - Drop any existing member dominated by `candidate`.
    - Otherwise append `candidate`.
    """
    for existing in front:
        if _dominates(existing, candidate):
            return front  # candidate is dominated; reject
    pruned = [p for p in front if not _dominates(candidate, p)]
    pruned.append(candidate)
    return pruned


def _pareto_row(metrics: dict) -> dict:
    return {k: metrics.get(k) for k in _PARETO_FIELDS}


def _write_preview(run_dir: Path, front: list[dict]) -> None:
    preview = sorted(front, key=lambda r: -float(r.get("delta_ler") or 0))[:_PARETO_PREVIEW_CAP]
    (run_dir / "pareto_preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def record_round(
    mem: RunMemory,
    round_metrics: dict,
    verify_verdict: str | None = None,
) -> None:
    """End-of-round bookkeeping: history row + Pareto update + log line.

    `round_metrics` is the flattened superset dict (RoundMetrics + §15.7
    fields). `verify_verdict` is the Analyst/Verify outcome — only
    "VERIFIED" admits the row into `pareto.json`.
    """
    mem.append_round(round_metrics)
    mem.append_log(
        f"- round {round_metrics.get('round')}: status={round_metrics.get('status')}"
    )

    if verify_verdict != "VERIFIED":
        return

    front = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
    candidate = _pareto_row(round_metrics)
    front = _non_dominated_merge(front, candidate)
    mem.update_pareto(front)
    _write_preview(mem.run_dir, front)
