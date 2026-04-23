"""End-of-round bookkeeping helper (Task B2 + §15.2 non-dominated Pareto).

Chains: `RoundMetrics` superset row + Analyst/Verify verdict + `VerifyReport`
    →
  (1) row appended to `history.jsonl`
  (2) `pareto.json` updated via non-dominated dominance filter when the
      round is VERIFIED and a VerifyReport carrying `delta_vs_baseline_holdout`
      is available (no size cap — the full non-dominated archive)
  (3) `pareto_preview.json` refreshed with the top-5-by-holdout-delta slice
  (4) one-liner appended to `log.md`.

Lives in its own module so `/autoqec-run` can call a single function
after each round instead of threading files through the orchestrator by
hand. Matches the contract in `docs/contracts/round_dir_layout.md`.

### §15.2 / §15.7 Pareto semantics

Pareto rows are keyed on holdout-side fields from a paired
`VerifyReport`, **not** the training-side `delta_ler` from `RoundMetrics`.
The dominance axis set is:

- `+delta_vs_baseline_holdout` (higher = better)  — from VerifyReport
- `-flops_per_syndrome` (lower = better)          — from RoundMetrics
- `-n_params` (lower = better)                    — from RoundMetrics

A candidate `a` dominates `b` iff `a` is at least as good as `b` on
every axis AND strictly better on at least one. The archive retains
every non-dominated row — it is never truncated. `pareto_preview.json`
is a separate top-5-by-holdout-delta slice intended for compact prompt
contexts. A VERIFIED round **without** a `verify_report` (or without
`delta_vs_baseline_holdout`) cannot be admitted: there is no holdout
axis to compare.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

from autoqec.orchestration.memory import RunMemory

log = logging.getLogger(__name__)

_PARETO_PREVIEW_CAP = 5

# §15.7 superset: RoundMetrics cost/provenance fields + VerifyReport quality fields.
# Any field missing from the source maps defaults to None in the Pareto row.
_PARETO_FIELDS = (
    # identity + round
    "round",
    "round_attempt_id",
    # provenance (from RoundMetrics)
    "branch",
    "commit_sha",
    "fork_from",
    "fork_from_canonical",
    "fork_from_ordered",
    "compose_mode",
    # hypothesis narrative
    "hypothesis",
    # quality (from VerifyReport — these are the Pareto axis keys)
    "delta_vs_baseline_holdout",
    "paired_eval_bundle_id",
    "verdict",
    "ler_holdout",
    # cost (from RoundMetrics)
    "flops_per_syndrome",
    "n_params",
    "train_wallclock_s",
    # artefact
    "checkpoint_path",
)

# RoundMetrics-side fields → pulled from round_metrics.
_ROUND_METRICS_FIELDS = frozenset({
    "round",
    "round_attempt_id",
    "branch",
    "commit_sha",
    "fork_from",
    "fork_from_canonical",
    "fork_from_ordered",
    "compose_mode",
    "hypothesis",
    "flops_per_syndrome",
    "n_params",
    "train_wallclock_s",
    "checkpoint_path",
})

# VerifyReport-side fields → pulled from verify_report (take precedence).
_VERIFY_REPORT_FIELDS = frozenset({
    "delta_vs_baseline_holdout",
    "paired_eval_bundle_id",
    "verdict",
    "ler_holdout",
})


_PARETO_AXES = ("delta_vs_baseline_holdout", "flops_per_syndrome", "n_params")


def _has_all_pareto_axes(row: Mapping[str, Any]) -> bool:
    """Pareto admission requires all three axes to be present and non-None.

    Earlier we coerced missing cost fields to 0 inside ``_dominates``, which
    let a malformed VERIFIED row "dominate" every real candidate on the
    cost axes and quietly evict genuine winners from ``pareto.json``.
    """
    return all(row.get(k) is not None for k in _PARETO_AXES)


def _dominates(a: dict, b: dict) -> bool:
    """Return True iff candidate `a` dominates `b` on holdout-delta / flops / params.

    Axes: `+delta_vs_baseline_holdout` (maximize), `-flops_per_syndrome` (minimize),
    `-n_params` (minimize). Assumes ``_has_all_pareto_axes`` has already passed
    on both rows — callers guarantee this at admission time.
    """
    a_d = float(a["delta_vs_baseline_holdout"])
    b_d = float(b["delta_vs_baseline_holdout"])
    a_f = int(a["flops_per_syndrome"])
    b_f = int(b["flops_per_syndrome"])
    a_p = int(a["n_params"])
    b_p = int(b["n_params"])
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


def _pareto_row(
    round_metrics: Mapping[str, Any],
    verify_report: Mapping[str, Any],
) -> dict:
    """Build a Pareto row by projecting _PARETO_FIELDS from both sources.

    VerifyReport fields win when both dicts carry a value (the VerifyReport
    is the canonical source for holdout quality numbers).
    """
    row: dict[str, Any] = {}
    for field in _PARETO_FIELDS:
        if field in _VERIFY_REPORT_FIELDS:
            row[field] = verify_report.get(field, round_metrics.get(field))
        elif field in _ROUND_METRICS_FIELDS:
            row[field] = round_metrics.get(field)
        else:
            # defensive — currently every _PARETO_FIELDS entry falls in one set
            row[field] = round_metrics.get(field, verify_report.get(field))
    return row


def _write_preview(run_dir: Path, front: list[dict]) -> None:
    preview = sorted(
        front,
        key=lambda r: -float(r.get("delta_vs_baseline_holdout") or 0),
    )[:_PARETO_PREVIEW_CAP]
    (run_dir / "pareto_preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def record_round(
    mem: RunMemory,
    round_metrics: dict,
    verify_verdict: Optional[str] = None,
    verify_report: Optional[Mapping[str, Any]] = None,
) -> None:
    """End-of-round bookkeeping: history row + Pareto update + log line.

    Parameters
    ----------
    mem:
        `RunMemory` handle pointing at the run directory.
    round_metrics:
        Flattened superset dict (RoundMetrics + §15.7 fields).
    verify_verdict:
        Analyst/Verify outcome — only ``"VERIFIED"`` admits the row into
        ``pareto.json``.
    verify_report:
        Paired-bundle VerifyReport as a dict (kept loose for contract
        compatibility). Required for Pareto admission: must carry
        ``delta_vs_baseline_holdout``. When missing, Pareto admission is
        skipped even if ``verify_verdict == "VERIFIED"``.
    """
    mem.append_round(round_metrics)
    mem.append_log(
        f"- round {round_metrics.get('round')}: status={round_metrics.get('status')}"
    )

    if verify_verdict != "VERIFIED":
        return

    if verify_report is None:
        log.warning(
            "record_round: VERIFIED round %s has no verify_report — skipping Pareto admission",
            round_metrics.get("round"),
        )
        return
    if verify_report.get("delta_vs_baseline_holdout") is None:
        log.warning(
            "record_round: VERIFIED round %s has no delta_vs_baseline_holdout — skipping Pareto admission",
            round_metrics.get("round"),
        )
        return

    front = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
    candidate = _pareto_row(round_metrics, verify_report)
    if not _has_all_pareto_axes(candidate):
        missing = [k for k in _PARETO_AXES if candidate.get(k) is None]
        log.warning(
            "record_round: VERIFIED round %s missing Pareto axes %s — skipping admission",
            round_metrics.get("round"),
            missing,
        )
        return
    front = _non_dominated_merge(front, candidate)
    mem.update_pareto(front)
    _write_preview(mem.run_dir, front)
