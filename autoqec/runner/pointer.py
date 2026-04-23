"""Write round_N_pointer.json — authoritative provenance per §15.5.

Read by autoqec.orchestration.reconcile to recover round_attempt_id
after a crash. Without this file reconcile emits a `pause` action.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoqec.runner.schema import RoundMetrics, RunnerConfig


def write_round_pointer(
    cfg: RunnerConfig,
    metrics: RoundMetrics,
    round_idx: int,
) -> Path:
    round_attempt_id = cfg.round_attempt_id or metrics.round_attempt_id
    reconcile_id = getattr(metrics, "reconcile_id", None)
    if not round_attempt_id and not reconcile_id:
        raise ValueError(
            "pointer must carry either round_attempt_id or reconcile_id "
            "(spec §15.2 mutual-exclusion)"
        )
    if round_attempt_id and reconcile_id:
        raise ValueError(
            "round_attempt_id and reconcile_id are mutually exclusive "
            "(spec §15.2 mutual-exclusion)"
        )
    pointer: dict[str, Any] = {
        "round_attempt_id": round_attempt_id or None,
        "reconcile_id": reconcile_id,
        "branch": metrics.branch,
        "commit_sha": metrics.commit_sha,
        "worktree_path": cfg.code_cwd,
        "fork_from": cfg.fork_from,
        "fork_from_ordered": getattr(cfg, "fork_from_ordered", None),
        "compose_mode": cfg.compose_mode,
        "status": metrics.status,
        "status_reason": getattr(metrics, "status_reason", None),
    }
    out = Path(cfg.round_dir) / f"round_{round_idx}_pointer.json"
    out.write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
