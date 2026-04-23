from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class VerifyReport(BaseModel):
    """Hold-out evaluation + ablation report (docs/contracts/interfaces.md §2.6)."""

    verdict: Literal["VERIFIED", "SUSPICIOUS", "FAILED"]
    ler_holdout: float
    ler_holdout_ci: tuple[float, float]
    delta_ler_holdout: float
    ler_shuffled: float
    ablation_sanity_ok: bool
    holdout_seeds_used: list[int]
    seed_leakage_check_ok: bool
    notes: str

    # §15 worktree fields
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    delta_vs_baseline_holdout: Optional[float] = None
    paired_eval_bundle_id: Optional[str] = None

    @model_validator(mode="after")
    def _branch_needs_commit(self):
        if self.branch is not None and self.commit_sha is None:
            raise ValueError("commit_sha is required whenever branch is set")
        return self
