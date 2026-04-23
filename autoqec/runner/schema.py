from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, model_validator


class RunnerConfig(BaseModel):
    # === existing fields ===
    env_name: str
    predecoder_config: dict
    training_profile: Literal["dev", "prod"] = "dev"
    seed: int = 0
    round_dir: str

    # === §15 additions ===
    code_cwd: Optional[str] = None
    branch: Optional[str] = None
    fork_from: Optional[Union[str, list[str]]] = None
    fork_from_canonical: Optional[str] = None
    fork_from_ordered: Optional[list[str]] = None
    compose_mode: Optional[Literal["pure", "with_edit"]] = None
    round_attempt_id: Optional[str] = None
    commit_message: Optional[str] = None
    """Coder's proposed commit message when the round runs on a branch; falls back to 'round-attempt <uuid>'."""
    env_yaml_path: Optional[str] = None
    """Path to the environment YAML used for this round; populated by the CLI so the Runner can hash it into `artifact_manifest.json` (P0.3)."""

    @model_validator(mode="after")
    def _worktree_fields_consistent(self):
        if self.code_cwd is not None and self.branch is None:
            raise ValueError("branch is required when code_cwd is set")
        if isinstance(self.fork_from, list) and self.compose_mode is None:
            raise ValueError(
                "compose_mode is required for compose rounds (fork_from is a list)"
            )
        return self


class RoundMetrics(BaseModel):
    # === existing fields ===
    status: Literal[
        "ok",
        "killed_by_safety",
        "compile_error",
        "train_error",
        "failed",  # P0.7 — post-training git commit step failed
        "compose_conflict",  # §15.6 — no commit, no branch
        "orphaned_branch",  # §15.10 reconciliation — commit may exist, no live loop ran
        "branch_manually_deleted",  # §15.10 — follow-up marker
    ]
    status_reason: Optional[str] = None
    ler_plain_classical: Optional[float] = None
    ler_predecoder: Optional[float] = None
    delta_ler: Optional[float] = None
    delta_ler_ci_low: Optional[float] = None
    delta_ler_ci_high: Optional[float] = None
    flops_per_syndrome: Optional[int] = None
    n_params: Optional[int] = None
    train_wallclock_s: float = 0.0
    eval_wallclock_s: float = 0.0
    vram_peak_gb: float = 0.0
    checkpoint_path: Optional[str] = None
    training_log_path: Optional[str] = None

    # === §15 additions ===
    round_attempt_id: Optional[str] = None
    reconcile_id: Optional[str] = None
    branch: Optional[str] = None
    commit_sha: Optional[str] = None
    fork_from: Optional[Union[str, list[str]]] = None
    fork_from_canonical: Optional[str] = None
    fork_from_ordered: Optional[list[str]] = None
    compose_mode: Optional[Literal["pure", "with_edit"]] = None
    delta_vs_parent: Optional[float] = None
    parent_ler: Optional[float] = None
    conflicting_files: Optional[list[str]] = None
    train_seed: Optional[int] = None

    @model_validator(mode="after")
    def _provenance_integrity(self):
        is_worktree_row = (
            self.branch is not None
            or self.fork_from is not None
            or self.status
            in ("compose_conflict", "orphaned_branch", "branch_manually_deleted")
        )
        if (
            is_worktree_row
            and self.round_attempt_id is None
            and self.reconcile_id is None
        ):
            raise ValueError(
                "worktree-path rows need round_attempt_id (normal) or reconcile_id (startup-reconstructed)"
            )
        if self.round_attempt_id is not None and self.reconcile_id is not None:
            raise ValueError(
                "round_attempt_id and reconcile_id are mutually exclusive"
            )
        if self.status == "compose_conflict" and (
            self.branch is not None or self.commit_sha is not None
        ):
            raise ValueError(
                "compose_conflict rows must have branch=None and commit_sha=None"
            )
        # `branch_manually_deleted` follow-ups carry the branch name for
        # downstream joins, but the branch is gone — so commit_sha is
        # legitimately unavailable. Every other row with a branch must
        # carry a commit_sha as provenance.
        if (
            self.branch is not None
            and self.commit_sha is None
            and self.status != "branch_manually_deleted"
        ):
            raise ValueError("commit_sha is required whenever branch is set")
        return self
