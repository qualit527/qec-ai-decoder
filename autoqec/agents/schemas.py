"""Pydantic schemas for subagent responses (docs/contracts/interfaces.md §2.5).

Used by `dispatch.parse_response` to enforce shape after the fenced JSON
block is extracted. Without this layer, a malformed-but-JSON subagent
output would slip through silently.
"""
from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator


class IdeatorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    expected_delta_ler: float
    expected_cost_s: int
    rationale: str
    dsl_hint: Optional[dict] = None

    # §15 additions — fork_from defaults to "baseline" so legacy responses validate.
    fork_from: Union[str, list[str]] = "baseline"
    compose_mode: Optional[Literal["pure", "with_edit"]] = None

    @model_validator(mode="after")
    def _compose_requires_mode(self) -> "IdeatorResponse":
        # Mirrors the RunnerConfig invariant: list fork_from means compose round,
        # which requires an explicit compose_mode (pure vs with_edit).
        if isinstance(self.fork_from, list) and self.compose_mode is None:
            raise ValueError(
                "compose_mode is required when fork_from is a list"
            )
        return self


class CoderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Literal["1", "2"]
    dsl_config: dict
    rationale: str

    # §15 addition — Coder sets this on the worktree path; Optional so legacy works.
    commit_message: Optional[str] = None


class AnalystResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_1line: str
    verdict: Literal["candidate", "ignore"]
    next_hypothesis_seed: str
    branch: Optional[str] = None
    commit_sha: Optional[str] = None


ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "ideator": IdeatorResponse,
    "coder": CoderResponse,
    "analyst": AnalystResponse,
}
