"""Pydantic schemas for subagent responses (docs/contracts/interfaces.md §2.5).

Used by `dispatch.parse_response` to enforce shape after the fenced JSON
block is extracted. Without this layer, a malformed-but-JSON subagent
output would slip through silently.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class IdeatorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    expected_delta_ler: float
    expected_cost_s: int
    rationale: str
    dsl_hint: Optional[dict] = None


class CoderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: Literal["1", "2"]
    dsl_config: dict
    rationale: str


class AnalystResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary_1line: str
    verdict: Literal["candidate", "ignore"]
    next_hypothesis_seed: str


ROLE_SCHEMAS: dict[str, type[BaseModel]] = {
    "ideator": IdeatorResponse,
    "coder": CoderResponse,
    "analyst": AnalystResponse,
}
