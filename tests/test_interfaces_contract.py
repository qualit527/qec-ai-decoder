"""Phase 2.1.3 — cross-check pydantic models against `docs/contracts/interfaces.md`.

Each pydantic model documented in the contract file has its authoritative
field list frozen here. Drift (a new field added in code without updating
the contract, or the contract declaring a field the code doesn't have)
fails the test with a readable diff, flagging the missing `contract-change`
sign-off per CLAUDE.md.

This is NOT a markdown parser — it is a brittle-on-purpose inventory. When
the contract changes, update both files in the same commit.
"""

from __future__ import annotations

from tests.fixture_utils import load_json_fixture

from pydantic import BaseModel

import pytest


_expected_fields_payload = load_json_fixture("public_api", "interface_model_fields.json")
EXPECTED_FIELDS: dict[str, set[str]] = {
    model_name: set(fields)
    for model_name, fields in _expected_fields_payload.items()
}


def _live_models() -> dict[str, type[BaseModel]]:
    """Import the live pydantic models from their authoritative modules."""
    from autoqec.agents.schemas import (
        AnalystResponse,
        CoderResponse,
        IdeatorResponse,
    )
    from autoqec.envs.schema import (
        CodeSpec,
        ConstraintsSpec,
        EnvSpec,
        EvalProtocol,
        NoiseSpec,
        SeedPolicy,
    )
    from autoqec.eval.schema import VerifyReport
    from autoqec.runner.schema import RoundMetrics, RunnerConfig

    return {
        "SeedPolicy": SeedPolicy,
        "NoiseSpec": NoiseSpec,
        "CodeSpec": CodeSpec,
        "ConstraintsSpec": ConstraintsSpec,
        "EvalProtocol": EvalProtocol,
        "EnvSpec": EnvSpec,
        "RunnerConfig": RunnerConfig,
        "RoundMetrics": RoundMetrics,
        "VerifyReport": VerifyReport,
        "IdeatorResponse": IdeatorResponse,
        "CoderResponse": CoderResponse,
        "AnalystResponse": AnalystResponse,
    }


@pytest.mark.parametrize("model_name", sorted(EXPECTED_FIELDS))
def test_interface_model_fields_match_contract(model_name: str) -> None:
    live_models = _live_models()
    model = live_models[model_name]
    actual = set(model.model_fields.keys())
    expected = EXPECTED_FIELDS[model_name]
    missing_in_code = expected - actual
    extra_in_code = actual - expected
    assert not missing_in_code and not extra_in_code, (
        f"{model_name} drift vs docs/contracts/interfaces.md:\n"
        f"  missing in code (documented but not defined): {sorted(missing_in_code)}\n"
        f"  extra in code   (defined but not documented):  {sorted(extra_in_code)}\n"
        "Contract changes require 3-of-3 owner sign-off (CLAUDE.md §Commit)."
    )


def test_all_contracted_models_importable() -> None:
    """Every model named in interfaces.md must actually exist."""
    live = _live_models()
    assert set(live) == set(EXPECTED_FIELDS), (
        f"coverage drift: live={sorted(live)} vs expected={sorted(EXPECTED_FIELDS)}"
    )
