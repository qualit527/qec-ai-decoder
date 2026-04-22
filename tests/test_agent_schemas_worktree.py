import pytest
from autoqec.agents.schemas import IdeatorResponse, CoderResponse


def test_ideator_legacy_response_gets_baseline_default():
    # Existing responses on main don't emit fork_from; default should be "baseline".
    r = IdeatorResponse(
        hypothesis="try gated MLP",
        expected_delta_ler=1e-4,
        expected_cost_s=600,
        rationale="prior rounds show plateau",
    )
    assert r.fork_from == "baseline"
    assert r.compose_mode is None


def test_ideator_accepts_compose_fork_from():
    r = IdeatorResponse(
        hypothesis="compose 02+04",
        expected_delta_ler=5e-4,
        expected_cost_s=900,
        rationale="test compositionality",
        fork_from=["exp/.../02-a", "exp/.../04-b"],
        compose_mode="pure",
    )
    assert isinstance(r.fork_from, list)
    assert r.compose_mode == "pure"


def test_coder_commit_message_optional():
    # Legacy in-process Coder responses should still validate.
    r = CoderResponse(dsl_config={"type": "gnn"}, tier="1", rationale="baseline GNN")
    assert r.commit_message is None
