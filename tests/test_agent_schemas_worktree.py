import pytest

from autoqec.agents.schemas import CoderResponse, IdeatorResponse


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


def test_ideator_compose_fork_from_without_mode_rejected():
    """fork_from as a list requires compose_mode — mirrors RunnerConfig invariant.

    The contract says a compose round must declare `pure` vs `with_edit`;
    IdeatorResponse was accepting the list without it, which let the
    underspecified payload propagate to downstream validators.
    """
    with pytest.raises(ValueError, match="compose_mode"):
        IdeatorResponse(
            hypothesis="compose 02+04",
            expected_delta_ler=5e-4,
            expected_cost_s=900,
            rationale="test compositionality",
            fork_from=["exp/.../02-a", "exp/.../04-b"],
            compose_mode=None,
        )


def test_ideator_scalar_fork_from_does_not_require_mode():
    """Scalar fork_from (single parent) never needs compose_mode."""
    r = IdeatorResponse(
        hypothesis="deepen on existing branch",
        expected_delta_ler=2e-4,
        expected_cost_s=600,
        rationale="try +1 layer",
        fork_from="exp/.../02-a",
    )
    assert r.compose_mode is None
