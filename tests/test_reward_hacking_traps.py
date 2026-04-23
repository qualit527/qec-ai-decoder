"""Acceptance tests for reward-hacking trap fixtures.

These are integration tests and additionally skip when the `.pt` fixtures
are absent (e.g., on CPU-only CI where ``make build-trap-fixtures`` has not
been run). Run with::

    pytest tests/test_reward_hacking_traps.py --run-integration
"""
from __future__ import annotations

from pathlib import Path

import pytest


FIX = Path(__file__).resolve().parent / "fixtures" / "reward_hacking"


@pytest.mark.skipif(
    not (FIX / "trap_A.pt").exists(),
    reason="reward-hacking fixture trap_A.pt not built (see make build-trap-fixtures)",
)
@pytest.mark.integration
def test_trap_A_fails_verification() -> None:
    from autoqec.envs.schema import load_env_yaml
    from autoqec.eval.independent_eval import independent_verify  # type: ignore[attr-defined]

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    # Construct a deliberately-leaky holdout: one seed inside train range.
    holdout = [env.noise.seed_policy.train[0]]  # leak
    with pytest.raises(ValueError, match="seed"):
        independent_verify(FIX / "trap_A.pt", env, holdout_seeds=holdout)


@pytest.mark.skipif(
    not (FIX / "trap_C.pt").exists(),
    reason="reward-hacking fixture trap_C.pt not built (see make build-trap-fixtures)",
)
@pytest.mark.integration
def test_trap_C_memorizer_fails_or_ci_crosses_zero() -> None:
    from autoqec.envs.schema import load_env_yaml
    from autoqec.eval.independent_eval import independent_verify  # type: ignore[attr-defined]

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    holdout_start = env.noise.seed_policy.holdout[0]
    holdout = list(range(holdout_start, holdout_start + 3))
    report = independent_verify(
        FIX / "trap_C.pt", env, holdout_seeds=holdout, n_shots=1000,
    )
    ci_lo, ci_hi = report.ler_holdout_ci
    assert report.verdict == "FAILED" or (ci_lo < 0 < ci_hi)
