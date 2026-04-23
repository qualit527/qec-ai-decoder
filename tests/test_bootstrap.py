import numpy as np


def test_bootstrap_ci_ler():
    from autoqec.eval.bootstrap import bootstrap_ci_mean
    rng = np.random.default_rng(0)
    # 200K shots with true error rate 1e-3
    outcomes = rng.binomial(1, 1e-3, size=200_000).astype(np.int32)
    mean, lo, hi = bootstrap_ci_mean(outcomes, n_resamples=500, ci=0.95, seed=1)
    assert 5e-4 < mean < 2e-3
    assert lo < mean < hi
    assert hi - lo < 5e-4   # tight CI at 200K shots
