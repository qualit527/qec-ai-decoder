import numpy as np


def bootstrap_ci_mean(outcomes: np.ndarray, n_resamples: int = 1000,
                      ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    means = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = outcomes[idx].mean()
    alpha = (1 - ci) / 2
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return float(outcomes.mean()), lo, hi
