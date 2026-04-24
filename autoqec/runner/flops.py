"""FLOPs estimation with a graceful fallback."""

from __future__ import annotations

import warnings


def estimate_flops(model, example_inputs) -> int:
    """Return per-forward FLOPs, falling back to ``2*n_params`` when
    fvcore is unavailable or its analysis is incomplete.

    fvcore silently returns a partial count when it hits operations
    it doesn't know how to score (`aten::index_add_`, `aten::sigmoid`,
    scatter / gather, etc.). Pareto uses ``flops_per_syndrome`` as a
    hard axis, so letting the partial count through makes architectures
    with unsupported ops look artificially cheap and preferentially
    survive the prune. We detect this via ``FlopCountAnalysis.unsupported_ops()``
    and fall back to the same ``2*n_params`` proxy every other partial
    architecture gets — fair, even if not exact.
    """
    n_params = sum(parameter.numel() for parameter in model.parameters())
    try:
        from fvcore.nn import FlopCountAnalysis

        analysis = FlopCountAnalysis(model, example_inputs)
        unsupported = {}
        try:
            unsupported = dict(analysis.unsupported_ops())
        except Exception:
            # Older fvcore without unsupported_ops() — keep going.
            pass
        if unsupported:
            warnings.warn(
                f"fvcore undercount detected — unsupported ops: {sorted(unsupported)}. "
                f"Falling back to 2*n_params proxy so Pareto stays fair.",
                stacklevel=2,
            )
            return int(2 * n_params)
        return int(analysis.total())
    except Exception:
        return int(2 * n_params)

