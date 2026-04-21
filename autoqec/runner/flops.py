"""FLOPs estimation with a graceful fallback."""

from __future__ import annotations


def estimate_flops(model, example_inputs) -> int:
    try:
        from fvcore.nn import FlopCountAnalysis

        return int(FlopCountAnalysis(model, example_inputs).total())
    except Exception:
        n_params = sum(parameter.numel() for parameter in model.parameters())
        return int(2 * n_params)

