"""Smoke test for scripts.benchmark_surface_baseline (Task A1.4).

The 1M-shot production run is manual (see docs/superpowers/plans/
2026-04-21-autoqec-person-a-chen.md). This test exercises the same
function with a tiny shot count so the logic is covered in CI.
"""
from __future__ import annotations


def test_benchmark_small_shot_count_returns_expected_schema() -> None:
    from scripts.benchmark_surface_baseline import benchmark

    result = benchmark(n_shots=1000, seed=1)
    # Schema keys
    for key in (
        "n_shots",
        "ler",
        "n_errors",
        "t_sample_s",
        "t_decode_s",
        "detections_shape",
    ):
        assert key in result, f"missing key {key} in {result}"
    # Sanity
    assert result["n_shots"] == 1000
    assert 0.0 <= result["ler"] <= 0.5
    assert result["n_errors"] == int(result["n_errors"])
    assert result["t_sample_s"] >= 0.0 and result["t_decode_s"] >= 0.0
    assert len(result["detections_shape"]) == 2
    assert result["detections_shape"][0] == 1000
