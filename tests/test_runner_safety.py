import torch

from autoqec.runner.safety import estimate_vram_gb, nan_rate


def test_nan_rate_detection() -> None:
    assert nan_rate([1.0, 2.0, float("nan"), 3.0]) == 0.25
    assert nan_rate([]) == 0.0


def test_vram_estimate_positive() -> None:
    model = torch.nn.Linear(64, 64)
    estimate = estimate_vram_gb(model, batch_size=32, hidden=64)
    assert estimate > 0

