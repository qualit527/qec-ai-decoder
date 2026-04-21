from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class RunnerSafety:
    WALL_CLOCK_HARD_CUTOFF_S: int = 900
    VRAM_PRE_CHECK: bool = True
    MAX_NAN_RATE: float = 0.01
    FORBIDDEN_IMPORTS: tuple[str, ...] = ("os.system", "subprocess", "sys.exit")


def estimate_vram_gb(model: torch.nn.Module, batch_size: int, hidden: int) -> float:
    n_params = sum(parameter.numel() for parameter in model.parameters())
    return (4 * n_params * 3 + batch_size * hidden * 8) / 1e9


def nan_rate(loss_history: list[float]) -> float:
    if not loss_history:
        return 0.0
    bad = sum(1 for loss in loss_history if not math.isfinite(loss))
    return bad / len(loss_history)

