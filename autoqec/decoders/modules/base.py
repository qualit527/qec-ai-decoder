"""Shared predecoder interface."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

OutputMode = Literal["hard_flip", "soft_priors"]


class PredecoderBase(nn.Module):
    output_mode: OutputMode = "soft_priors"

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def expected_output_shape(self) -> str:
        if self.output_mode == "hard_flip":
            # Soft probabilities per detector; backend thresholds at 0.5.
            return "[batch, n_checks] float in [0, 1]"
        return "[batch, n_faults] float in [0, 1]"

