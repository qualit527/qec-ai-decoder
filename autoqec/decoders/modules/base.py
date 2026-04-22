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
            return "[batch, n_checks] long"
        return "[batch, n_faults] float in [0, 1]"

