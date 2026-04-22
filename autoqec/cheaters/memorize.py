"""A hand-crafted cheating predecoder for calibrating independent_eval.

Memorizes the (syndrome → correction) map on train seeds and returns
zeros on anything unseen. On holdout it should NOT beat baseline."""
from pathlib import Path

import torch


class MemorizerPredecoder(torch.nn.Module):
    output_mode = "hard_flip"

    def __init__(self):
        super().__init__()
        self.table: dict[tuple, torch.Tensor] = {}

    def forward(self, syndrome: torch.Tensor, ctx=None):
        out = torch.zeros_like(syndrome)
        for i, s in enumerate(syndrome):
            key = tuple(s.tolist())
            if key in self.table:
                out[i] = self.table[key]
        return out


def save_memorizer_ckpt(path: Path) -> None:
    m = MemorizerPredecoder()
    torch.save({
        "class_name": "MemorizerPredecoder",
        "model": m,
        "state_dict": m.state_dict(),
    }, path)
