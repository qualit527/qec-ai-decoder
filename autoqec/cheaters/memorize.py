"""A hand-crafted cheating predecoder for calibrating independent_eval.

Memorizes the (syndrome → perfect_correction) map from training data.
On unseen holdout syndromes it falls back to zeros — so it should NOT
beat the plain baseline on holdout, and independent_eval must flag it.
"""
from pathlib import Path

import numpy as np
import torch


class MemorizerPredecoder(torch.nn.Module):
    output_mode = "hard_flip"

    def __init__(self):
        super().__init__()
        self.table: dict[tuple, list[float]] = {}
        # Dummy parameter so ablation shuffle has something to permute
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, syndrome: torch.Tensor, ctx=None):
        out = torch.zeros_like(syndrome)
        for i in range(syndrome.shape[0]):
            key = tuple(syndrome[i].tolist())
            if key in self.table:
                out[i] = torch.tensor(self.table[key], dtype=syndrome.dtype)
        return out

    def memorize(self, syndromes: np.ndarray, corrections: np.ndarray) -> None:
        """Fill the lookup table from training data.

        For each unique syndrome, store the first matching correction.
        Collisions are overwritten (last-write-wins), which is fine for
        a cheating predecoder — it only needs to look good on train.
        """
        for i in range(syndromes.shape[0]):
            key = tuple(syndromes[i].tolist())
            self.table[key] = corrections[i].tolist()


def train_memorizer(
    env_spec,
    artifacts,
    n_shots: int = 10_000,
) -> "MemorizerPredecoder":
    """Build a memorizer by reading train-seed syndromes and their
    perfect corrections (error patterns).

    Uses the data pipeline but with a large sample so the table is dense.
    """
    from autoqec.runner.data import sample_syndromes

    syndrome, target = sample_syndromes(
        env_spec, artifacts, env_spec.noise.seed_policy.train, n_shots,
    )
    m = MemorizerPredecoder()
    m.memorize(syndrome.numpy(), target.numpy())
    return m


def save_memorizer_ckpt(path: Path, env_spec=None, artifacts=None) -> None:
    """Save a Memorizer checkpoint.

    If env_spec + artifacts are provided, fills the table from training data.
    Otherwise saves an empty table (identity fallback on everything).
    """
    if env_spec is not None and artifacts is not None:
        m = train_memorizer(env_spec, artifacts)
    else:
        m = MemorizerPredecoder()

    torch.save({
        "class_name": "MemorizerPredecoder",
        "model": m,
        "state_dict": m.state_dict(),
        "output_mode": "hard_flip",
        "dsl_config": None,
    }, path)
