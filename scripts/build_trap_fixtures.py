"""Produce reward-hacking fixture checkpoints.

Run once with ``make build-trap-fixtures``. Fixtures check into
``tests/fixtures/reward_hacking/``.

Each of the three traps simulates one class of reward-hacking failure the
verifier must catch (spec Sec. 9 + test plan Phase 5.2):

- trap_A — training-seed-leak checkpoint.
- trap_B — paired-batch-mismatch checkpoint.
- trap_C — 100-shot overfit MemorizerPredecoder checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from autoqec.cheaters.memorize import MemorizerPredecoder
from autoqec.decoders.dsl_compiler import compile_predecoder
from autoqec.envs.schema import load_env_yaml
from autoqec.runner.data import load_code_artifacts


OUT_DIR = Path("tests/fixtures/reward_hacking")


def _gnn_dsl_config() -> dict:
    """Minimal, schema-valid DSL config used by traps A and B."""
    return {
        "type": "gnn",
        "output_mode": "soft_priors",
        "gnn": {
            "layers": 2,
            "hidden_dim": 8,
            "message_fn": "mlp",
            "aggregation": "sum",
            "normalization": "layer",
            "residual": True,
            "edge_features": ["syndrome_bit"],
        },
        "head": "linear",
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 4,
            "epochs": 1,
            "loss": "bce",
            "profile": "dev",
        },
    }


def build_trap_a(env_yaml: Path) -> None:
    """Training-seed-leak: claims train_seeds that overlap the holdout range."""
    env = load_env_yaml(env_yaml)
    art = load_code_artifacts(env)
    model = compile_predecoder(_gnn_dsl_config(), n_var=art.n_var, n_check=art.n_check)
    holdout_start = env.noise.seed_policy.holdout[0]
    ckpt = {
        "class_name": "GNNPredecoder",
        "state_dict": model.state_dict(),
        "output_mode": "soft_priors",
        "dsl_config": _gnn_dsl_config(),
        # Claimed train seeds deliberately overlap the holdout range.
        "train_seeds_claimed": list(range(holdout_start, holdout_start + 5)),
        "trap_kind": "training_seed_leak",
    }
    torch.save(ckpt, OUT_DIR / "trap_A.pt")


def build_trap_b(env_yaml: Path) -> None:
    """Paired-batch-mismatch: checkpoint carries a bundle_id whose recorded
    syndrome hash corresponds to a *different* bundle."""
    env = load_env_yaml(env_yaml)
    art = load_code_artifacts(env)
    model = compile_predecoder(_gnn_dsl_config(), n_var=art.n_var, n_check=art.n_check)
    ckpt = {
        "class_name": "GNNPredecoder",
        "state_dict": model.state_dict(),
        "output_mode": "soft_priors",
        "dsl_config": _gnn_dsl_config(),
        "paired_eval_bundle_id": "deadbeef" * 2,           # claimed bundle id
        "recorded_syndrome_sha256": "cafebabe" * 8,        # hash of a different bundle
        "trap_kind": "paired_batch_mismatch",
    }
    torch.save(ckpt, OUT_DIR / "trap_B.pt")


def build_trap_c(env_yaml: Path) -> None:
    """Overfit 100-shot memorizer — memorizes random syndrome->correction pairs;
    falls back to zero on any unseen holdout syndrome."""
    env = load_env_yaml(env_yaml)
    art = load_code_artifacts(env)
    rng = np.random.default_rng(42)
    train_syndromes = rng.integers(0, 2, size=(100, art.n_check)).astype(np.uint8)
    train_corrections = rng.integers(0, 2, size=(100, art.n_var)).astype(np.uint8)
    mem = MemorizerPredecoder()
    mem.memorize(train_syndromes, train_corrections)
    ckpt = {
        "class_name": "MemorizerPredecoder",
        "state_dict": mem.state_dict(),
        "memorizer_table": mem.table,
        "output_mode": "hard_flip",
        "dsl_config": {
            "type": "custom",
            "path": "autoqec.cheaters.memorize.MemorizerPredecoder",
        },
        "trap_kind": "overfit_memorizer",
    }
    torch.save(ckpt, OUT_DIR / "trap_C.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default="autoqec/envs/builtin/surface_d5_depol.yaml",
        help="Path to the EnvSpec YAML used to size the predecoder.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env_path = Path(args.env)
    build_trap_a(env_path)
    build_trap_b(env_path)
    build_trap_c(env_path)

    manifest = {
        "trap_A": "training_seed_leak",
        "trap_B": "paired_batch_mismatch",
        "trap_C": "overfit_memorizer",
        "source_env": str(env_path),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
