import pytest
import torch
import numpy as np
from pathlib import Path


def test_independent_verify_honest_predecoder(tmp_path):
    """A trivially-identity predecoder (returns the syndrome unchanged) must
    produce ablation_sanity_ok=True, delta_ler ≈ 0, and verdict != FAILED."""
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    # Build a fake "identity" checkpoint
    ckpt = tmp_path / "identity.pt"
    torch.save({"class_name": "IdentityPredecoder", "state_dict": {}}, ckpt)
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9010)),
                                n_shots=5000, n_bootstrap=200)
    assert report.seed_leakage_check_ok is True
    assert report.verdict in ("VERIFIED", "SUSPICIOUS")
    assert abs(report.delta_ler_holdout) < 0.05


def test_independent_verify_rejects_leaky_seeds():
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    # Train-range seed 500 in holdout → must flag leak
    with pytest.raises(ValueError, match="holdout.*overlaps"):
        independent_verify(Path("nonexistent.pt"), env, holdout_seeds=[500])
