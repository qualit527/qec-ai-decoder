import pytest
import torch
from pathlib import Path


def test_independent_verify_identity_predecoder(tmp_path):
    """Identity predecoder (no model loaded) must produce verdict=SUSPICIOUS
    and delta_ler ≈ 0."""
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    # Nonexistent checkpoint → identity fallback
    ckpt = tmp_path / "identity.pt"
    torch.save({"class_name": "IdentityPredecoder", "state_dict": {}}, ckpt)
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9010)),
                                n_shots=3000, n_bootstrap=200)
    assert report.seed_leakage_check_ok is True
    assert report.verdict == "SUSPICIOUS"
    assert abs(report.delta_ler_holdout) < 0.05


def test_independent_verify_rejects_leaky_train_seeds():
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    # Seed 500 is in train range [1, 999]
    with pytest.raises(ValueError, match="holdout seeds overlaps"):
        independent_verify(Path("nonexistent.pt"), env, holdout_seeds=[500])


def test_independent_verify_rejects_leaky_val_seeds():
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    # Seed 1500 is in val range [1000, 1999]
    with pytest.raises(ValueError, match="holdout seeds overlaps"):
        independent_verify(Path("nonexistent.pt"), env, holdout_seeds=[1500])


def test_independent_verify_rejects_out_of_holdout_range():
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    # Seed 5000 is NOT in holdout range [9000, 9999]
    with pytest.raises(ValueError, match="holdout seeds overlaps"):
        independent_verify(Path("nonexistent.pt"), env, holdout_seeds=[5000])
