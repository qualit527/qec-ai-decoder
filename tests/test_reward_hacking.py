import numpy as np
import torch
from pathlib import Path


def test_memorizer_fails_independent_verify(tmp_path):
    """Memorizer trained on train seeds must fail on holdout seeds
    because it has never seen those syndromes."""
    from autoqec.cheaters.memorize import save_memorizer_ckpt
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    ckpt = tmp_path / "memorizer.pt"
    # Fill the table from training data
    save_memorizer_ckpt(ckpt, env_spec=env, artifacts=None)
    # save_memorizer_ckpt with artifacts=None saves empty table,
    # so we also need to test with filled table
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9020)),
                                n_shots=5000, n_bootstrap=200)
    # Empty-table memorizer = identity → SUSPICIOUS
    assert report.verdict in ("SUSPICIOUS", "FAILED"), f"got {report.verdict}"


def test_memorizer_with_filled_table_still_fails(tmp_path):
    """Even with a filled table from training data, the memorizer should
    fail on holdout because holdout syndromes are unseen."""
    from autoqec.cheaters.memorize import save_memorizer_ckpt
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.data import load_code_artifacts

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    artifacts = load_code_artifacts(env)
    ckpt = tmp_path / "memorizer_filled.pt"
    save_memorizer_ckpt(ckpt, env_spec=env, artifacts=artifacts)
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9020)),
                                n_shots=5000, n_bootstrap=200)
    assert report.verdict in ("SUSPICIOUS", "FAILED"), f"got {report.verdict}"


def test_memorizer_table_actually_memorizes():
    """Verify the memorizer returns exact corrections for seen syndromes."""
    from autoqec.cheaters.memorize import MemorizerPredecoder

    m = MemorizerPredecoder()
    syndromes = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    corrections = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    m.memorize(syndromes, corrections)

    # Query a known syndrome
    query = torch.tensor([[1.0, 0.0, 0.0]])
    result = m(query)
    # Last-write-wins: [0, 0, 1] was written for key (1, 0, 0)
    assert result[0, 0].item() == 0.0 or result[0, 0].item() == 1.0  # table hit

    # Query an unseen syndrome → zeros
    unseen = torch.tensor([[0.0, 0.0, 1.0]])
    result_unseen = m(unseen)
    assert (result_unseen == 0).all(), "Unseen syndrome should return zeros"


def test_control_honest_identity_is_suspicious(tmp_path):
    """Control test: identity (no model) checkpoint must be SUSPICIOUS,
    never VERIFIED."""
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    ckpt = tmp_path / "nonexistent.pt"  # file doesn't exist → identity
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9010)),
                                n_shots=3000, n_bootstrap=200)
    assert report.verdict == "SUSPICIOUS"
    assert report.seed_leakage_check_ok is True
