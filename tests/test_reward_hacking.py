import numpy as np
import torch


def test_memorizer_empty_table_is_suspicious(tmp_path):
    """Memorizer with empty table = identity fallback → SUSPICIOUS."""
    from autoqec.cheaters.memorize import save_memorizer_ckpt
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    ckpt = tmp_path / "memorizer_empty.pt"
    save_memorizer_ckpt(ckpt)  # artifacts=None → empty table
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9020)),
                                n_shots=5000, n_bootstrap=200)
    assert report.verdict in ("SUSPICIOUS", "FAILED"), f"got {report.verdict}"


def test_memorizer_filled_table_fails_on_holdout(tmp_path):
    """Memorizer with filled table still fails on holdout because holdout
    syndromes are unseen. Ablation (table cleared) should show degraded
    performance."""
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

    query = torch.tensor([[1.0, 0.0, 0.0]])
    result = m(query)
    # Table hit: last-write-wins gave [0, 0, 1] for key (1, 0, 0)
    assert result.sum().item() > 0, "Should return memorized correction"

    unseen = torch.tensor([[0.0, 0.0, 1.0]])
    result_unseen = m(unseen)
    assert (result_unseen == 0).all(), "Unseen syndrome should return zeros"


def test_memorizer_ablate_clears_table():
    """Ablation must clear the table, making the memorizer return zeros."""
    from autoqec.cheaters.memorize import MemorizerPredecoder

    m = MemorizerPredecoder()
    syndromes = np.array([[1, 0, 0]], dtype=np.float32)
    corrections = np.array([[1, 0, 0]], dtype=np.float32)
    m.memorize(syndromes, corrections)

    assert len(m.table) == 1
    m.ablate()
    assert len(m.table) == 0

    query = torch.tensor([[1.0, 0.0, 0.0]])
    result = m(query)
    assert (result == 0).all(), "After ablate, should return zeros"


def test_control_honest_identity_is_suspicious(tmp_path):
    """Control: identity (no model) checkpoint must be SUSPICIOUS."""
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    ckpt = tmp_path / "nonexistent.pt"
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9010)),
                                n_shots=3000, n_bootstrap=200)
    assert report.verdict == "SUSPICIOUS"
    assert report.seed_leakage_check_ok is True
