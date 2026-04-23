import pytest
import numpy as np
import torch
from pathlib import Path
import types


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


def test_load_predecoder_legacy_model_and_double_failure(monkeypatch):
    from autoqec.eval.independent_eval import _load_predecoder

    ckpt = Path("dummy.pt")
    ckpt.write_text("stub", encoding="utf-8")
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: {"model": model})
    assert _load_predecoder(ckpt, 1, 1) is model

    calls = {"n": 0}

    def raising_load(*_args, **_kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(torch, "load", raising_load)
    assert _load_predecoder(ckpt, 1, 1) is None
    assert calls["n"] == 2
    ckpt.unlink()


def test_load_code_artifacts_and_holdout_sampling_cover_parity_path():
    from autoqec.eval.independent_eval import _load_code_artifacts, _sample_holdout
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    artifacts = _load_code_artifacts(env)
    assert artifacts.code_type == "parity_check_matrix"
    assert artifacts.parity_check_matrix is not None

    synd, err = _sample_holdout(env, artifacts, holdout_seeds=[9000, 9001], n_shots=5)
    assert synd.shape[0] == 5
    assert err.shape[0] == 5

    bad_env = env.model_copy(update={"code": env.code.model_copy(update={"type": "tanner_graph"})})
    with pytest.raises(ValueError, match="Unsupported code type"):
        _load_code_artifacts(bad_env)


def test_decode_holdout_covers_model_none_and_osd_path(monkeypatch):
    from autoqec.eval import independent_eval as ie
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    artifacts = ie._load_code_artifacts(env)
    syndrome = torch.zeros((2, artifacts.n_check), dtype=torch.float32)
    target = torch.zeros((2, artifacts.n_var), dtype=torch.int64)

    monkeypatch.setattr(ie, "_sample_holdout", lambda *_args, **_kwargs: (syndrome, target))
    plain, pred, shuffled = ie._decode_holdout(None, env, artifacts, [9000], 2)
    assert plain.shape == pred.shape == shuffled.shape == (2,)


def test_decode_holdout_sets_parity_ctx_and_ablation(monkeypatch):
    from autoqec.eval import independent_eval as ie
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    artifacts = ie._load_code_artifacts(env)
    syndrome = torch.zeros((2, artifacts.n_check), dtype=torch.float32)
    target = torch.zeros((2, artifacts.n_var), dtype=torch.int64)
    monkeypatch.setattr(ie, "_sample_holdout", lambda *_args, **_kwargs: (syndrome, target))

    class FakeModel(torch.nn.Module):
        output_mode = "soft_priors"

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, syndrome, ctx):
            assert "parity_check_matrix" in ctx
            return torch.full((syndrome.shape[0], artifacts.n_var), 0.1)

    model = FakeModel()
    monkeypatch.setattr(ie, "decode_with_predecoder", lambda *_args, **_kwargs: np.zeros((2, artifacts.n_var), dtype=np.int64))
    plain, pred, shuffled = ie._decode_holdout(model, env, artifacts, [9000], 2)
    assert plain.shape == pred.shape == shuffled.shape == (2,)


def test_independent_verify_can_return_failed_and_verified(monkeypatch):
    from autoqec.eval import independent_eval as ie
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    monkeypatch.setattr(ie, "_load_code_artifacts", lambda _env: types.SimpleNamespace(n_var=1, n_check=1))
    monkeypatch.setattr(ie, "_load_predecoder", lambda *_args, **_kwargs: object())

    def fake_decode_failed(*_args, **_kwargs):
        return (
            np.array([0, 0, 0], dtype=np.int32),
            np.array([1, 1, 1], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int32),
        )

    monkeypatch.setattr(ie, "_decode_holdout", fake_decode_failed)
    monkeypatch.setattr(
        ie,
        "bootstrap_ci_mean",
        lambda outcomes, *_args, **_kwargs: (float(outcomes.mean()), 0.1, 0.1001),
    )
    failed = ie.independent_verify(Path("dummy.pt"), env, holdout_seeds=[9000], n_shots=3)
    assert failed.verdict == "FAILED"

    def fake_decode_verified(*_args, **_kwargs):
        return (
            np.array([1, 1, 1], dtype=np.int32),
            np.array([0, 0, 0], dtype=np.int32),
            np.array([1, 1, 1], dtype=np.int32),
        )

    monkeypatch.setattr(ie, "_decode_holdout", fake_decode_verified)
    monkeypatch.setattr(
        ie,
        "bootstrap_ci_mean",
        lambda outcomes, *_args, **_kwargs: (float(outcomes.mean()), 0.4, 0.5),
    )
    verified = ie.independent_verify(Path("dummy.pt"), env, holdout_seeds=[9000], n_shots=3)
    assert verified.verdict == "VERIFIED"
