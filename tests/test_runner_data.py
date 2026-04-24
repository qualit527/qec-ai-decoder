from __future__ import annotations

import numpy as np
import stim
import torch

from autoqec.envs.schema import load_env_yaml
from autoqec.runner import data as runner_data


def test_select_seeds_respects_range_and_caps_unique_count() -> None:
    assert runner_data._select_seeds((10, 12), 0) == [10]
    assert runner_data._select_seeds((10, 12), 2) == [10, 11]
    assert runner_data._select_seeds((10, 30), 20) == list(range(10, 18))


def test_stim_artifacts_and_sampling_cover_stim_path() -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    artifacts = runner_data.load_code_artifacts(env)

    assert artifacts.code_type == "stim_circuit"
    assert artifacts.n_var > 0
    assert artifacts.n_check > 0
    assert artifacts.edge_index.shape[0] == 2
    assert artifacts.prior_p.shape[0] == artifacts.n_var

    batch = runner_data.sample_syndromes(
        env,
        artifacts,
        env.noise.seed_policy.train,
        5,
    )
    assert batch.syndrome.shape[0] == 5
    assert batch.syndrome.shape[1] == artifacts.n_check
    assert batch.errors.shape[0] == 5
    assert batch.observables.shape[0] == 5


def test_stim_edge_index_and_prior_skips_non_error_instructions() -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    circuit = stim.Circuit.from_file(env.code.source)
    edge_index, prior = runner_data._stim_edge_index_and_prior(circuit)

    assert edge_index.shape[0] == 2
    assert prior.ndim == 1
    assert prior.numel() > 0


def test_parity_edge_index_and_sampling_cover_parity_path() -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    artifacts = runner_data.load_code_artifacts(env)

    assert artifacts.code_type == "parity_check_matrix"
    assert artifacts.parity_check_matrix is not None
    assert artifacts.edge_index.shape[0] == 2
    assert artifacts.prior_p.shape[0] == artifacts.n_var

    batch = runner_data.sample_syndromes(
        env,
        artifacts,
        env.noise.seed_policy.train,
        6,
    )
    assert batch.syndrome.shape == (6, artifacts.n_check)
    assert batch.errors.shape == (6, artifacts.n_var)


def test_parity_edge_index_uses_nonzero_entries() -> None:
    parity = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    edge_index = runner_data._parity_edge_index(parity)
    assert torch.equal(edge_index, torch.tensor([[0, 2, 1], [0, 0, 1]]))


def test_sample_parity_respects_requested_shot_count() -> None:
    parity = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    batch = runner_data._sample_parity(parity, 0.1, (1, 5), 3)
    assert batch.syndrome.shape == (3, 2)
    assert batch.errors.shape == (3, 3)
    # parity-check path aliases observables to errors
    assert torch.equal(batch.errors, batch.observables)


def test_sample_syndromes_stim_returns_errors_and_observables() -> None:
    """For stim_circuit envs, sampling must yield DEM-error labels (for
    supervised soft_priors training) AND logical observables (for LER
    computation). Prior to 2026-04-24 the sampler only exposed observables,
    forcing the training loop to use `target=syndrome` — identity learning.
    """
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    artifacts = runner_data.load_code_artifacts(env)

    batch = runner_data.sample_syndromes(env, artifacts, env.noise.seed_policy.train, 8)

    assert batch.syndrome.shape == (8, artifacts.n_check)
    assert batch.errors.shape == (8, artifacts.n_var), (
        "errors shape must match predecoder soft_priors output width (n_var = "
        "num DEM error mechanisms) so BCE(pred, errors) is well-defined"
    )
    # Single-logical surface code: num_observables = 1.
    assert batch.observables.shape == (8, 1)
    # Errors and observables must NOT be identical arrays — they are distinct
    # quantities for stim circuits.
    assert batch.errors.shape != batch.observables.shape


def test_sample_syndromes_parity_populates_all_three_fields() -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    artifacts = runner_data.load_code_artifacts(env)

    batch = runner_data.sample_syndromes(env, artifacts, env.noise.seed_policy.train, 6)

    assert batch.syndrome.shape == (6, artifacts.n_check)
    assert batch.errors.shape == (6, artifacts.n_var)
    # Parity-check codes have no separate observables; we duplicate errors
    # so callers can still ask for `.observables` uniformly.
    assert torch.equal(batch.errors, batch.observables)


def test_load_code_artifacts_rejects_unsupported_code_type() -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    bad_env = env.model_copy(
        update={"code": env.code.model_copy(update={"type": "tanner_graph"})}
    )
    try:
        runner_data.load_code_artifacts(bad_env)
    except ValueError as exc:
        assert "Unsupported code type" in str(exc)
    else:
        raise AssertionError("expected unsupported code type to raise ValueError")
