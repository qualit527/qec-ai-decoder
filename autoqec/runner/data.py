from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim
import torch

from autoqec.envs.schema import EnvSpec


@dataclass
class CodeArtifacts:
    code_type: str
    code_artifact: stim.Circuit | np.ndarray
    edge_index: torch.Tensor
    n_var: int
    n_check: int
    prior_p: torch.Tensor
    parity_check_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class SampleBatch:
    """One batch of sampled shots, split into the three surfaces callers
    need.

    ``syndrome`` feeds the predecoder's forward pass. ``errors`` is the
    supervised target for ``soft_priors`` training — one entry per DEM
    error mechanism (stim) or per physical qubit (parity-check). It has
    the same width as the predecoder's ``soft_priors`` output so
    ``BCE(pred, errors.float())`` is well-defined with no slicing.
    ``observables`` is the logical outcome used by the LER comparison at
    eval. For parity-check codes there is no separate observable surface,
    so ``observables`` aliases ``errors``.
    """

    syndrome: torch.Tensor
    errors: torch.Tensor
    observables: torch.Tensor


def _select_seeds(seed_range: tuple[int, int], n_shots: int, max_unique: int = 8) -> list[int]:
    start, end = seed_range
    count = min(max_unique, max(1, n_shots))
    return list(range(start, min(end + 1, start + count)))


def _stim_edge_index_and_prior(circuit: stim.Circuit) -> tuple[torch.Tensor, torch.Tensor]:
    dem = circuit.detector_error_model(decompose_errors=True)
    edges: list[tuple[int, int]] = []
    prior_p: list[float] = []
    var_idx = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        prior_p.append(float(instruction.args_copy()[0]))
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                edges.append((var_idx, target.val))
        var_idx += 1
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    prior = torch.tensor(prior_p, dtype=torch.float32)
    return edge_index, prior


def _parity_edge_index(parity_check: np.ndarray) -> torch.Tensor:
    check_idx, var_idx = np.nonzero(parity_check)
    return torch.tensor(np.stack([var_idx, check_idx]), dtype=torch.long)


def load_code_artifacts(env_spec: EnvSpec) -> CodeArtifacts:
    path = Path(env_spec.code.source)
    if env_spec.code.type == "stim_circuit":
        circuit = stim.Circuit.from_file(str(path))
        edge_index, prior = _stim_edge_index_and_prior(circuit)
        dem = circuit.detector_error_model(decompose_errors=True)
        return CodeArtifacts(
            code_type="stim_circuit",
            code_artifact=circuit,
            edge_index=edge_index,
            n_var=dem.num_errors,
            n_check=circuit.num_detectors,
            prior_p=prior,
        )
    if env_spec.code.type == "parity_check_matrix":
        parity_check = np.load(path).astype(np.uint8)
        edge_index = _parity_edge_index(parity_check)
        prior = torch.full((parity_check.shape[1],), float(env_spec.noise.p[0]), dtype=torch.float32)
        return CodeArtifacts(
            code_type="parity_check_matrix",
            code_artifact=parity_check,
            edge_index=edge_index,
            n_var=parity_check.shape[1],
            n_check=parity_check.shape[0],
            prior_p=prior,
            parity_check_matrix=parity_check,
        )
    raise ValueError(f"Unsupported code type: {env_spec.code.type}")


def _sample_stim(circuit: stim.Circuit, seed_range: tuple[int, int], n_shots: int) -> SampleBatch:
    """Sample from the DEM sampler so we get error labels alongside
    detectors and observables.

    We use ``stim.DetectorErrorModel.compile_sampler`` rather than
    ``circuit.compile_detector_sampler`` because the latter only exposes
    ``(detections, observables)`` — insufficient for supervised training
    of a soft-priors predecoder. The DEM sampler agrees with the circuit
    sampler on detector and observable marginals for decomposed DEMs.
    """
    dem = circuit.detector_error_model(decompose_errors=True)
    seeds = _select_seeds(seed_range, n_shots)
    per_seed = max(1, int(np.ceil(n_shots / len(seeds))))
    detections_all: list[np.ndarray] = []
    errors_all: list[np.ndarray] = []
    observables_all: list[np.ndarray] = []
    for seed in seeds:
        sampler = dem.compile_sampler(seed=seed)
        detections, observables, errors = sampler.sample(shots=per_seed, return_errors=True)
        detections_all.append(detections)
        errors_all.append(errors)
        observables_all.append(observables)
    detections = np.concatenate(detections_all, axis=0)[:n_shots]
    errors = np.concatenate(errors_all, axis=0)[:n_shots]
    observables = np.concatenate(observables_all, axis=0)[:n_shots]
    return SampleBatch(
        syndrome=torch.from_numpy(detections.astype(np.float32)),
        errors=torch.from_numpy(errors.astype(np.int64)),
        observables=torch.from_numpy(observables.astype(np.int64)),
    )


def _sample_parity(
    parity_check: np.ndarray,
    p_error: float,
    seed_range: tuple[int, int],
    n_shots: int,
) -> SampleBatch:
    seeds = _select_seeds(seed_range, n_shots)
    per_seed = max(1, int(np.ceil(n_shots / len(seeds))))
    syndromes_all: list[np.ndarray] = []
    errors_all: list[np.ndarray] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        errors = rng.binomial(1, p_error, size=(per_seed, parity_check.shape[1])).astype(np.uint8)
        syndromes = (errors @ parity_check.T) % 2
        syndromes_all.append(syndromes)
        errors_all.append(errors)
    syndromes = np.concatenate(syndromes_all, axis=0)[:n_shots]
    errors = np.concatenate(errors_all, axis=0)[:n_shots]
    errors_tensor = torch.from_numpy(errors.astype(np.int64))
    return SampleBatch(
        syndrome=torch.from_numpy(syndromes.astype(np.float32)),
        errors=errors_tensor,
        # Parity-check codes have no distinct observable surface; callers
        # that want "what LER compares against" can use `.observables`
        # unconditionally.
        observables=errors_tensor,
    )


def sample_syndromes(
    env_spec: EnvSpec,
    artifacts: CodeArtifacts,
    seed_range: tuple[int, int],
    n_shots: int,
) -> SampleBatch:
    if artifacts.code_type == "stim_circuit":
        return _sample_stim(artifacts.code_artifact, seed_range, n_shots)  # type: ignore[arg-type]
    return _sample_parity(
        artifacts.code_artifact,  # type: ignore[arg-type]
        float(env_spec.noise.p[0]),
        seed_range,
        n_shots,
    )
