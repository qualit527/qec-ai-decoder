"""ISOLATED — must not import the runner module.

The verifier:
  1. Seed-isolation check (holdout seeds must fall within seed_policy.holdout range).
  2. Re-sample holdout detection events using Stim + the env's circuit.
  3. Run the predecoder + classical backend on holdout → LER + bootstrap CI.
  4. Ablation sanity: shuffle predecoder params → LER must degrade.
  5. Verdict rule.

Data-loading helpers are copy-pasted (not imported) to maintain isolation.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymatching
import stim
import torch

from autoqec.decoders.backend_adapter import decode_with_predecoder
from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline
from autoqec.decoders.dsl_compiler import compile_predecoder
from autoqec.envs.schema import EnvSpec
from autoqec.eval.bootstrap import bootstrap_ci_mean
from autoqec.eval.schema import VerifyReport


# ---------------------------------------------------------------------------
# Copy-pasted data helpers — kept local for isolation
# ---------------------------------------------------------------------------

@dataclass
class _CodeArtifacts:
    code_type: str
    code_artifact: stim.Circuit | np.ndarray
    edge_index: torch.Tensor
    n_var: int
    n_check: int
    prior_p: torch.Tensor
    parity_check_matrix: np.ndarray | None = None


def _load_code_artifacts(env_spec: EnvSpec) -> _CodeArtifacts:
    path = Path(env_spec.code.source)
    if env_spec.code.type == "stim_circuit":
        circuit = stim.Circuit.from_file(str(path))
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
        return _CodeArtifacts(
            code_type="stim_circuit",
            code_artifact=circuit,
            edge_index=edge_index,
            n_var=dem.num_errors,
            n_check=circuit.num_detectors,
            prior_p=prior,
        )
    if env_spec.code.type == "parity_check_matrix":
        parity_check = np.load(path).astype(np.uint8)
        check_idx, var_idx = np.nonzero(parity_check)
        edge_index = torch.tensor(np.stack([var_idx, check_idx]), dtype=torch.long)
        prior = torch.full(
            (parity_check.shape[1],), float(env_spec.noise.p[0]), dtype=torch.float32,
        )
        return _CodeArtifacts(
            code_type="parity_check_matrix",
            code_artifact=parity_check,
            edge_index=edge_index,
            n_var=parity_check.shape[1],
            n_check=parity_check.shape[0],
            prior_p=prior,
            parity_check_matrix=parity_check,
        )
    raise ValueError(f"Unsupported code type: {env_spec.code.type}")


def _select_seeds(seed_range: tuple[int, int], n_shots: int, max_unique: int = 8) -> list[int]:
    start, end = seed_range
    count = min(max_unique, max(1, n_shots))
    return list(range(start, min(end + 1, start + count)))


def _sample_holdout(
    env_spec: EnvSpec,
    artifacts: _CodeArtifacts,
    holdout_seeds: list[int],
    n_shots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample syndromes + targets for holdout evaluation."""
    seeds = holdout_seeds
    per_seed = max(1, int(math.ceil(n_shots / len(seeds))))
    total = per_seed * len(seeds)

    if artifacts.code_type == "stim_circuit":
        circuit = artifacts.code_artifact
        det_all, obs_all = [], []
        for s in seeds:
            sampler = circuit.compile_detector_sampler(seed=s)
            det, obs = sampler.sample(shots=per_seed, separate_observables=True)
            det_all.append(det)
            obs_all.append(obs)
        det = np.concatenate(det_all, axis=0)[:n_shots]
        obs = np.concatenate(obs_all, axis=0)[:n_shots]
        return (
            torch.from_numpy(det.astype(np.float32)),
            torch.from_numpy(obs.astype(np.int64)),
        )

    # Parity-check mode
    pc = artifacts.code_artifact
    synd_all, err_all = [], []
    for s in seeds:
        rng = np.random.default_rng(s)
        errors = rng.binomial(1, float(env_spec.noise.p[0]), size=(per_seed, pc.shape[1])).astype(np.uint8)
        syndromes = (errors @ pc.T) % 2
        synd_all.append(syndromes)
        err_all.append(errors)
    synd = np.concatenate(synd_all, axis=0)[:n_shots]
    err = np.concatenate(err_all, axis=0)[:n_shots]
    return (
        torch.from_numpy(synd.astype(np.float32)),
        torch.from_numpy(err.astype(np.int64)),
    )


# ---------------------------------------------------------------------------
# Core verifier
# ---------------------------------------------------------------------------

def _seed_leakage_check(
    train: tuple[int, int],
    val: tuple[int, int],
    holdout_range: tuple[int, int],
    holdout_seeds: list[int],
) -> bool:
    for s in holdout_seeds:
        if train[0] <= s <= train[1]:
            return False
        if val[0] <= s <= val[1]:
            return False
        if not (holdout_range[0] <= s <= holdout_range[1]):
            return False
    return True


def _load_predecoder(ckpt: Path, n_var: int, n_check: int):
    """Load a predecoder from Runner checkpoint format.

    Runner writes: {"class_name", "state_dict", "output_mode", "dsl_config"}.
    We rebuild the module via compile_predecoder and load the state_dict.
    """
    if not ckpt.exists():
        return None
    try:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception:
        return None

    if blob.get("class_name") == "IdentityPredecoder":
        return None

    # Runner format: rebuild from dsl_config
    dsl_config = blob.get("dsl_config")
    state_dict = blob.get("state_dict")
    if dsl_config is not None and state_dict is not None:
        model = compile_predecoder(dsl_config, n_var, n_check)
        model.load_state_dict(state_dict)
        return model

    # Legacy format: direct model object
    model = blob.get("model")
    if isinstance(model, torch.nn.Module):
        return model

    return None


def _shuffle_model_params(model: torch.nn.Module) -> None:
    for param in model.parameters():
        with torch.no_grad():
            param.data = param.data[
                torch.randperm(param.data.numel()).reshape(param.data.shape)
            ]


def _decode_holdout(
    model: torch.nn.Module | None,
    env_spec: EnvSpec,
    artifacts: _CodeArtifacts,
    holdout_seeds: list[int],
    n_shots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run plain baseline, predecoder, and ablation on the same holdout shots.

    Returns (plain_errors, pred_errors, shuffled_errors).
    """
    syndrome, target = _sample_holdout(env_spec, artifacts, holdout_seeds, n_shots)
    syndrome_np = syndrome.numpy()
    target_np = target.numpy()
    n_obs = target_np.shape[1]

    # Plain baseline
    if env_spec.classical_backend == "mwpm":
        baseline = PymatchingBaseline.from_circuit(artifacts.code_artifact)
        plain_pred = baseline.decode_batch(syndrome_np.astype(bool))
    else:
        uniform = np.full(
            (syndrome_np.shape[0], artifacts.n_var),
            float(env_spec.noise.p[0]),
        )
        plain_pred = decode_with_predecoder(
            uniform, env_spec, syndrome_np, artifacts.code_artifact, "soft_priors",
        )
    plain_errors = (plain_pred[:, :n_obs] != target_np).any(axis=1).astype(np.int32)

    if model is None:
        pred_errors = plain_errors.copy()
        shuffled_errors = plain_errors.copy()
        return plain_errors, pred_errors, shuffled_errors

    # Predecoder path
    model.eval()
    ctx = {
        "edge_index": artifacts.edge_index,
        "n_var": artifacts.n_var,
        "n_check": artifacts.n_check,
        "prior_p": artifacts.prior_p,
    }
    if artifacts.parity_check_matrix is not None:
        ctx["parity_check_matrix"] = torch.from_numpy(artifacts.parity_check_matrix)

    with torch.no_grad():
        pred_out = model(syndrome, ctx).numpy()
    pred_labels = decode_with_predecoder(
        pred_out, env_spec, syndrome_np, artifacts.code_artifact, model.output_mode,
    )
    pred_errors = (pred_labels[:, :n_obs] != target_np).any(axis=1).astype(np.int32)

    # Ablation: shuffle params and re-run
    ablation_model = copy.deepcopy(model)
    _shuffle_model_params(ablation_model)
    ablation_model.eval()
    with torch.no_grad():
        ablation_out = ablation_model(syndrome, ctx).numpy()
    ablation_labels = decode_with_predecoder(
        ablation_out, env_spec, syndrome_np, artifacts.code_artifact, ablation_model.output_mode,
    )
    shuffled_errors = (ablation_labels[:, :n_obs] != target_np).any(axis=1).astype(np.int32)

    return plain_errors, pred_errors, shuffled_errors


def independent_verify(
    checkpoint: Path,
    env_spec: EnvSpec,
    holdout_seeds: list[int],
    n_shots: int | None = None,
    n_bootstrap: int = 1000,
) -> VerifyReport:
    sp = env_spec.noise.seed_policy
    if not _seed_leakage_check(sp.train, sp.val, sp.holdout, holdout_seeds):
        raise ValueError("holdout seeds overlaps train/val range or falls outside holdout policy")

    n_shots = math.ceil(n_shots or env_spec.eval_protocol.min_shots_verify)

    artifacts = _load_code_artifacts(env_spec)
    model = _load_predecoder(checkpoint, artifacts.n_var, artifacts.n_check)

    plain_errors, pred_errors, shuffled_errors = _decode_holdout(
        model, env_spec, artifacts, holdout_seeds, n_shots,
    )

    ler_plain, plo, phi = bootstrap_ci_mean(plain_errors, n_bootstrap, 0.95, seed=0)
    ler_pred, _, _ = bootstrap_ci_mean(pred_errors, n_bootstrap, 0.95, seed=1)
    ler_shuffled = float(shuffled_errors.mean())

    delta = float(ler_plain - ler_pred)
    ci_half = (phi - plo) / 2
    ablation_ok = ler_shuffled >= ler_pred - 1e-4

    # Verdict rule
    if model is None:
        verdict = "SUSPICIOUS"
    elif not ablation_ok:
        verdict = "FAILED"
    elif delta < -ci_half:
        verdict = "FAILED"
    elif abs(delta) < ci_half:
        verdict = "SUSPICIOUS"
    else:
        verdict = "VERIFIED"

    return VerifyReport(
        verdict=verdict,
        ler_holdout=ler_pred,
        ler_holdout_ci=(plo, phi),
        delta_ler_holdout=delta,
        ler_shuffled=ler_shuffled,
        ablation_sanity_ok=ablation_ok,
        holdout_seeds_used=holdout_seeds,
        seed_leakage_check_ok=True,
        notes=f"n_shots={n_shots}, plain_ler={ler_plain:.4g}, pred_ler={ler_pred:.4g}, shuffled_ler={ler_shuffled:.4g}",
    )
