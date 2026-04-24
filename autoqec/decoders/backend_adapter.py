from __future__ import annotations

from typing import Any

import numpy as np
import stim
from ldpc import bposd_decoder

from autoqec.envs.schema import EnvSpec


def _decode_osd_batch(
    syndrome: np.ndarray,
    parity_check: np.ndarray,
    *,
    priors: np.ndarray | None,
    osd_order: int,
) -> np.ndarray:
    out = np.zeros((syndrome.shape[0], parity_check.shape[1]), dtype=np.uint8)
    for idx in range(syndrome.shape[0]):
        if priors is not None:
            decoder = bposd_decoder(
                parity_check,
                error_rate=None,
                channel_probs=np.clip(priors[idx], 1e-4, 1 - 1e-4),
                osd_method="osd_cs",
                osd_order=osd_order,
            )
        else:
            decoder = bposd_decoder(
                parity_check,
                error_rate=0.05,
                osd_method="osd_cs",
                osd_order=osd_order,
            )
        out[idx] = decoder.decode(np.asarray(syndrome[idx], dtype=np.uint8))
    return out


def _rebuild_dem_with_priors(
    dem: stim.DetectorErrorModel, priors: np.ndarray
) -> stim.DetectorErrorModel:
    """Clone ``dem`` but replace each error mechanism's probability with
    the matching entry from ``priors``.

    ``priors`` is a length-``dem.num_errors`` vector in [0, 1]. We walk
    ``dem.flattened()`` in the same iteration order as the circuit-DEM
    converter in ``autoqec.runner.data``, which is the same order the
    predecoder's soft-priors output uses, so `priors[i]` corresponds
    exactly to the i-th error mechanism. Probabilities are clamped to
    (1e-12, 1 - 1e-12) so ``-log(p/(1-p))`` stays finite — pymatching
    itself refuses ``p in {0, 1}``.
    """
    new_dem = stim.DetectorErrorModel()
    i = 0
    for inst in dem.flattened():
        if inst.type == "error":
            targets = inst.targets_copy()
            p = float(priors[i])
            if not np.isfinite(p):
                p = float(inst.args_copy()[0])  # fall back to native prior
            p = max(min(p, 1 - 1e-12), 1e-12)
            new_dem.append("error", [p], targets)
            i += 1
        else:
            new_dem.append(inst)
    return new_dem


def _decode_mwpm_reweighted(
    priors: np.ndarray,
    syndrome: np.ndarray,
    circuit: stim.Circuit,
) -> np.ndarray:
    """Decode ``syndrome`` with MWPM where each sample's DEM error
    probabilities are overridden by ``priors[i]``.

    One ``Matching`` object is built per sample — this is the tractable
    price for honoring per-shot reweighting since pymatching has no
    per-sample weight override API. For B shots × |error mechanisms|
    around (100, 2k) this is O(B * build_cost) ≈ a few seconds on CPU,
    an acceptable floor while correctness is the priority.
    """
    import pymatching  # noqa: PLC0415 — lazy so OSD-only envs don't pay the import

    dem = circuit.detector_error_model(decompose_errors=True)
    n_shots = syndrome.shape[0]
    n_obs = dem.num_observables
    predictions = np.zeros((n_shots, n_obs), dtype=np.uint8)
    for b in range(n_shots):
        rebuilt = _rebuild_dem_with_priors(dem, priors[b])
        matching = pymatching.Matching.from_detector_error_model(rebuilt)
        predictions[b] = matching.decode(syndrome[b].astype(np.uint8))
    return predictions


def decode_with_predecoder(
    predecoder_output: Any,
    env_spec: EnvSpec,
    syndrome_raw: np.ndarray,
    code_artifact: stim.Circuit | np.ndarray,
    output_mode: str,
) -> np.ndarray:
    if env_spec.classical_backend == "mwpm":
        import pymatching  # noqa: PLC0415

        if not isinstance(code_artifact, stim.Circuit):
            raise TypeError("MWPM backend requires a Stim circuit artifact")
        if output_mode == "hard_flip":
            # Threshold the soft predecoder output (model is always trained
            # in differentiable form; the hard decision happens here, not
            # inside the model — otherwise gradients never flow).
            raw = np.asarray(predecoder_output)
            if raw.dtype != bool:
                cleaned = raw > 0.5
            else:
                cleaned = raw
            dem = code_artifact.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(dem)
            return matching.decode_batch(cleaned.astype(bool))
        # soft_priors: honor the predecoder's per-sample error probabilities
        # via DEM reweighting.
        priors = np.asarray(predecoder_output, dtype=float)
        return _decode_mwpm_reweighted(priors, np.asarray(syndrome_raw), code_artifact)

    if env_spec.classical_backend == "osd":
        parity_check = np.asarray(code_artifact, dtype=np.uint8)
        osd_order = max(env_spec.eval_protocol.osd_orders_reported or [0])
        if output_mode == "hard_flip":
            cleaned = np.asarray(predecoder_output, dtype=np.uint8)
            return _decode_osd_batch(cleaned, parity_check, priors=None, osd_order=osd_order)
        priors = np.asarray(predecoder_output, dtype=float)
        return _decode_osd_batch(
            np.asarray(syndrome_raw, dtype=np.uint8),
            parity_check,
            priors=priors,
            osd_order=osd_order,
        )

    raise ValueError(f"Unknown backend: {env_spec.classical_backend}")
