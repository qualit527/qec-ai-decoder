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


def decode_with_predecoder(
    predecoder_output: Any,
    env_spec: EnvSpec,
    syndrome_raw: np.ndarray,
    code_artifact: stim.Circuit | np.ndarray,
    output_mode: str,
) -> np.ndarray:
    if env_spec.classical_backend == "mwpm":
        import pymatching

        if not isinstance(code_artifact, stim.Circuit):
            raise TypeError("MWPM backend requires a Stim circuit artifact")
        dem = code_artifact.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        if output_mode == "hard_flip":
            cleaned = np.asarray(predecoder_output, dtype=bool)
            return matching.decode_batch(cleaned)
        return matching.decode_batch(np.asarray(syndrome_raw, dtype=bool))

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
