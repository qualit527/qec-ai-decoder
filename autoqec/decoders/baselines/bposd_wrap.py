from __future__ import annotations
import numpy as np
from ldpc import bposd_decoder


class BpOsdBaseline:
    """Wrap ldpc.bposd_decoder for batch decoding. One decoder per (H, p, order)."""

    def __init__(self, H: np.ndarray, error_rate: float, osd_order: int = 0,
                 bp_method: str = "ps", max_iter: int = 50):
        self.dec = bposd_decoder(
            H,
            error_rate=error_rate,
            bp_method=bp_method,
            max_iter=max_iter,
            osd_method="osd_e" if osd_order > 0 else "osd_0",
            osd_order=osd_order,
        )
        self.n_bits = H.shape[1]

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        out = np.empty((syndromes.shape[0], self.n_bits), dtype=np.uint8)
        for i in range(syndromes.shape[0]):
            out[i] = self.dec.decode(syndromes[i].astype(np.uint8))
        return out
