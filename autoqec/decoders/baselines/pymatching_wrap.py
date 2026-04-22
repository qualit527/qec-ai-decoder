from __future__ import annotations

import numpy as np
import pymatching
import stim


class PymatchingBaseline:
    """Thin wrapper around PyMatching for surface-code style DEM decoding."""

    def __init__(self, matching: pymatching.Matching, n_observables: int):
        self.matching = matching
        self.n_observables = n_observables

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> "PymatchingBaseline":
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        return cls(matching, circuit.num_observables)

    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        return self.matching.decode_batch(np.asarray(detections, dtype=bool))

