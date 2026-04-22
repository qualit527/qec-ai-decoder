"""Generate the canonical d=5 rotated surface-code circuit used by the MVP."""

from __future__ import annotations

from pathlib import Path

import stim


def generate(p: float = 5e-3, distance: int = 5, rounds: int = 5) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


if __name__ == "__main__":
    out = Path("circuits/surface_d5.stim")
    out.parent.mkdir(parents=True, exist_ok=True)
    circuit = generate()
    circuit.to_file(str(out))
    print(f"Wrote {out} ({circuit.num_qubits} qubits, {circuit.num_detectors} detectors)")

