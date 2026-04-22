"""PyMatching baseline benchmark on surface_d5 at circuit-level depolarising noise.

Purpose
-------

Pin the PyMatching numbers the Analyst will compare Δ LER against, and give
B's verification slice a known-good reference for holdout evaluation.

Default is 1,000,000 shots (~1–3 min on a laptop). Use `--n-shots` to
shrink for smoke runs, `--output` to capture the JSON to disk.

Example::

    python scripts/benchmark_surface_baseline.py --n-shots 1000000 \\
        --output demos/demo-1-surface-d5/expected_output/baseline_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import stim

from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline


def benchmark(
    n_shots: int = 1_000_000,
    seed: int = 42,
    circuit_path: str | Path = "circuits/surface_d5.stim",
) -> dict:
    """Run PyMatching on `n_shots` shots of `circuit_path` and return timing + LER."""
    circuit = stim.Circuit.from_file(str(circuit_path))
    decoder = PymatchingBaseline.from_circuit(circuit)

    sampler = circuit.compile_detector_sampler(seed=seed)
    t0 = time.time()
    detections, observables = sampler.sample(shots=n_shots, separate_observables=True)
    t_sample = time.time() - t0

    t0 = time.time()
    predictions = decoder.decode_batch(detections)
    t_decode = time.time() - t0

    errors = int(np.asarray(predictions != observables).any(axis=1).sum())
    return {
        "n_shots": int(n_shots),
        "ler": float(errors / n_shots),
        "n_errors": errors,
        "t_sample_s": float(t_sample),
        "t_decode_s": float(t_decode),
        "detections_shape": list(detections.shape),
        "circuit_path": str(circuit_path),
        "seed": int(seed),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-shots", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--circuit", default="circuits/surface_d5.stim")
    p.add_argument("--output", help="write JSON result to this path as well as stdout")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = benchmark(n_shots=args.n_shots, seed=args.seed, circuit_path=args.circuit)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)


if __name__ == "__main__":
    main()
