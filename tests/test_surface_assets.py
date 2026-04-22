from pathlib import Path

import numpy as np
import stim

from autoqec.envs.schema import load_env_yaml


def test_surface_d5_circuit_exists() -> None:
    path = Path("circuits/surface_d5.stim")
    assert path.exists()
    circuit = stim.Circuit.from_file(str(path))
    assert circuit.num_qubits > 0
    assert circuit.num_detectors > 0
    assert circuit.num_observables >= 1


def test_surface_env_yaml_loads() -> None:
    spec = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    assert spec.name == "surface_d5_depol"
    assert spec.classical_backend == "mwpm"
    assert spec.noise.p == [1e-3, 5e-3, 1e-2]


def test_bb72_artifacts_exist() -> None:
    hx = np.load("circuits/bb72_Hx.npy")
    hz = np.load("circuits/bb72_Hz.npy")
    assert hx.shape == (36, 72)
    assert hz.shape == (36, 72)

