import numpy as np
import stim

from autoqec.decoders.backend_adapter import decode_with_predecoder
from autoqec.envs.schema import load_env_yaml


def test_hard_flip_passes_to_mwpm() -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    circuit = stim.Circuit.from_file(env.code.source)
    sampler = circuit.compile_detector_sampler(seed=0)
    detections, observables = sampler.sample(shots=500, separate_observables=True)
    out = decode_with_predecoder(detections.astype(bool), env, detections, circuit, "hard_flip")
    assert out.shape == observables.shape


def test_soft_priors_drive_osd_decode_shape() -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    parity = np.load(env.code.source)
    syndrome = np.zeros((3, parity.shape[0]), dtype=np.uint8)
    priors = np.full((3, parity.shape[1]), 0.05, dtype=float)

    out = decode_with_predecoder(priors, env, syndrome, parity, "soft_priors")

    assert out.shape == (3, parity.shape[1])


def test_hard_flip_drives_osd_decode_shape() -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    parity = np.load(env.code.source)
    cleaned = np.zeros((2, parity.shape[0]), dtype=np.uint8)

    out = decode_with_predecoder(cleaned, env, cleaned, parity, "hard_flip")

    assert out.shape == (2, parity.shape[1])
