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

