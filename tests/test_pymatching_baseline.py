import stim

from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline


def test_pymatching_on_surface_d5() -> None:
    circuit = stim.Circuit.from_file("circuits/surface_d5.stim")
    decoder = PymatchingBaseline.from_circuit(circuit)
    sampler = circuit.compile_detector_sampler(seed=1)
    detections, observables = sampler.sample(shots=1000, separate_observables=True)
    predictions = decoder.decode_batch(detections)
    assert predictions.shape == observables.shape
    ler = (predictions != observables).mean()
    assert 0.0 <= ler <= 0.5

