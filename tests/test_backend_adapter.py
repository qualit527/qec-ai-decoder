import numpy as np
import pymatching
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


def _dem_native_priors(dem: stim.DetectorErrorModel) -> np.ndarray:
    """Extract the DEM's built-in per-mechanism probabilities."""
    priors: list[float] = []
    for inst in dem.flattened():
        if inst.type == "error":
            priors.append(float(inst.args_copy()[0]))
    return np.asarray(priors, dtype=float)


def test_mwpm_soft_priors_with_native_priors_matches_plain_baseline() -> None:
    """When the predecoder outputs the DEM's own prior probabilities as
    soft_priors, the reweighted MWPM must reproduce the plain baseline's
    decode (up to tie-breaking). If they differ, the reweighting logic
    is introducing spurious distortion.
    """
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    circuit = stim.Circuit.from_file(env.code.source)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = dem.compile_sampler(seed=0)
    detections, _observables, _errors = sampler.sample(shots=64, return_errors=True)

    matching = pymatching.Matching.from_detector_error_model(dem)
    plain = matching.decode_batch(detections.astype(bool))

    native = _dem_native_priors(dem)
    priors_batch = np.broadcast_to(native, (detections.shape[0], native.size)).copy()
    reweighted = decode_with_predecoder(
        priors_batch,
        env,
        detections.astype(np.uint8),
        circuit,
        "soft_priors",
    )
    assert reweighted.shape == plain.shape
    assert np.array_equal(reweighted, plain), (
        "Native-prior reweighting must reproduce the plain MWPM result"
    )


def test_mwpm_soft_priors_honor_predecoder_output(monkeypatch) -> None:
    """Predecoder priors must flow through the MWPM backend on every
    shot. Prior to 2026-04-24 the adapter silently called
    ``matching.decode_batch(syndrome_raw)`` regardless of
    ``predecoder_output`` and gave delta_ler=0 by construction on
    every surface-code run.

    Tested as a white-box check on the reweighted code path: the
    per-sample DEM rebuild (``_rebuild_dem_with_priors``) must fire
    once per shot and see the priors the caller passed. A purely
    behavioural "predictions differ" check turned out to be
    platform-flaky because pymatching's tie-break on uniform edge
    weights can coincide with the native-weighted matching on short
    shot counts.
    """
    from autoqec.decoders import backend_adapter

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    circuit = stim.Circuit.from_file(env.code.source)
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = dem.compile_sampler(seed=0)
    detections, _observables, _errors = sampler.sample(shots=8, return_errors=True)

    seen_priors: list[np.ndarray] = []
    orig = backend_adapter._rebuild_dem_with_priors

    def spy(dem_in, priors):
        seen_priors.append(np.asarray(priors, dtype=float).copy())
        return orig(dem_in, priors)

    monkeypatch.setattr(backend_adapter, "_rebuild_dem_with_priors", spy)

    priors_batch = np.linspace(0.001, 0.05, dem.num_errors, dtype=float)
    priors_batch = np.broadcast_to(priors_batch, (detections.shape[0], dem.num_errors)).copy()
    # Row-specific perturbation so we can verify each shot's priors are
    # routed through, not just the first row's.
    priors_batch = priors_batch * np.linspace(0.5, 1.5, detections.shape[0])[:, None]

    decode_with_predecoder(
        priors_batch,
        env,
        detections.astype(np.uint8),
        circuit,
        "soft_priors",
    )

    assert len(seen_priors) == detections.shape[0], (
        "reweighted MWPM must rebuild the DEM once per shot "
        f"(got {len(seen_priors)}/{detections.shape[0]})"
    )
    for row, priors in enumerate(seen_priors):
        np.testing.assert_allclose(priors, priors_batch[row], rtol=0, atol=0), (
            f"shot {row} received the wrong priors"
        )


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
