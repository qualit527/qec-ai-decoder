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


def test_mwpm_soft_priors_honor_predecoder_output() -> None:
    """Predecoder priors must actually change the decode when they move
    significantly away from the DEM's natives. Prior to 2026-04-24 the
    MWPM path silently ignored predecoder_output, giving delta_ler=0 by
    construction on every surface-code run.
    """
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    circuit = stim.Circuit.from_file(env.code.source)
    dem = circuit.detector_error_model(decompose_errors=True)
    n_err = dem.num_errors
    sampler = dem.compile_sampler(seed=0)
    detections, _observables, _errors = sampler.sample(shots=64, return_errors=True)

    # Flatten error probabilities across the board — this should alter
    # which path MWPM prefers relative to the DEM's natural weighting
    # (especially for shots with multiple candidate corrections).
    flat_priors = np.full((detections.shape[0], n_err), 1e-6, dtype=float)
    reweighted = decode_with_predecoder(
        flat_priors,
        env,
        detections.astype(np.uint8),
        circuit,
        "soft_priors",
    )

    matching = pymatching.Matching.from_detector_error_model(dem)
    plain = matching.decode_batch(detections.astype(bool))
    # The reweighting must have influenced at least one prediction.
    assert not np.array_equal(reweighted, plain), (
        "predecoder soft_priors were not honored by the MWPM backend — "
        "reweighted output matches plain baseline on every shot"
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
