import numpy as np
import time


def test_bposd_small_parity_check():
    from autoqec.decoders.baselines.bposd_wrap import BpOsdBaseline
    # Toy: repetition code parity-check [1 1 0; 0 1 1]
    H = np.array([[1,1,0],[0,1,1]], dtype=np.uint8)
    dec = BpOsdBaseline(H, error_rate=0.05, osd_order=0)
    syndrome = np.array([[1, 0]], dtype=np.uint8)
    correction = dec.decode_batch(syndrome)
    assert correction.shape == (1, 3)
    # syndrome=(1,0) → one error at bit 0 or 1 (bit 0 is minimum-weight)
    assert correction[0, 0] + correction[0, 1] >= 1


def test_bposd_bb72_single_shot_latency_under_budget():
    from autoqec.decoders.baselines.bposd_wrap import BpOsdBaseline
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    parity_check = np.load(env.code.source).astype(np.uint8)
    dec = BpOsdBaseline(parity_check, error_rate=float(env.noise.p[0]), osd_order=0)
    syndrome = np.zeros((1, parity_check.shape[0]), dtype=np.uint8)

    started = time.perf_counter()
    correction = dec.decode_batch(syndrome)
    elapsed_s = time.perf_counter() - started

    assert correction.shape == (1, parity_check.shape[1])
    assert elapsed_s < 0.2
