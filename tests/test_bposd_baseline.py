import numpy as np
import pytest


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
