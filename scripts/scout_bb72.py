"""Try three sources for the bb72 ([[72,12,6]]) parity-check matrix / Stim circuit:
1. pip `qLDPC` package
2. pip `stimbposd` package
3. Bravyi et al 2024 github (hand-clone)

Fallback: build H manually from the bivariate-bicycle construction (requires
some LOC; see DECODER_ROADMAP §3)."""
import importlib
import numpy as np
from pathlib import Path


def try_qldpc():
    try:
        qldpc = importlib.import_module("qldpc")
    except ImportError:
        return None
    # API surface varies; adjust if available.
    try:
        code = qldpc.codes.bivariate_bicycle(72, 12, 6)  # hypothetical; may need different call
        return code.parity_check_matrix()
    except Exception as e:
        print(f"[qldpc] failed: {e}")
        return None


def try_stimbposd():
    try:
        sbp = importlib.import_module("stimbposd")
        # No known helper; skip unless found.
        return None
    except ImportError:
        return None


def build_manual():
    """Bivariate-bicycle [[72,12,6]] construction (Bravyi et al 2024).
    n = 2 * 36. Stabilisers from two commuting matrices A = x^3 + y + y^2,
    B = y^3 + x + x^2 over F_2[x,y] / (x^6-1, y^6-1). See DECODER_ROADMAP §3."""
    l = 6
    m = 6
    def shift(size, k):
        I = np.eye(size, dtype=np.uint8)
        return np.roll(I, k, axis=1)
    Il, Im = np.eye(l, dtype=np.uint8), np.eye(m, dtype=np.uint8)
    x = np.kron(shift(l, 1), Im)
    y = np.kron(Il, shift(m, 1))
    A = (x @ x @ x + y + y @ y) % 2
    B = (y @ y @ y + x + x @ x) % 2
    Hx = np.hstack([A, B])
    Hz = np.hstack([B.T, A.T])
    return Hx, Hz


def main():
    out_dir = Path("circuits")
    out_dir.mkdir(exist_ok=True)
    for fn in (try_qldpc, try_stimbposd):
        H = fn()
        if H is not None:
            np.save(out_dir / "bb72_H.npy", H)
            print(f"Sourced from {fn.__name__}; shape={H.shape}")
            return
    Hx, Hz = build_manual()
    np.save(out_dir / "bb72_Hx.npy", Hx)
    np.save(out_dir / "bb72_Hz.npy", Hz)
    print(f"Manually built: Hx={Hx.shape}, Hz={Hz.shape}")


if __name__ == "__main__":
    main()
