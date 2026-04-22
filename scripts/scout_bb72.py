"""Build a minimal bb72 parity-check artifact for the MVP qLDPC smoke path."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def build_manual() -> tuple[np.ndarray, np.ndarray]:
    """Manual bivariate bicycle construction from the owner plan."""
    lattice_x = 6
    lattice_y = 6

    def shift(size: int, k: int) -> np.ndarray:
        eye = np.eye(size, dtype=np.uint8)
        return np.roll(eye, k, axis=1)

    il = np.eye(lattice_x, dtype=np.uint8)
    im = np.eye(lattice_y, dtype=np.uint8)
    x = np.kron(shift(lattice_x, 1), im)
    y = np.kron(il, shift(lattice_y, 1))
    a = (x @ x @ x + y + y @ y) % 2
    b = (y @ y @ y + x + x @ x) % 2
    hx = np.hstack([a, b]).astype(np.uint8)
    hz = np.hstack([b.T, a.T]).astype(np.uint8)
    return hx, hz


if __name__ == "__main__":
    out_dir = Path("circuits")
    out_dir.mkdir(parents=True, exist_ok=True)
    hx, hz = build_manual()
    np.save(out_dir / "bb72_Hx.npy", hx)
    np.save(out_dir / "bb72_Hz.npy", hz)
    print(f"Manually built: Hx={hx.shape}, Hz={hz.shape}")
