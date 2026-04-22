import torch
import numpy as np
from pathlib import Path


def test_memorizer_fails_independent_verify(tmp_path):
    from autoqec.cheaters.memorize import MemorizerPredecoder, save_memorizer_ckpt
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    ckpt = tmp_path / "memorizer.pt"
    save_memorizer_ckpt(ckpt)
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9020)),
                                n_shots=5000, n_bootstrap=200)
    # Memorizer sees unseen holdout → behaves no better than baseline
    # Verdict must NOT be VERIFIED
    assert report.verdict in ("SUSPICIOUS", "FAILED"), f"got {report.verdict}"
