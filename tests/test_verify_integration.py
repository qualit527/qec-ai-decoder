import pytest
import torch
from pathlib import Path


def test_verify_real_gnn_checkpoint(tmp_path):
    """End-to-end: compile GNN from seed template → save in Runner format →
    independent_verify loads it and runs the full pipeline including shuffle."""
    import yaml
    from autoqec.decoders.dsl_compiler import compile_predecoder
    from autoqec.envs.schema import load_env_yaml
    from autoqec.eval.independent_eval import independent_verify

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")

    # Load GNN config and compile model
    config = yaml.safe_load(Path("autoqec/example_db/gnn_small.yaml").read_text())
    # Need n_var and n_check from the circuit
    from autoqec.runner.data import load_code_artifacts
    artifacts = load_code_artifacts(env)
    model = compile_predecoder(config, artifacts.n_var, artifacts.n_check)

    # Save in Runner checkpoint format
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({
        "class_name": type(model).__name__,
        "state_dict": model.state_dict(),
        "output_mode": model.output_mode,
        "dsl_config": config,
    }, ckpt)

    # Run verify — should not crash on shuffle
    report = independent_verify(ckpt, env, holdout_seeds=list(range(9000, 9010)),
                                n_shots=3000, n_bootstrap=200)
    assert report.seed_leakage_check_ok is True
    assert report.verdict in ("VERIFIED", "SUSPICIOUS", "FAILED")
    # Untrained model should not be VERIFIED
    assert report.verdict != "VERIFIED" or report.delta_ler_holdout < 0.01
