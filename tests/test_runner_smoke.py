import pytest

from autoqec.envs.schema import load_env_yaml
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig


@pytest.mark.integration
def test_runner_end_to_end(tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "soft_priors",
            "gnn": {
                "layers": 1,
                "hidden_dim": 8,
                "message_fn": "mlp",
                "aggregation": "sum",
                "normalization": "none",
                "residual": False,
                "edge_features": [],
            },
            "head": "linear",
            "training": {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 1,
                "loss": "bce",
                "profile": "dev",
            },
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_0"),
    )
    metrics = run_round(cfg, env)
    assert metrics.status == "ok"
