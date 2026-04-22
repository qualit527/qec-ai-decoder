import pytest

from autoqec.decoders.dsl_schema import PredecoderDSL


def test_valid_gnn_config() -> None:
    cfg = PredecoderDSL(
        type="gnn",
        output_mode="soft_priors",
        gnn={
            "layers": 3,
            "hidden_dim": 32,
            "message_fn": "mlp",
            "aggregation": "sum",
            "normalization": "layer",
            "residual": True,
            "edge_features": ["syndrome_bit"],
        },
        head="linear",
        training={
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 3,
            "loss": "bce",
            "profile": "dev",
        },
    )
    assert cfg.type == "gnn"


def test_rejects_bad_hidden_dim() -> None:
    with pytest.raises(Exception):
        PredecoderDSL(
            type="gnn",
            output_mode="soft_priors",
            gnn={
                "layers": 1,
                "hidden_dim": -1,
                "message_fn": "mlp",
                "aggregation": "sum",
                "normalization": "layer",
                "residual": False,
                "edge_features": [],
            },
            head="linear",
            training={
                "learning_rate": 1e-3,
                "batch_size": 1,
                "epochs": 1,
                "loss": "bce",
                "profile": "dev",
            },
        )


def test_valid_neural_bp_config() -> None:
    cfg = PredecoderDSL(
        type="neural_bp",
        output_mode="soft_priors",
        neural_bp={
            "iterations": 5,
            "weight_sharing": "per_layer",
            "damping": "learnable_scalar",
            "attention_aug": False,
            "attention_heads": 1,
        },
        head="linear",
        training={
            "learning_rate": 5e-4,
            "batch_size": 32,
            "epochs": 3,
            "loss": "bce",
            "profile": "dev",
        },
    )
    assert cfg.neural_bp is not None
    assert cfg.neural_bp.iterations == 5

