import torch

from autoqec.decoders.dsl_compiler import compile_predecoder


def test_compile_gnn_config() -> None:
    cfg = {
        "type": "gnn",
        "output_mode": "soft_priors",
        "gnn": {
            "layers": 2,
            "hidden_dim": 16,
            "message_fn": "mlp",
            "aggregation": "sum",
            "normalization": "layer",
            "residual": True,
            "edge_features": ["syndrome_bit"],
        },
        "head": "linear",
        "training": {
            "learning_rate": 1e-3,
            "batch_size": 4,
            "epochs": 1,
            "loss": "bce",
            "profile": "dev",
        },
    }
    model = compile_predecoder(cfg, n_var=20, n_check=12)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_compile_neural_bp_config() -> None:
    cfg = {
        "type": "neural_bp",
        "output_mode": "soft_priors",
        "neural_bp": {
            "iterations": 3,
            "weight_sharing": "per_layer",
            "damping": "learnable_scalar",
            "attention_aug": False,
            "attention_heads": 1,
        },
        "head": "linear",
        "training": {
            "learning_rate": 5e-4,
            "batch_size": 4,
            "epochs": 1,
            "loss": "bce",
            "profile": "dev",
        },
    }
    model = compile_predecoder(cfg, n_var=20, n_check=12)
    assert isinstance(model, torch.nn.Module)

