from __future__ import annotations

from autoqec.decoders.dsl_schema import PredecoderDSL
from autoqec.decoders.modules.gnn import BipartiteGNN
from autoqec.decoders.modules.neural_bp import NeuralBP


def compile_predecoder(config_dict: dict, n_var: int, n_check: int):
    dsl = PredecoderDSL(**config_dict)
    head_type = dsl.head if isinstance(dsl.head, str) else "linear"
    if dsl.type == "gnn":
        gnn = dsl.gnn
        assert gnn is not None
        message_fn = gnn.message_fn if isinstance(gnn.message_fn, str) else "mlp"
        aggregation = gnn.aggregation if isinstance(gnn.aggregation, str) else "sum"
        return BipartiteGNN(
            n_var=n_var,
            n_check=n_check,
            hidden_dim=gnn.hidden_dim,
            layers=gnn.layers,
            message_fn=message_fn,
            aggregation=aggregation,
            normalization=gnn.normalization,
            residual=gnn.residual,
            output_mode=dsl.output_mode,
            head_type=head_type,
        )
    neural_bp = dsl.neural_bp
    assert neural_bp is not None
    return NeuralBP(
        n_var=n_var,
        n_check=n_check,
        iterations=neural_bp.iterations,
        weight_sharing=neural_bp.weight_sharing,
        damping=neural_bp.damping,
        attention_aug=neural_bp.attention_aug,
        attention_heads=neural_bp.attention_heads,
        output_mode=dsl.output_mode,
        head_type=head_type,
    )

