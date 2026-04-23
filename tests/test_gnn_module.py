import pytest
import torch

from autoqec.decoders.modules.base import PredecoderBase
from autoqec.decoders.modules.gnn import (
    BipartiteGNN,
    _aggregate,
    _make_message_fn,
)
from autoqec.decoders.modules.mlp import GatedMLP, ResidualMLP, make_head, make_scalar_head


def test_gnn_forward_shape() -> None:
    n_var, n_check, hidden = 40, 24, 16
    model = BipartiteGNN(
        n_var=n_var,
        n_check=n_check,
        hidden_dim=hidden,
        layers=2,
        message_fn="mlp",
        aggregation="sum",
        normalization="layer",
        residual=True,
        output_mode="soft_priors",
    )
    syndrome = torch.rand(4, n_check)
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(n_var), torch.arange(n_check), indexing="ij"),
        dim=0,
    ).reshape(2, -1)
    out = model(syndrome, {"edge_index": edge_index, "n_var": n_var, "n_check": n_check})
    assert out.shape == (4, n_var)


def test_gnn_hard_flip_shape() -> None:
    model = BipartiteGNN(
        n_var=10,
        n_check=6,
        hidden_dim=8,
        layers=1,
        message_fn="mlp",
        aggregation="mean",
        normalization="none",
        residual=False,
        output_mode="hard_flip",
    )
    syndrome = torch.rand(2, 6)
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(10), torch.arange(6), indexing="ij"),
        dim=0,
    ).reshape(2, -1)
    out = model(syndrome, {"edge_index": edge_index, "n_var": 10, "n_check": 6})
    assert out.shape == (2, 6)


def test_make_message_fn_variants_and_helpers() -> None:
    x = torch.randn(3, 8)
    gated = _make_message_fn("gated_mlp", 4)
    residual = _make_message_fn("residual_mlp", 4)
    normalized = _make_message_fn("normalized_mlp", 4)
    plain = _make_message_fn("mlp", 4)

    assert gated(torch.randn(3, 8)).shape == (3, 4)
    assert residual(torch.randn(3, 8)).shape == (3, 4)
    assert normalized(torch.randn(3, 8)).shape == (3, 4)
    assert plain(torch.randn(3, 8)).shape == (3, 4)
    assert GatedMLP(8, 4)(x).shape == (3, 4)
    assert ResidualMLP(8, 4)(x).shape == (3, 4)
    assert make_head("mlp_small", 4, 2)(torch.randn(3, 4)).shape == (3, 2)
    assert make_head("linear", 4, 2)(torch.randn(3, 4)).shape == (3, 2)
    assert make_scalar_head("mlp_small")(torch.randn(3, 1)).shape == (3, 1)
    assert make_scalar_head("linear")(torch.randn(3, 1)).shape == (3, 1)


def test_aggregate_variants_and_errors() -> None:
    messages = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])

    assert torch.equal(_aggregate("sum", messages, index, 2), torch.tensor([[4.0, 6.0], [5.0, 6.0]]))
    assert torch.equal(_aggregate("mean", messages, index, 2), torch.tensor([[2.0, 3.0], [5.0, 6.0]]))
    assert torch.equal(_aggregate("max", messages, index, 2), torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(_aggregate("attention_pool", messages, index, 2), torch.tensor([[4.0, 6.0], [5.0, 6.0]]))
    with pytest.raises(ValueError, match="unsupported aggregation"):
        _aggregate("bogus", messages, index, 2)


def test_gnn_normalization_and_3d_forward_cover_extra_paths() -> None:
    model = BipartiteGNN(
        n_var=6,
        n_check=4,
        hidden_dim=4,
        layers=1,
        message_fn="normalized_mlp",
        aggregation="max",
        normalization="batch",
        residual=True,
        output_mode="soft_priors",
    )
    syndrome = torch.rand(2, 3, 4)
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(6), torch.arange(4), indexing="ij"),
        dim=0,
    ).reshape(2, -1)
    out = model(syndrome, {"edge_index": edge_index, "n_var": 6, "n_check": 4})
    assert out.shape == (2, 6)
    with pytest.raises(ValueError, match="unsupported normalization"):
        model._apply_norm("bogus", torch.randn(2, 4))


def test_predecoder_base_expected_output_shape_switches() -> None:
    base = PredecoderBase()
    assert base.expected_output_shape == "[batch, n_faults] float in [0, 1]"
    base.output_mode = "hard_flip"
    assert base.expected_output_shape == "[batch, n_checks] long"
