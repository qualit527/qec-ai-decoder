import torch

from autoqec.decoders.modules.gnn import BipartiteGNN


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

