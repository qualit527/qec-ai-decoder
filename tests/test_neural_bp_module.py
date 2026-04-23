import torch

from autoqec.decoders.modules.neural_bp import NeuralBP


def test_neural_bp_forward_shape() -> None:
    n_var, n_check = 20, 12
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(n_var), torch.arange(n_check), indexing="ij"),
        dim=0,
    ).reshape(2, -1)
    model = NeuralBP(
        n_var=n_var,
        n_check=n_check,
        iterations=3,
        weight_sharing="per_layer",
        damping="learnable_scalar",
        output_mode="soft_priors",
    )
    syndrome = torch.rand(4, n_check)
    out = model(syndrome, {"edge_index": edge_index, "n_var": n_var, "n_check": n_check})
    assert out.shape == (4, n_var)
    assert torch.all(out >= 0)
    assert torch.all(out <= 1)


def test_neural_bp_covers_fixed_damping_per_check_and_hard_flip_paths() -> None:
    n_var, n_check = 6, 4
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(n_var), torch.arange(n_check), indexing="ij"),
        dim=0,
    ).reshape(2, -1)
    parity = torch.randint(0, 2, (n_check, n_var))
    syndrome = torch.rand(3, 2, n_check)

    model = NeuralBP(
        n_var=n_var,
        n_check=n_check,
        iterations=2,
        weight_sharing="per_check",
        damping="fixed",
        output_mode="hard_flip",
    )
    out = model(
        syndrome,
        {
            "edge_index": edge_index,
            "n_var": n_var,
            "n_check": n_check,
            "parity_check_matrix": parity,
            "prior_p": None,
        },
    )
    assert out.shape == (3, n_check)
    assert model._damping_at(0).dim() == 0
    assert model._check_weight_at(0, edge_index[1]).shape[0] == edge_index.shape[1]


def test_neural_bp_covers_learnable_vector_and_no_parity_fallback() -> None:
    n_var, n_check = 5, 3
    edge_index = torch.stack(
        torch.meshgrid(torch.arange(n_var), torch.arange(n_check), indexing="ij"),
        dim=0,
    ).reshape(2, -1)
    syndrome = torch.rand(2, n_check)

    model = NeuralBP(
        n_var=n_var,
        n_check=n_check,
        iterations=3,
        weight_sharing="none",
        damping="learnable_vector",
        output_mode="hard_flip",
    )
    out = model(
        syndrome,
        {
            "edge_index": edge_index,
            "n_var": n_var,
            "n_check": n_check,
            "prior_p": torch.full((n_var,), 0.2),
        },
    )
    assert out.shape == (2, n_check)
    assert model._damping_at(1).shape == torch.Size([])
