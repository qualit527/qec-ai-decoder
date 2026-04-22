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

