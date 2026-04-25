from __future__ import annotations

import torch
from torch import Tensor, nn

from autoqec.decoders.modules.base import PredecoderBase
from autoqec.decoders.modules.mlp import GatedMLP, ResidualMLP, make_head


def _make_message_fn(name: str, hidden_dim: int) -> nn.Module:
    input_dim = hidden_dim * 2
    if name == "gated_mlp":
        return GatedMLP(input_dim, hidden_dim)
    if name == "residual_mlp":
        return ResidualMLP(input_dim, hidden_dim)
    if name == "normalized_mlp":
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _aggregate(agg: str, messages: Tensor, index: Tensor, n_targets: int) -> Tensor:
    hidden_dim = messages.shape[-1]
    out = torch.zeros(n_targets, hidden_dim, device=messages.device, dtype=messages.dtype)
    if agg in {"sum", "mean", "attention_pool", "set_transformer", "gated_sum"}:
        out.index_add_(0, index, messages)
        if agg == "mean":
            counts = torch.zeros(n_targets, device=messages.device, dtype=messages.dtype)
            counts.index_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
            out = out / counts.clamp(min=1).unsqueeze(-1)
        return out
    if agg == "max":
        out.fill_(float("-inf"))
        for target in range(n_targets):
            mask = index == target
            if mask.any():
                out[target] = messages[mask].max(dim=0).values
        out[out == float("-inf")] = 0
        return out
    raise ValueError(f"unsupported aggregation: {agg}")


def _zero_init_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


class BipartiteGNN(PredecoderBase):
    def __init__(
        self,
        n_var: int,
        n_check: int,
        hidden_dim: int,
        layers: int,
        message_fn: str,
        aggregation: str,
        normalization: str,
        residual: bool,
        output_mode: str,
        head_type: str = "linear",
    ):
        super().__init__()
        self.output_mode = output_mode
        self.n_var = n_var
        self.n_check = n_check
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        self.residual = residual
        self.var_embed = nn.Embedding(n_var, hidden_dim)
        self.check_embed = nn.Linear(1, hidden_dim)
        self.layers_v2c = nn.ModuleList(_make_message_fn(message_fn, hidden_dim) for _ in range(layers))
        self.layers_c2v = nn.ModuleList(_make_message_fn(message_fn, hidden_dim) for _ in range(layers))
        self.norm_name = normalization
        if output_mode == "soft_priors":
            self.head = make_head(head_type, hidden_dim, 1)
            _zero_init_last_linear(self.head)
        else:
            self.head = make_head(head_type, hidden_dim, 1)

    def _apply_norm(self, name: str, tensor: Tensor) -> Tensor:
        if name == "none":
            return tensor
        if name in {"layer", "edge_norm", "graph_norm"}:
            return nn.functional.layer_norm(tensor, (tensor.shape[-1],))
        if name == "batch":
            flat = tensor.reshape(-1, tensor.shape[-1])
            mean = flat.mean(dim=0, keepdim=True)
            var = flat.var(dim=0, keepdim=True, unbiased=False)
            normed = (flat - mean) / torch.sqrt(var + 1e-5)
            return normed.reshape_as(tensor)
        raise ValueError(f"unsupported normalization: {name}")

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        if syndrome.dim() == 3:
            syndrome = syndrome.mean(dim=1)
        batch_size = syndrome.shape[0]
        edge_index = ctx["edge_index"].to(syndrome.device)
        n_var = int(ctx.get("n_var", self.n_var))
        n_check = int(ctx.get("n_check", self.n_check))
        var_ids = torch.arange(n_var, device=syndrome.device)
        h_v = self.var_embed(var_ids).unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        h_c = self.check_embed(syndrome.unsqueeze(-1))
        v_idx, c_idx = edge_index[0], edge_index[1]

        for v2c, c2v in zip(self.layers_v2c, self.layers_c2v):
            v_to_c = v2c(torch.cat([h_v[:, v_idx], h_c[:, c_idx]], dim=-1))
            h_c_new = torch.stack(
                [_aggregate(self.aggregation, v_to_c[b], c_idx, n_check) for b in range(batch_size)],
                dim=0,
            )
            h_c = h_c + h_c_new if self.residual else h_c_new
            h_c = self._apply_norm(self.norm_name, h_c)

            c_to_v = c2v(torch.cat([h_c[:, c_idx], h_v[:, v_idx]], dim=-1))
            h_v_new = torch.stack(
                [_aggregate(self.aggregation, c_to_v[b], v_idx, n_var) for b in range(batch_size)],
                dim=0,
            )
            h_v = h_v + h_v_new if self.residual else h_v_new
            h_v = self._apply_norm(self.norm_name, h_v)

        if self.output_mode == "soft_priors":
            logits = self.head(h_v).squeeze(-1)
            prior_p = ctx.get("prior_p")
            if prior_p is not None:
                prior = torch.as_tensor(
                    prior_p,
                    dtype=logits.dtype,
                    device=logits.device,
                ).clamp(1e-6, 1 - 1e-6)
                logits = logits + torch.logit(prior).unsqueeze(0)
            return torch.sigmoid(logits)

        # hard_flip: return soft sigmoid probabilities so gradients flow
        # through training. The hard threshold at 0.5 is applied by the
        # classical-backend adapter at eval time (see
        # autoqec.decoders.backend_adapter.decode_with_predecoder).
        logits = self.head(h_c).squeeze(-1)
        return torch.sigmoid(logits)
