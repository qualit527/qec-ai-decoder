"""Deep-unfolded BP over a Tanner graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from autoqec.decoders.modules.base import PredecoderBase
from autoqec.decoders.modules.mlp import make_scalar_head


class NeuralBP(PredecoderBase):
    def __init__(
        self,
        n_var: int,
        n_check: int,
        iterations: int,
        weight_sharing: str = "per_layer",
        damping: str = "learnable_scalar",
        attention_aug: bool = False,
        attention_heads: int = 1,
        output_mode: str = "soft_priors",
        head_type: str = "linear",
    ):
        super().__init__()
        self.output_mode = output_mode
        self.n_var = n_var
        self.n_check = n_check
        self.iterations = iterations
        self.weight_sharing = weight_sharing
        self.attention_aug = attention_aug
        self.attention_heads = attention_heads
        if damping == "fixed":
            self.register_buffer("damp", torch.tensor(0.9))
        elif damping == "learnable_scalar":
            self.damp = nn.Parameter(torch.tensor(0.9))
        else:
            self.damp = nn.Parameter(torch.full((iterations,), 0.9))
        if weight_sharing == "per_check":
            self.check_w = nn.Parameter(torch.ones(iterations, n_check))
        elif weight_sharing == "per_layer":
            self.check_w = nn.Parameter(torch.ones(iterations))
        else:
            self.register_buffer("check_w", torch.ones(iterations))
        self.head = make_scalar_head(head_type)

    def _damping_at(self, iteration: int) -> Tensor:
        if self.damp.dim() == 0:
            return self.damp
        return self.damp[iteration]

    def _check_weight_at(self, iteration: int, c_idx: Tensor) -> Tensor:
        if self.check_w.dim() == 1:
            return self.check_w[iteration]
        return self.check_w[iteration][c_idx]

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        if syndrome.dim() == 3:
            syndrome = syndrome.mean(dim=1)
        batch_size = syndrome.shape[0]
        edge_index = ctx["edge_index"].to(syndrome.device)
        parity_check = ctx.get("parity_check_matrix")
        prior_p = ctx.get("prior_p")
        if prior_p is None:
            prior_llr = torch.zeros(self.n_var, device=syndrome.device)
        else:
            clipped = torch.as_tensor(prior_p, dtype=syndrome.dtype, device=syndrome.device).clamp(
                1e-4, 1 - 1e-4
            )
            prior_llr = torch.log((1 - clipped) / clipped)
        v_idx, c_idx = edge_index[0], edge_index[1]
        edge_count = edge_index.shape[1]
        mu_v2c = prior_llr[v_idx].unsqueeze(0).expand(batch_size, edge_count).clone()

        for iteration in range(self.iterations):
            damp = self._damping_at(iteration)
            mu_c2v = torch.zeros_like(mu_v2c)
            check_weights = self._check_weight_at(iteration, c_idx)
            for check in range(self.n_check):
                edge_mask = c_idx == check
                if not edge_mask.any():
                    continue
                edge_ids = edge_mask.nonzero(as_tuple=False).flatten()
                sign = (1 - 2 * syndrome[:, check : check + 1]).to(mu_v2c.dtype)
                incoming = mu_v2c[:, edge_ids]
                tanh_vals = torch.tanh(incoming / 2).clamp(-0.999, 0.999)
                outgoing = torch.zeros_like(tanh_vals)
                for local_idx in range(edge_ids.numel()):
                    if edge_ids.numel() == 1:
                        product = torch.ones(batch_size, 1, device=syndrome.device, dtype=mu_v2c.dtype)
                    else:
                        other_mask = torch.ones(edge_ids.numel(), dtype=torch.bool, device=syndrome.device)
                        other_mask[local_idx] = False
                        product = tanh_vals[:, other_mask].prod(dim=1, keepdim=True)
                    outgoing[:, local_idx : local_idx + 1] = product
                mu_c2v[:, edge_ids] = sign * 2 * torch.atanh(outgoing.clamp(-0.999, 0.999))
            mu_c2v = mu_c2v * check_weights

            mu_v2c_new = torch.zeros_like(mu_v2c)
            for var in range(self.n_var):
                edge_mask = v_idx == var
                if not edge_mask.any():
                    continue
                edge_ids = edge_mask.nonzero(as_tuple=False).flatten()
                incoming = mu_c2v[:, edge_ids]
                total = prior_llr[var] + incoming.sum(dim=1, keepdim=True)
                mu_v2c_new[:, edge_ids] = total - incoming
            mu_v2c = damp * mu_v2c + (1 - damp) * mu_v2c_new

        logits = torch.zeros(batch_size, self.n_var, device=syndrome.device, dtype=syndrome.dtype)
        for var in range(self.n_var):
            edge_mask = v_idx == var
            if edge_mask.any():
                logits[:, var] = prior_llr[var] + mu_v2c[:, edge_mask].sum(dim=1)
            else:
                logits[:, var] = prior_llr[var]
        logits = self.head(logits.unsqueeze(-1)).squeeze(-1)
        probs = torch.sigmoid(-logits)
        if self.output_mode == "soft_priors":
            return probs

        # hard_flip: keep gradients alive by using the soft probabilities
        # straight through a differentiable parity projection. The
        # classical-backend adapter thresholds at 0.5 at eval time.
        if parity_check is not None:
            parity = torch.as_tensor(parity_check, dtype=probs.dtype, device=syndrome.device)
            # Soft "cleaned syndrome" — matmul of soft error probs with H^T.
            # Kept in [0, 1]-ish range via clamp; no hard Booleanization here.
            cleaned = (probs @ parity.T).clamp(0.0, 1.0)
            return cleaned
        # No parity matrix (stim path) — use the syndrome passthrough as a
        # differentiable weak baseline.
        return syndrome.float()
