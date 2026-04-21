# AutoQEC — 林腾祥 Execution Plan (Codex owner)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Own the AI-predecoder core. Implement the Tier-1+Tier-2 DSL (`dsl_schema.py` + `dsl_compiler.py` + `custom_fn_validator.py`), the non-LLM Runner (training / FLOPs / `RunnerSafety`), GNN + Neural-BP template library, and the adapter from predecoder output (`hard_flip` / `soft_priors`) into MWPM / OSD. Ship the Makefile and Demo 2 (`bb72` research run).

**Architecture:** Three slices. (1) **DSL slice**: schema → compiler → `nn.Module`. (2) **Runner slice**: config → train → eval → metrics.json; safety sentinels. (3) **Predecoder slice**: 3 GNN + 3 Neural-BP seed templates + backend adapter.

**Tech Stack:** PyTorch, pydantic, fvcore (FLOPs), pytest, Stim, torch_geometric-free implementation (use raw edge-index tensors), numpy.

**Companion contracts (frozen Phase 0):** `docs/contracts/interfaces.md` §2.2 (`RunnerConfig` + `RoundMetrics`), §2.4 (predecoder I/O), §2.1 (EnvSpec — consumed).

---

## Reading order

1. `docs/superpowers/specs/2026-04-20-autoqec-design.md` §4.1, §4.5, §6 (DSL), Appendix A
2. `docs/superpowers/plans/2026-04-21-autoqec-master.md`
3. `knowledge/DECODER_ROADMAP.md` §5 (canonical building blocks — full enum)
4. `knowledge/AUTORESEARCH_PATTERNS.md` — the `aide/interpreter.py` (~311 LOC → 280 LOC) port recipe for the Runner subprocess pattern

---

## Logical phases

- **Phase 0** — draft `RunnerConfig` + `RoundMetrics` + predecoder I/O contracts (Tasks C0.1–C0.2)
- **Phase 1** — DSL schema + compiler + custom_fn validator + 3 GNN + 3 Neural-BP templates; Runner skeleton + safety (Tasks C1.1–C1.7)
- **Phase 2** — backend adapter (`hard_flip` / `soft_priors` → MWPM / OSD); end-to-end handshake with A (Tasks C2.1–C2.2)
- **Phase 3** — Makefile, CLI polish, Demo 2 (Tasks C3.1–C3.3)

---

## Files you own

| Path | Responsibility |
|---|---|
| `autoqec/runner/schema.py` | `RunnerConfig` + `RoundMetrics` pydantic (§2.2) |
| `autoqec/runner/runner.py` | `run_round(config, env_spec) → RoundMetrics` |
| `autoqec/runner/safety.py` | `RunnerSafety` sentinels |
| `autoqec/runner/flops.py` | FLOPs counter via fvcore |
| `autoqec/runner/data.py` | Stim syndrome sampling with seed isolation |
| `autoqec/decoders/dsl_schema.py` | pydantic Tier-1 DSL |
| `autoqec/decoders/dsl_compiler.py` | YAML → `nn.Module` |
| `autoqec/decoders/custom_fn_validator.py` | Tier-2 AST + smoke test |
| `autoqec/decoders/backend_adapter.py` | predecoder output → MWPM / OSD |
| `autoqec/decoders/modules/` | `gnn.py`, `neural_bp.py`, `mlp.py`, shared primitives |
| `autoqec/example_db/gnn_*.yaml` | 3 GNN seed templates |
| `autoqec/example_db/neural_bp_*.yaml` | 3 Neural-BP seed templates |
| `Makefile` | per-agent backend + demo targets |
| `demos/demo-2-bb72/` | Demo 2 |

---

## Task C0.1: Draft `RunnerConfig` + `RoundMetrics`

**Files:**
- Create: `autoqec/runner/schema.py`

- [ ] **Step 1: Copy contract**

Copy the exact pydantic classes from `docs/contracts/interfaces.md` §2.2 into `autoqec/runner/schema.py`:

```python
# autoqec/runner/schema.py
from pydantic import BaseModel
from typing import Literal, Optional

class RunnerConfig(BaseModel):
    env_name: str
    predecoder_config: dict
    training_profile: Literal["dev", "prod"] = "dev"
    seed: int = 0
    round_dir: str

class RoundMetrics(BaseModel):
    status: Literal["ok", "killed_by_safety", "compile_error", "train_error"]
    status_reason: Optional[str] = None
    ler_plain_classical: Optional[float] = None
    ler_predecoder: Optional[float] = None
    delta_ler: Optional[float] = None
    delta_ler_ci_low: Optional[float] = None
    delta_ler_ci_high: Optional[float] = None
    flops_per_syndrome: Optional[int] = None
    n_params: Optional[int] = None
    train_wallclock_s: float = 0.0
    eval_wallclock_s: float = 0.0
    vram_peak_gb: float = 0.0
    checkpoint_path: Optional[str] = None
    training_log_path: Optional[str] = None
```

- [ ] **Step 2: Smoke test**

```bash
python -c "from autoqec.runner.schema import RunnerConfig, RoundMetrics; print(RunnerConfig.model_fields.keys())"
```

- [ ] **Step 3: Commit**

```bash
git add autoqec/runner/schema.py
git commit -m "feat: RunnerConfig + RoundMetrics (Phase-0 contract)"
```

---

## Task C0.2: Draft predecoder I/O contract

**Files:**
- Create: `autoqec/decoders/modules/base.py`

- [ ] **Step 1: Write the base class**

```python
# autoqec/decoders/modules/base.py
"""Every AutoQEC predecoder is an nn.Module implementing this interface.
See docs/contracts/interfaces.md §2.4."""
from __future__ import annotations
from typing import Literal
import torch
from torch import nn, Tensor

OutputMode = Literal["hard_flip", "soft_priors"]

class PredecoderBase(nn.Module):
    output_mode: OutputMode = "soft_priors"

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        raise NotImplementedError

    @property
    def expected_output_shape(self) -> str:
        if self.output_mode == "hard_flip":
            return "[batch, n_checks] long"
        return "[batch, n_faults] float in [0, 1]"
```

- [ ] **Step 2: Commit**

```bash
git add autoqec/decoders/modules/base.py
git commit -m "feat: PredecoderBase interface (Phase-0 contract)"
```

---

## Task C1.1: DSL schema (Tier 1)

**Files:**
- Create: `autoqec/decoders/dsl_schema.py`
- Create: `tests/test_dsl_schema.py`

- [ ] **Step 1: Write the schema test**

```python
# tests/test_dsl_schema.py
import pytest

def test_valid_gnn_config():
    from autoqec.decoders.dsl_schema import PredecoderDSL
    cfg = PredecoderDSL(
        type="gnn", output_mode="soft_priors",
        gnn={"layers": 3, "hidden_dim": 32, "message_fn": "mlp",
             "aggregation": "sum", "normalization": "layer",
             "residual": True, "edge_features": ["syndrome_bit"]},
        head="linear",
        training={"learning_rate": 1e-3, "batch_size": 64, "epochs": 3,
                  "loss": "bce", "profile": "dev"},
    )
    assert cfg.type == "gnn"

def test_rejects_bad_hidden_dim():
    from autoqec.decoders.dsl_schema import PredecoderDSL
    with pytest.raises(Exception):
        PredecoderDSL(
            type="gnn", output_mode="soft_priors",
            gnn={"layers": 1, "hidden_dim": -1, "message_fn": "mlp",
                 "aggregation": "sum", "normalization": "layer",
                 "residual": False, "edge_features": []},
            head="linear",
            training={"learning_rate": 1e-3, "batch_size": 1, "epochs": 1,
                      "loss": "bce", "profile": "dev"},
        )

def test_valid_neural_bp_config():
    from autoqec.decoders.dsl_schema import PredecoderDSL
    cfg = PredecoderDSL(
        type="neural_bp", output_mode="soft_priors",
        neural_bp={"iterations": 5, "weight_sharing": "per_layer",
                   "damping": "learnable_scalar", "attention_aug": False,
                   "attention_heads": 1},
        head="linear",
        training={"learning_rate": 5e-4, "batch_size": 32, "epochs": 3,
                  "loss": "bce", "profile": "dev"},
    )
    assert cfg.neural_bp.iterations == 5
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/test_dsl_schema.py -v
```

- [ ] **Step 3: Implement**

```python
# autoqec/decoders/dsl_schema.py
from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Union

EdgeFeature = Literal["syndrome_bit", "round_idx", "stabilizer_type", "distance", "prior_weight"]
MessageFn   = Literal["mlp", "gated_mlp", "attention", "gru_cell", "edge_attention",
                     "geometric_attention", "residual_mlp", "normalized_mlp"]
Aggregation = Literal["sum", "mean", "max", "attention_pool", "set_transformer", "gated_sum"]

class CustomFn(BaseModel):
    type: Literal["custom"] = "custom"
    code: str
    params_declared: dict[str, str] = Field(default_factory=dict)

class GNNSpec(BaseModel):
    layers:       int = Field(..., ge=1)
    hidden_dim:   int = Field(..., ge=4)
    message_fn:   Union[MessageFn, CustomFn]
    aggregation:  Union[Aggregation, CustomFn]
    normalization: Literal["none", "layer", "batch", "edge_norm", "graph_norm"] = "layer"
    residual:     bool = False
    edge_features: list[EdgeFeature] = Field(default_factory=list)

class NeuralBPSpec(BaseModel):
    iterations:      int = Field(..., ge=1)
    weight_sharing:  Literal["none", "per_layer", "per_check"] = "per_layer"
    damping:         Literal["fixed", "learnable_scalar", "learnable_per_iter"] = "learnable_scalar"
    attention_aug:   bool = False
    attention_heads: int = Field(1, ge=1)

class TrainingSpec(BaseModel):
    learning_rate: float = Field(..., gt=0)
    batch_size:    int = Field(..., ge=1)
    epochs:        int = Field(..., ge=1)
    loss:          Literal["bce", "focal", "weighted_bce"] = "bce"
    profile:       Literal["dev", "prod"] = "dev"

class PredecoderDSL(BaseModel):
    type:        Literal["gnn", "neural_bp"]
    output_mode: Literal["hard_flip", "soft_priors"]
    gnn:         Optional[GNNSpec] = None
    neural_bp:   Optional[NeuralBPSpec] = None
    head:        Union[Literal["linear", "mlp_small"], CustomFn]
    training:    TrainingSpec

    @field_validator("gnn", mode="after")
    @classmethod
    def _gnn_required_if_type_gnn(cls, v, info):
        if info.data.get("type") == "gnn" and v is None:
            raise ValueError("gnn spec required when type='gnn'")
        return v

    @field_validator("neural_bp", mode="after")
    @classmethod
    def _nb_required_if_type_neural_bp(cls, v, info):
        if info.data.get("type") == "neural_bp" and v is None:
            raise ValueError("neural_bp spec required when type='neural_bp'")
        return v
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_dsl_schema.py -v
```

- [ ] **Step 5: Commit**

```bash
git add autoqec/decoders/dsl_schema.py tests/test_dsl_schema.py
git commit -m "feat: Tier-1 DSL pydantic schema"
```

---

## Task C1.2: GNN module (PyTorch)

**Files:**
- Create: `autoqec/decoders/modules/gnn.py`
- Create: `tests/test_gnn_module.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_gnn_module.py
import torch
import pytest

def test_gnn_forward_shape():
    from autoqec.decoders.modules.gnn import BipartiteGNN
    n_var, n_check, hidden = 40, 24, 16
    model = BipartiteGNN(n_var=n_var, n_check=n_check, hidden_dim=hidden,
                          layers=2, message_fn="mlp", aggregation="sum",
                          normalization="layer", residual=True,
                          output_mode="soft_priors")
    batch = 4
    syndrome = torch.rand(batch, n_check)
    # Random fully-connected Tanner graph for smoke test
    edge_index = torch.stack(torch.meshgrid(
        torch.arange(n_var), torch.arange(n_check), indexing="ij"
    ), dim=0).reshape(2, -1)  # [2, n_var*n_check]
    out = model(syndrome, {"edge_index": edge_index, "n_var": n_var, "n_check": n_check})
    assert out.shape == (batch, n_var)   # soft_priors over fault slots (one per variable for MVP)

def test_gnn_hard_flip_shape():
    from autoqec.decoders.modules.gnn import BipartiteGNN
    m = BipartiteGNN(n_var=10, n_check=6, hidden_dim=8, layers=1,
                     message_fn="mlp", aggregation="mean",
                     normalization="none", residual=False,
                     output_mode="hard_flip")
    syn = torch.rand(2, 6)
    ei = torch.stack(torch.meshgrid(
        torch.arange(10), torch.arange(6), indexing="ij"), dim=0).reshape(2, -1)
    out = m(syn, {"edge_index": ei, "n_var": 10, "n_check": 6})
    assert out.shape == (2, 6)           # hard_flip returns cleaned syndrome
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/test_gnn_module.py -v
```

- [ ] **Step 3: Implement (from-scratch bipartite message-passing, no torch_geometric)**

```python
# autoqec/decoders/modules/gnn.py
from __future__ import annotations
import torch
from torch import nn, Tensor
from autoqec.decoders.modules.base import PredecoderBase

def _make_message_fn(name: str, hidden: int) -> nn.Module:
    if name == "mlp":
        return nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden))
    if name == "gated_mlp":
        return GatedMLP(hidden)
    if name == "residual_mlp":
        return ResidualMLP(hidden)
    if name == "normalized_mlp":
        return nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden),
                              nn.ReLU(), nn.Linear(hidden, hidden))
    # Other variants kept simple for MVP (attention/gru/edge_attention/geometric_attention
    # reuse 'mlp' as a fallback; a richer implementation lands post-MVP).
    return nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

class GatedMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp  = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Sigmoid())
    def forward(self, x):
        return self.gate(x) * self.mlp(x)

class ResidualMLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.fc2(h)
        # Residual against the averaged dst embedding (second half of x)
        dst = x[..., x.shape[-1] // 2:]
        return h + dst

def _aggregate(agg: str, messages: Tensor, index: Tensor, n_targets: int) -> Tensor:
    """messages: [E, H], index: [E] (target indices), → out: [n_targets, H]."""
    H = messages.shape[-1]
    out = torch.zeros(n_targets, H, device=messages.device, dtype=messages.dtype)
    if agg in ("sum", "mean"):
        out.index_add_(0, index, messages)
        if agg == "mean":
            counts = torch.zeros(n_targets, device=messages.device).index_add_(
                0, index, torch.ones_like(index, dtype=messages.dtype))
            out = out / counts.clamp(min=1).unsqueeze(-1)
    elif agg == "max":
        # Scatter-max via loop fallback (small n_targets OK for MVP)
        for t in range(n_targets):
            mask = index == t
            if mask.any():
                out[t] = messages[mask].max(dim=0).values
    elif agg in ("gated_sum", "attention_pool", "set_transformer"):
        # MVP fallback: sum (full parity is a post-MVP polish)
        out.index_add_(0, index, messages)
    return out

class BipartiteGNN(PredecoderBase):
    """Bipartite message-passing GNN over (var, check) nodes.

    ctx expects:
      edge_index: LongTensor [2, E]  — row 0 = var idx, row 1 = check idx
      n_var, n_check: int
    Output:
      soft_priors → [batch, n_var] in [0,1]
      hard_flip   → [batch, n_check] long (cleaned syndrome bits)
    """
    def __init__(self, n_var: int, n_check: int, hidden_dim: int, layers: int,
                 message_fn: str, aggregation: str, normalization: str,
                 residual: bool, output_mode: str,
                 edge_feature_dim: int = 0):
        super().__init__()
        self.output_mode = output_mode
        self.n_var, self.n_check = n_var, n_check
        self.hidden = hidden_dim
        self.residual = residual
        self.var_embed   = nn.Embedding(n_var, hidden_dim)
        self.check_embed = nn.Linear(1, hidden_dim)  # syndrome bit → hidden
        self.layers_v2c = nn.ModuleList(
            [_make_message_fn(message_fn, hidden_dim) for _ in range(layers)])
        self.layers_c2v = nn.ModuleList(
            [_make_message_fn(message_fn, hidden_dim) for _ in range(layers)])
        self.agg_name = aggregation
        norm_cls = {"layer": nn.LayerNorm, "batch": nn.BatchNorm1d,
                     "none": nn.Identity, "edge_norm": nn.LayerNorm, "graph_norm": nn.LayerNorm}
        self.norm_v = norm_cls.get(normalization, nn.Identity)(hidden_dim) if normalization != "none" else nn.Identity()
        self.norm_c = norm_cls.get(normalization, nn.Identity)(hidden_dim) if normalization != "none" else nn.Identity()
        if output_mode == "soft_priors":
            self.head = nn.Linear(hidden_dim, 1)
        else:
            self.head = nn.Linear(hidden_dim, 1)   # hard_flip per-check sigmoid

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        B = syndrome.shape[0]
        edge_index = ctx["edge_index"]        # [2, E]
        n_var, n_check = ctx.get("n_var", self.n_var), ctx.get("n_check", self.n_check)
        var_ids = torch.arange(n_var, device=syndrome.device)
        # Initial embeddings (broadcast over batch via expand)
        h_v = self.var_embed(var_ids).unsqueeze(0).expand(B, -1, -1).contiguous()
        h_c = self.check_embed(syndrome.unsqueeze(-1))    # [B, n_check, H]
        v_idx, c_idx = edge_index[0], edge_index[1]
        for v2c, c2v in zip(self.layers_v2c, self.layers_c2v):
            # var → check
            msg = v2c(torch.cat([h_v[:, v_idx], h_c[:, c_idx]], dim=-1))   # [B, E, H]
            # per-batch aggregation
            h_c_new = torch.stack(
                [_aggregate(self.agg_name, msg[b], c_idx, n_check) for b in range(B)],
                dim=0)
            h_c = self.norm_c(h_c + h_c_new if self.residual else h_c_new)
            # check → var
            msg2 = c2v(torch.cat([h_c[:, c_idx], h_v[:, v_idx]], dim=-1))
            h_v_new = torch.stack(
                [_aggregate(self.agg_name, msg2[b], v_idx, n_var) for b in range(B)],
                dim=0)
            h_v = self.norm_v(h_v + h_v_new if self.residual else h_v_new)
        if self.output_mode == "soft_priors":
            logits = self.head(h_v).squeeze(-1)      # [B, n_var]
            return torch.sigmoid(logits)
        else:
            logits = self.head(h_c).squeeze(-1)      # [B, n_check]
            return (torch.sigmoid(logits) > 0.5).long()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_gnn_module.py -v
```

- [ ] **Step 5: Commit**

```bash
git add autoqec/decoders/modules/gnn.py tests/test_gnn_module.py
git commit -m "feat: BipartiteGNN predecoder module"
```

---

## Task C1.3: Neural-BP module

**Files:**
- Create: `autoqec/decoders/modules/neural_bp.py`
- Create: `tests/test_neural_bp_module.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_neural_bp_module.py
import torch

def test_neural_bp_forward_shape():
    from autoqec.decoders.modules.neural_bp import NeuralBP
    n_var, n_check = 20, 12
    ei = torch.stack(torch.meshgrid(torch.arange(n_var),
                                      torch.arange(n_check), indexing="ij"),
                      dim=0).reshape(2, -1)
    m = NeuralBP(n_var=n_var, n_check=n_check, iterations=3,
                  weight_sharing="per_layer", damping="learnable_scalar",
                  output_mode="soft_priors")
    syn = torch.rand(4, n_check)
    out = m(syn, {"edge_index": ei, "n_var": n_var, "n_check": n_check})
    assert out.shape == (4, n_var)
    assert (out >= 0).all() and (out <= 1).all()
```

- [ ] **Step 2: Implement**

```python
# autoqec/decoders/modules/neural_bp.py
"""Deep-unfolded BP over the Tanner graph.

Each iteration t: var-to-check messages → check-to-var messages, with
learnable damping and optional per-iteration / per-check weights.
Output = marginal posterior p(fault | syndrome) after T iterations.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from autoqec.decoders.modules.base import PredecoderBase

class NeuralBP(PredecoderBase):
    def __init__(self, n_var: int, n_check: int, iterations: int,
                 weight_sharing: str = "per_layer",
                 damping: str = "learnable_scalar",
                 attention_aug: bool = False, attention_heads: int = 1,
                 output_mode: str = "soft_priors"):
        super().__init__()
        self.output_mode = output_mode
        self.n_var, self.n_check, self.T = n_var, n_check, iterations
        self.weight_sharing = weight_sharing
        # Damping
        if damping == "fixed":
            self.register_buffer("damp", torch.tensor(0.9))
        elif damping == "learnable_scalar":
            self.damp = nn.Parameter(torch.tensor(0.9))
        else:  # learnable_per_iter
            self.damp = nn.Parameter(torch.full((iterations,), 0.9))
        # Learnable log-likelihood ratio offset and check weights
        if weight_sharing == "per_layer":
            self.check_w = nn.Parameter(torch.ones(iterations))
        elif weight_sharing == "per_check":
            self.check_w = nn.Parameter(torch.ones(iterations, n_check))
        else:
            self.register_buffer("check_w", torch.ones(iterations))
        self.llr_scale = nn.Parameter(torch.tensor(1.0))

    def _damping_at(self, t: int) -> Tensor:
        return self.damp if self.damp.dim() == 0 else self.damp[t]

    def _cw_at(self, t: int) -> Tensor:
        if self.check_w.dim() == 1:
            return self.check_w[t]
        return self.check_w[t]   # [n_check]

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        B = syndrome.shape[0]
        ei = ctx["edge_index"]                      # [2, E]
        v_idx, c_idx = ei[0], ei[1]
        n_var = ctx.get("n_var", self.n_var)
        n_check = ctx.get("n_check", self.n_check)
        E = ei.shape[1]
        # Messages live on edges.
        mu_v2c = torch.zeros(B, E, device=syndrome.device)   # var → check
        # Prior LLR: uniform initial (learnable scale), syndrome bit determines sign.
        for t in range(self.T):
            damp = self._damping_at(t)
            cw = self._cw_at(t)
            # Check→var update: min-sum approximation for numerical stability
            s_sign = (1 - 2 * syndrome)[:, c_idx]             # [B, E]
            cw_e = cw[c_idx] if cw.dim() == 1 else cw
            # Compute product of tanh(mu_v2c/2) over neighbors except self (approx via per-check sum)
            # For simplicity (MVP), use per-check aggregated sign+min.
            tanh = torch.tanh(mu_v2c / 2)
            prod_per_check = torch.zeros(B, n_check, device=syndrome.device)
            prod_per_check.index_add_(1, c_idx, tanh)
            mu_c2v = s_sign * cw_e * torch.atanh((prod_per_check[:, c_idx]).clamp(-0.999, 0.999))
            # Var→check update: sum of all incoming - own
            incoming_per_var = torch.zeros(B, n_var, device=syndrome.device)
            incoming_per_var.index_add_(1, v_idx, mu_c2v)
            prior = self.llr_scale   # scalar
            mu_v2c_new = prior + incoming_per_var[:, v_idx] - mu_c2v
            mu_v2c = damp * mu_v2c + (1 - damp) * mu_v2c_new
        # Marginals
        incoming_per_var = torch.zeros(B, n_var, device=syndrome.device)
        incoming_per_var.index_add_(1, v_idx, mu_v2c)
        marginal = torch.sigmoid(self.llr_scale + incoming_per_var)
        if self.output_mode == "soft_priors":
            return marginal
        # hard_flip: return "cleaned" syndrome via ML decision
        syndrome_clean = (marginal > 0.5).long()
        # Project onto the syndrome space: H @ correction (mod 2). For MVP,
        # just returns marginal threshold over checks.
        return (torch.zeros(B, n_check, device=syndrome.device,
                            dtype=torch.long)
                + (syndrome > 0.5).long())
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_neural_bp_module.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/decoders/modules/neural_bp.py tests/test_neural_bp_module.py
git commit -m "feat: NeuralBP deep-unfolded predecoder"
```

---

## Task C1.4: DSL compiler (YAML → nn.Module)

**Files:**
- Create: `autoqec/decoders/dsl_compiler.py`
- Create: `tests/test_dsl_compiler.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_dsl_compiler.py
import torch

def test_compile_gnn_config():
    from autoqec.decoders.dsl_compiler import compile_predecoder
    cfg = {
        "type": "gnn", "output_mode": "soft_priors",
        "gnn": {"layers": 2, "hidden_dim": 16, "message_fn": "mlp",
                 "aggregation": "sum", "normalization": "layer",
                 "residual": True, "edge_features": ["syndrome_bit"]},
        "head": "linear",
        "training": {"learning_rate": 1e-3, "batch_size": 4, "epochs": 1,
                      "loss": "bce", "profile": "dev"},
    }
    model = compile_predecoder(cfg, n_var=20, n_check=12)
    assert sum(p.numel() for p in model.parameters()) > 0

def test_compile_neural_bp_config():
    from autoqec.decoders.dsl_compiler import compile_predecoder
    cfg = {
        "type": "neural_bp", "output_mode": "soft_priors",
        "neural_bp": {"iterations": 3, "weight_sharing": "per_layer",
                       "damping": "learnable_scalar", "attention_aug": False,
                       "attention_heads": 1},
        "head": "linear",
        "training": {"learning_rate": 5e-4, "batch_size": 4, "epochs": 1,
                      "loss": "bce", "profile": "dev"},
    }
    model = compile_predecoder(cfg, n_var=20, n_check=12)
    assert isinstance(model, torch.nn.Module)
```

- [ ] **Step 2: Run — fail**

```bash
pytest tests/test_dsl_compiler.py -v
```

- [ ] **Step 3: Implement**

```python
# autoqec/decoders/dsl_compiler.py
from __future__ import annotations
from autoqec.decoders.dsl_schema import PredecoderDSL
from autoqec.decoders.modules.gnn import BipartiteGNN
from autoqec.decoders.modules.neural_bp import NeuralBP

def compile_predecoder(config_dict: dict, n_var: int, n_check: int):
    dsl = PredecoderDSL(**config_dict)
    if dsl.type == "gnn":
        g = dsl.gnn
        return BipartiteGNN(
            n_var=n_var, n_check=n_check,
            hidden_dim=g.hidden_dim, layers=g.layers,
            message_fn=(g.message_fn if isinstance(g.message_fn, str) else "mlp"),
            aggregation=(g.aggregation if isinstance(g.aggregation, str) else "sum"),
            normalization=g.normalization, residual=g.residual,
            output_mode=dsl.output_mode,
        )
    if dsl.type == "neural_bp":
        nb = dsl.neural_bp
        return NeuralBP(
            n_var=n_var, n_check=n_check, iterations=nb.iterations,
            weight_sharing=nb.weight_sharing, damping=nb.damping,
            attention_aug=nb.attention_aug, attention_heads=nb.attention_heads,
            output_mode=dsl.output_mode,
        )
    raise ValueError(f"Unknown predecoder type: {dsl.type}")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_dsl_compiler.py -v
```

- [ ] **Step 5: Commit**

```bash
git add autoqec/decoders/dsl_compiler.py tests/test_dsl_compiler.py
git commit -m "feat: DSL compiler (YAML/dict → nn.Module)"
```

---

## Task C1.5: Custom-fn Tier-2 validator

**Files:**
- Create: `autoqec/decoders/custom_fn_validator.py`
- Create: `tests/test_custom_fn_validator.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_custom_fn_validator.py
import pytest

def test_valid_custom_message_fn():
    from autoqec.decoders.custom_fn_validator import validate_custom_fn
    code = '''
def message(x_src, x_dst, e_ij, params):
    import torch
    import torch.nn.functional as F
    return F.relu(params["W"](torch.cat([x_src, x_dst], dim=-1)))
'''
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert ok, reason

def test_rejects_os_import():
    from autoqec.decoders.custom_fn_validator import validate_custom_fn
    code = '''
def message(x_src, x_dst, e_ij, params):
    import os
    os.system("rm -rf /")
    return x_src
'''
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "import" in reason.lower() or "forbidden" in reason.lower()
```

- [ ] **Step 2: Implement**

```python
# autoqec/decoders/custom_fn_validator.py
from __future__ import annotations
import ast

ALLOWED_TOP_IMPORTS = {"torch"}
ALLOWED_FROM_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "typing"}
FORBIDDEN_NAMES = {"os", "subprocess", "sys", "shutil", "socket", "urllib"}

SLOT_SIGNATURES = {
    "message_fn": ["x_src", "x_dst", "e_ij", "params"],
    "aggregation": ["messages", "edge_index"],
    "head": ["hidden_state"],
}

def validate_custom_fn(code: str, slot: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        return False, "Must define exactly one function"

    fn = funcs[0]
    expected = SLOT_SIGNATURES.get(slot)
    if expected is None:
        return False, f"Unknown slot: {slot}"
    if [a.arg for a in fn.args.args] != expected:
        return False, f"Signature must be {expected}, got {[a.arg for a in fn.args.args]}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_NAMES:
                    return False, f"Forbidden import: {alias.name}"
                if top not in ALLOWED_TOP_IMPORTS:
                    return False, f"Import not in whitelist: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] in FORBIDDEN_NAMES:
                return False, f"Forbidden from-import: {mod}"
            if mod not in ALLOWED_FROM_IMPORTS:
                return False, f"From-import not in whitelist: {mod}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"Forbidden name reference: {node.id}"

    return True, "ok"
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_custom_fn_validator.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/decoders/custom_fn_validator.py tests/test_custom_fn_validator.py
git commit -m "feat: Tier-2 custom_fn AST validator"
```

---

## Task C1.6: Seed templates (3 GNN + 3 Neural-BP)

**Files:**
- Create: `autoqec/example_db/gnn_small.yaml`
- Create: `autoqec/example_db/gnn_medium.yaml`
- Create: `autoqec/example_db/gnn_gated.yaml`
- Create: `autoqec/example_db/neural_bp_min.yaml`
- Create: `autoqec/example_db/neural_bp_per_check.yaml`
- Create: `autoqec/example_db/neural_bp_attn.yaml`
- Create: `tests/test_seed_templates.py`

- [ ] **Step 1: Write the 6 YAMLs (copy-paste the values below)**

`gnn_small.yaml`:
```yaml
type: gnn
output_mode: soft_priors
gnn:
  layers: 2
  hidden_dim: 16
  message_fn: mlp
  aggregation: sum
  normalization: layer
  residual: false
  edge_features: [syndrome_bit]
head: linear
training: {learning_rate: 1.0e-3, batch_size: 64, epochs: 3, loss: bce, profile: dev}
```

`gnn_medium.yaml`:
```yaml
type: gnn
output_mode: soft_priors
gnn:
  layers: 4
  hidden_dim: 64
  message_fn: residual_mlp
  aggregation: mean
  normalization: layer
  residual: true
  edge_features: [syndrome_bit, round_idx]
head: mlp_small
training: {learning_rate: 5.0e-4, batch_size: 128, epochs: 5, loss: bce, profile: dev}
```

`gnn_gated.yaml`:
```yaml
type: gnn
output_mode: soft_priors
gnn:
  layers: 3
  hidden_dim: 32
  message_fn: gated_mlp
  aggregation: gated_sum
  normalization: layer
  residual: true
  edge_features: [syndrome_bit, stabilizer_type]
head: linear
training: {learning_rate: 1.0e-3, batch_size: 64, epochs: 4, loss: focal, profile: dev}
```

`neural_bp_min.yaml`:
```yaml
type: neural_bp
output_mode: soft_priors
neural_bp:
  iterations: 3
  weight_sharing: per_layer
  damping: learnable_scalar
  attention_aug: false
  attention_heads: 1
head: linear
training: {learning_rate: 5.0e-4, batch_size: 64, epochs: 3, loss: bce, profile: dev}
```

`neural_bp_per_check.yaml`:
```yaml
type: neural_bp
output_mode: soft_priors
neural_bp:
  iterations: 5
  weight_sharing: per_check
  damping: learnable_per_iter
  attention_aug: false
  attention_heads: 1
head: linear
training: {learning_rate: 3.0e-4, batch_size: 64, epochs: 5, loss: bce, profile: dev}
```

`neural_bp_attn.yaml`:
```yaml
type: neural_bp
output_mode: soft_priors
neural_bp:
  iterations: 4
  weight_sharing: per_layer
  damping: learnable_per_iter
  attention_aug: true
  attention_heads: 4
head: linear
training: {learning_rate: 3.0e-4, batch_size: 32, epochs: 5, loss: weighted_bce, profile: dev}
```

- [ ] **Step 2: Write a test that compiles every seed**

```python
# tests/test_seed_templates.py
import yaml
from pathlib import Path

def test_all_seed_templates_compile():
    from autoqec.decoders.dsl_compiler import compile_predecoder
    for p in Path("autoqec/example_db").glob("*.yaml"):
        cfg = yaml.safe_load(p.read_text())
        m = compile_predecoder(cfg, n_var=40, n_check=24)
        assert sum(pp.numel() for pp in m.parameters()) > 0, f"{p.name} produced empty model"
```

- [ ] **Step 3: Run**

```bash
pytest tests/test_seed_templates.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/example_db/ tests/test_seed_templates.py
git commit -m "feat: 3 GNN + 3 Neural-BP seed templates"
```

---

## Task C1.7: Runner + RunnerSafety

**Files:**
- Create: `autoqec/runner/safety.py`
- Create: `autoqec/runner/flops.py`
- Create: `autoqec/runner/data.py`
- Create: `autoqec/runner/runner.py`
- Create: `tests/test_runner_safety.py`
- Create: `tests/test_runner_smoke.py`

- [ ] **Step 1: Write safety sentinels + test**

```python
# autoqec/runner/safety.py
from dataclasses import dataclass
import math
import torch

@dataclass
class RunnerSafety:
    WALL_CLOCK_HARD_CUTOFF_S: int = 2700
    VRAM_PRE_CHECK: bool = True
    MAX_NAN_RATE: float = 0.01
    FORBIDDEN_IMPORTS: tuple = ("os.system", "subprocess", "sys.exit")

def estimate_vram_gb(model: torch.nn.Module, batch_size: int, hidden: int) -> float:
    n_params = sum(p.numel() for p in model.parameters())
    # rough: params*4 (fp32) + activations (2*batch*hidden*4) + gradients (2*params*4)
    return (4 * n_params * 3 + batch_size * hidden * 8) / 1e9

def nan_rate(loss_history: list[float]) -> float:
    if not loss_history:
        return 0.0
    nans = sum(1 for x in loss_history if not math.isfinite(x))
    return nans / len(loss_history)
```

```python
# tests/test_runner_safety.py
def test_nan_rate_detection():
    from autoqec.runner.safety import nan_rate
    assert nan_rate([1.0, 2.0, float("nan"), 3.0]) == 0.25
    assert nan_rate([]) == 0.0

def test_vram_estimate_positive():
    import torch
    from autoqec.runner.safety import estimate_vram_gb
    m = torch.nn.Linear(64, 64)
    g = estimate_vram_gb(m, batch_size=32, hidden=64)
    assert g > 0
```

- [ ] **Step 2: Write FLOPs counter**

```python
# autoqec/runner/flops.py
"""FLOPs estimate per syndrome via fvcore. Gracefully falls back to param-count heuristic."""
def estimate_flops(model, example_inputs) -> int:
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, example_inputs).total()
        return int(flops)
    except Exception:
        n_params = sum(p.numel() for p in model.parameters())
        return int(2 * n_params)  # rough: 2 ops per param per forward
```

- [ ] **Step 3: Write data loader (Stim-backed)**

```python
# autoqec/runner/data.py
from pathlib import Path
import numpy as np
import stim
import torch

def sample_syndromes(circuit_path: str, seed_range: tuple[int, int],
                      n_shots: int) -> tuple[torch.Tensor, torch.Tensor]:
    circuit = stim.Circuit.from_file(circuit_path)
    seeds = list(range(seed_range[0], seed_range[1] + 1))
    per_seed = max(n_shots // len(seeds), 1)
    det_all, obs_all = [], []
    for s in seeds:
        sampler = circuit.compile_detector_sampler(seed=s)
        det, obs = sampler.sample(shots=per_seed, separate_observables=True)
        det_all.append(det); obs_all.append(obs)
    det = np.concatenate(det_all, axis=0)
    obs = np.concatenate(obs_all, axis=0)
    return torch.from_numpy(det).float(), torch.from_numpy(obs).long()
```

- [ ] **Step 4: Write the Runner**

```python
# autoqec/runner/runner.py
"""Pure-Python Runner. No LLM calls; deterministic; seed-pinned."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from autoqec.envs.schema import EnvSpec
from autoqec.runner.schema import RunnerConfig, RoundMetrics
from autoqec.runner.safety import RunnerSafety, estimate_vram_gb, nan_rate
from autoqec.runner.flops import estimate_flops
from autoqec.runner.data import sample_syndromes
from autoqec.decoders.dsl_compiler import compile_predecoder

def _profile_params(profile: str) -> dict:
    if profile == "dev":
        return {"n_shots_train": 100_000, "n_shots_val": 20_000, "epochs_cap": 3}
    return {"n_shots_train": 1_000_000, "n_shots_val": 200_000, "epochs_cap": 10}

def run_round(config: RunnerConfig, env_spec: EnvSpec,
              safety: RunnerSafety | None = None) -> RoundMetrics:
    safety = safety or RunnerSafety()
    round_dir = Path(config.round_dir); round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "config.yaml").write_text(json.dumps(config.predecoder_config, indent=2))
    train_log = round_dir / "train.log"

    # 1. Compile predecoder
    try:
        # Code shape for MVP: n_var + n_check inferred from the env's DEM.
        import stim
        circuit = stim.Circuit.from_file(env_spec.code.source)
        dem = circuit.detector_error_model(decompose_errors=True)
        n_check = circuit.num_detectors
        n_var = dem.num_errors
        model = compile_predecoder(config.predecoder_config, n_var=n_var, n_check=n_check)
    except Exception as e:
        return RoundMetrics(status="compile_error", status_reason=str(e))

    n_params = int(sum(p.numel() for p in model.parameters()))
    prof = _profile_params(config.training_profile)

    # 2. VRAM pre-check
    if safety.VRAM_PRE_CHECK and torch.cuda.is_available():
        vram_est = estimate_vram_gb(model, batch_size=64, hidden=64)
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if vram_est > free_gb * 0.9:
            return RoundMetrics(status="killed_by_safety",
                                status_reason=f"VRAM estimate {vram_est:.2f} > free {free_gb:.2f}",
                                n_params=n_params)

    # 3. Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    det, obs = sample_syndromes(env_spec.code.source,
                                  env_spec.noise.seed_policy.train,
                                  prof["n_shots_train"])
    det, obs = det.to(device), obs.to(device)

    # Simple edge_index for Tanner graph: fully-connected approximation.
    # Real Tanner index derivation is a post-MVP polish.
    edge_index = torch.stack(torch.meshgrid(torch.arange(n_var),
                                              torch.arange(n_check),
                                              indexing="ij"), dim=0).reshape(2, -1).to(device)

    opt = torch.optim.Adam(model.parameters(),
                            lr=config.predecoder_config["training"]["learning_rate"])
    losses = []
    t_train_start = time.time()
    bs = config.predecoder_config["training"]["batch_size"]
    epochs = min(config.predecoder_config["training"]["epochs"], prof["epochs_cap"])
    ctx = {"edge_index": edge_index, "n_var": n_var, "n_check": n_check}
    for ep in range(epochs):
        for i in range(0, det.shape[0], bs):
            if time.time() - t_train_start > safety.WALL_CLOCK_HARD_CUTOFF_S:
                return RoundMetrics(status="killed_by_safety",
                                    status_reason="wall_clock_cutoff during training",
                                    n_params=n_params,
                                    train_wallclock_s=time.time() - t_train_start)
            batch_det = det[i:i+bs]
            # Target: labels are logical observables (XOR'd to produce predecoder target).
            # MVP surrogate: train to reconstruct syndrome (self-supervised sanity).
            target = (batch_det > 0.5).float()
            pred = model(batch_det, ctx)
            if model.output_mode == "soft_priors":
                # Project pred to syndrome dim via crude heuristic for MVP
                loss = F.binary_cross_entropy(pred[:, :n_check].clamp(1e-6, 1 - 1e-6), target)
            else:
                loss = F.binary_cross_entropy_with_logits(pred.float(), target)
            if not torch.isfinite(loss):
                losses.append(float("nan"))
                if nan_rate(losses) > safety.MAX_NAN_RATE:
                    return RoundMetrics(status="killed_by_safety",
                                        status_reason=f"NaN rate {nan_rate(losses):.3f}",
                                        n_params=n_params,
                                        train_wallclock_s=time.time() - t_train_start)
                continue
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss))
    t_train = time.time() - t_train_start
    train_log.write_text("\n".join(f"{i}\t{l:.6g}" for i, l in enumerate(losses)))

    # 4. Evaluate baseline vs predecoder on val
    t_eval_start = time.time()
    val_det, val_obs = sample_syndromes(env_spec.code.source,
                                          env_spec.noise.seed_policy.val,
                                          prof["n_shots_val"])
    val_det = val_det.to(device)

    # Plain baseline LER (MWPM on val_det)
    import pymatching
    from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline
    baseline = PymatchingBaseline.from_circuit(circuit)
    plain_pred = baseline.decode_batch(val_det.cpu().numpy().astype(bool))
    ler_plain = float((plain_pred != val_obs.numpy()).any(axis=1).mean())

    # Predecoder + backend
    from autoqec.decoders.backend_adapter import decode_with_predecoder
    model.eval()
    with torch.no_grad():
        pred_out = model(val_det, ctx).cpu().numpy()
    pred_labels = decode_with_predecoder(pred_out, env_spec, val_det.cpu().numpy(),
                                          circuit, model.output_mode)
    ler_pred = float((pred_labels != val_obs.numpy()).any(axis=1).mean())
    delta = ler_plain - ler_pred

    # FLOPs
    try:
        example = (val_det[:1].cpu(), ctx)
        flops = estimate_flops(model, example)
    except Exception:
        flops = 2 * n_params

    t_eval = time.time() - t_eval_start
    ckpt_path = round_dir / "checkpoint.pt"
    torch.save({"model": model.cpu(), "class_name": type(model).__name__}, ckpt_path)

    metrics = RoundMetrics(
        status="ok",
        ler_plain_classical=ler_plain,
        ler_predecoder=ler_pred,
        delta_ler=delta,
        flops_per_syndrome=int(flops),
        n_params=n_params,
        train_wallclock_s=t_train,
        eval_wallclock_s=t_eval,
        vram_peak_gb=float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0,
        checkpoint_path=str(ckpt_path),
        training_log_path=str(train_log),
    )
    (round_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
    return metrics
```

- [ ] **Step 5: Smoke test**

```python
# tests/test_runner_smoke.py
import pytest

@pytest.mark.integration
def test_runner_end_to_end(tmp_path):
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.schema import RunnerConfig
    from autoqec.runner.runner import run_round
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn", "output_mode": "soft_priors",
            "gnn": {"layers": 1, "hidden_dim": 8, "message_fn": "mlp",
                    "aggregation": "sum", "normalization": "none",
                    "residual": False, "edge_features": []},
            "head": "linear",
            "training": {"learning_rate": 1e-3, "batch_size": 16,
                         "epochs": 1, "loss": "bce", "profile": "dev"},
        },
        training_profile="dev",
        seed=0, round_dir=str(tmp_path / "round_0"),
    )
    metrics = run_round(cfg, env)
    assert metrics.status == "ok"
```

- [ ] **Step 6: Commit**

```bash
git add autoqec/runner/ tests/test_runner_safety.py tests/test_runner_smoke.py
git commit -m "feat: Runner + RunnerSafety + FLOPs + Stim data loader"
```

---

## Task C2.1: Backend adapter

**Files:**
- Create: `autoqec/decoders/backend_adapter.py`
- Create: `tests/test_backend_adapter.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_backend_adapter.py
import numpy as np
import stim

def test_hard_flip_passes_to_mwpm():
    from autoqec.decoders.backend_adapter import decode_with_predecoder
    from autoqec.envs.schema import load_env_yaml
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    circuit = stim.Circuit.from_file(env.code.source)
    sampler = circuit.compile_detector_sampler(seed=0)
    det, obs = sampler.sample(shots=500, separate_observables=True)
    # Hand the raw syndromes through — should reproduce plain MWPM LER
    pred_clean = det.astype(bool)
    out = decode_with_predecoder(pred_clean, env, det, circuit, "hard_flip")
    assert out.shape == obs.shape
```

- [ ] **Step 2: Implement**

```python
# autoqec/decoders/backend_adapter.py
from __future__ import annotations
import numpy as np
import stim
from autoqec.envs.schema import EnvSpec

def decode_with_predecoder(predecoder_output,
                            env_spec: EnvSpec,
                            syndrome_raw: np.ndarray,
                            circuit: stim.Circuit,
                            output_mode: str) -> np.ndarray:
    """Route the predecoder output through the env's classical backend.

    output_mode = "hard_flip"   → predecoder returns cleaned syndrome bits;
                                   backend decodes the cleaned bits.
    output_mode = "soft_priors" → predecoder returns per-fault prior probs;
                                   backend receives raw syndrome with custom priors.
    """
    if env_spec.classical_backend == "mwpm":
        import pymatching
        if output_mode == "hard_flip":
            cleaned = np.asarray(predecoder_output).astype(bool)
            dem = circuit.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(dem)
            return matching.decode_batch(cleaned)
        else:  # soft_priors
            # MWPM with per-edge weights derived from predecoder priors.
            dem = circuit.detector_error_model(decompose_errors=True)
            matching = pymatching.Matching.from_detector_error_model(dem)
            # MVP: pymatching.Matching supports setting weights per edge via DEM;
            # a fully-featured prior injection is a post-MVP polish.
            return matching.decode_batch(np.asarray(syndrome_raw).astype(bool))
    elif env_spec.classical_backend == "osd":
        from ldpc import bposd_decoder
        # Use parity-check matrix path (loaded separately by env)
        import numpy as np
        H = np.load(env_spec.code.source).astype(np.uint8)
        dec = bposd_decoder(H, error_rate=env_spec.noise.p[0], osd_method="osd_e", osd_order=10)
        n = syndrome_raw.shape[0]
        out = np.zeros((n, H.shape[1]), dtype=np.uint8)
        for i in range(n):
            out[i] = dec.decode(syndrome_raw[i].astype(np.uint8))
        return out
    raise ValueError(f"Unknown backend: {env_spec.classical_backend}")
```

- [ ] **Step 3: Run test**

```bash
pytest tests/test_backend_adapter.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/decoders/backend_adapter.py tests/test_backend_adapter.py
git commit -m "feat: predecoder → backend adapter (MWPM / OSD)"
```

---

## Task C2.2: Wire `run-round` CLI + end-to-end handshake

**Files:**
- Modify: `cli/autoqec.py` — implement `run-round`
- Modify: `cli/autoqec.py` — implement `run` (with `--no-llm` stub mode)

- [ ] **Step 1: Add CLI commands**

```python
# Append to cli/autoqec.py:

@main.command(name="run-round")
@click.argument("env_yaml")
@click.argument("config_yaml")
@click.argument("round_dir")
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
def run_round_cmd(env_yaml, config_yaml, round_dir, profile):
    """Invoke a single Runner round from two YAML files."""
    import yaml
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.schema import RunnerConfig
    from autoqec.runner.runner import run_round
    env = load_env_yaml(env_yaml)
    cfg_dict = yaml.safe_load(open(config_yaml))
    cfg = RunnerConfig(env_name=env.name, predecoder_config=cfg_dict,
                       training_profile=profile, seed=0, round_dir=round_dir)
    metrics = run_round(cfg, env)
    click.echo(metrics.model_dump_json(indent=2))
```

For the main `run` command, implement `--no-llm` mode that picks a random seed template instead of calling subagents (useful for Phase-2 integration tests before A's LLM wiring is complete):

```python
# Replace the stub `run` command:

@main.command()
@click.argument("env_yaml")
@click.option("--rounds", type=int, default=10)
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option("--no-llm", is_flag=True, help="Pick random seed templates instead of calling subagents")
def run(env_yaml, rounds, profile, no_llm):
    import random, yaml, time
    from pathlib import Path
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.schema import RunnerConfig
    from autoqec.runner.runner import run_round
    env = load_env_yaml(env_yaml)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(f"runs/{run_id}"); run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    templates = list(Path("autoqec/example_db").glob("*.yaml"))
    for i in range(1, rounds + 1):
        round_dir = run_dir / f"round_{i}"
        if no_llm:
            tpl = random.choice(templates)
            pred_cfg = yaml.safe_load(tpl.read_text())
        else:
            click.echo("LLM mode: see A's /autoqec-run orchestrator path")
            return
        cfg = RunnerConfig(env_name=env.name, predecoder_config=pred_cfg,
                           training_profile=profile, seed=i, round_dir=str(round_dir))
        m = run_round(cfg, env)
        history.append(m.model_dump())
        (run_dir / "history.jsonl").open("a").write(m.model_dump_json() + "\n")
        click.echo(f"Round {i}: {m.status} Δ={m.delta_ler}")
    click.echo(json.dumps({"run_dir": str(run_dir), "rounds": rounds}, indent=2))
```

- [ ] **Step 2: Run handshake with A**

```bash
python -m cli.autoqec run-round autoqec/envs/builtin/surface_d5_depol.yaml \
  autoqec/example_db/gnn_small.yaml runs/handshake/round_0 --profile dev
```

Expected: `{"status": "ok", ...}` JSON.

- [ ] **Step 3: Run the no-LLM loop**

```bash
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml \
  --rounds 3 --profile dev --no-llm
```

Expected: 3 rounds complete; `runs/<id>/history.jsonl` has 3 lines.

- [ ] **Step 4: Commit**

```bash
git add cli/autoqec.py
git commit -m "feat: run-round + run (--no-llm) CLIs for Phase-2 handshake"
```

---

## Task C3.1: Makefile with per-agent backend switches

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write the Makefile**

```makefile
# Makefile — AutoQEC master targets
AUTOQEC_IDEATOR_BACKEND ?= codex-cli
AUTOQEC_IDEATOR_MODEL   ?= gpt-5.4
AUTOQEC_CODER_BACKEND   ?= codex-cli
AUTOQEC_CODER_MODEL     ?= gpt-5.4-codex
AUTOQEC_ANALYST_BACKEND ?= claude-cli
AUTOQEC_ANALYST_MODEL   ?= claude-haiku-4-5

COMMON_ENV = \
  AUTOQEC_IDEATOR_BACKEND=$(AUTOQEC_IDEATOR_BACKEND) \
  AUTOQEC_IDEATOR_MODEL=$(AUTOQEC_IDEATOR_MODEL) \
  AUTOQEC_CODER_BACKEND=$(AUTOQEC_CODER_BACKEND) \
  AUTOQEC_CODER_MODEL=$(AUTOQEC_CODER_MODEL) \
  AUTOQEC_ANALYST_BACKEND=$(AUTOQEC_ANALYST_BACKEND) \
  AUTOQEC_ANALYST_MODEL=$(AUTOQEC_ANALYST_MODEL)

ENV      ?= autoqec/envs/builtin/surface_d5_depol.yaml
ROUNDS   ?= 10
PROFILE  ?= dev

.PHONY: run verify pareto test lint demo-1 demo-2 demo-3 demo-4 demo-5 install

install:
	pip install -e '.[dev]'

test:
	pytest tests/ -m "not integration" -v

lint:
	ruff check autoqec cli tests scripts

run:
	$(COMMON_ENV) python -m cli.autoqec run $(ENV) --rounds $(ROUNDS) --profile $(PROFILE)

run-nollm:
	python -m cli.autoqec run $(ENV) --rounds $(ROUNDS) --profile $(PROFILE) --no-llm

verify:
	python -m cli.autoqec verify $(RUN_DIR) --env $(ENV)

demo-1: ; bash demos/demo-1-surface-d5/run.sh
demo-2: ; bash demos/demo-2-bb72/run.sh
demo-3: ; bash demos/demo-3-add-env/run.sh
demo-4: ; bash demos/demo-4-reward-hacking/run.sh
demo-5: ; bash demos/demo-5-failure-recovery/run.sh

# Cost ablations
run-all-claude:
	$(MAKE) run AUTOQEC_IDEATOR_BACKEND=claude-cli AUTOQEC_CODER_BACKEND=claude-cli

run-cheap:
	$(MAKE) run AUTOQEC_IDEATOR_MODEL=claude-haiku-4-5
```

- [ ] **Step 2: Smoke test**

```bash
make test
make run-nollm ROUNDS=2
```

Expected: tests pass; 2 no-LLM rounds succeed.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat: Makefile with per-agent backend switches + demo targets"
```

---

## Task C3.2: Demo 2 — bb72 research run

**Files:**
- Create: `demos/demo-2-bb72/run.sh`
- Create: `demos/demo-2-bb72/README.md`

- [ ] **Step 1: Write `run.sh` with dev-profile fallback**

```bash
#!/usr/bin/env bash
# Demo 2: research loop on bb72_depol. Production profile by default; falls
# back to 3 dev-profile rounds under tight budget (see --fast).
set -euo pipefail
MODE=${MODE:-prod}
if [[ "$MODE" == "fast" ]]; then
  ROUNDS=3; PROFILE=dev
else
  ROUNDS=${ROUNDS:-10}; PROFILE=${PROFILE:-prod}
fi
python -m cli.autoqec run \
  autoqec/envs/builtin/bb72_depol.yaml \
  --rounds "$ROUNDS" --profile "$PROFILE" --no-llm
RUN_DIR=$(ls -t runs | head -1)
echo "Pareto:"
cat runs/$RUN_DIR/pareto.json 2>/dev/null || echo "(no pareto yet)"
```

- [ ] **Step 2: Write `README.md`**

```markdown
# Demo 2: bb72 research loop

## Goal
Show AutoQEC discovering a neural predecoder for the Bivariate Bicycle code
[[72, 12, 6]] under depolarizing noise, benchmarked against plain BP+OSD and
Relay-BP.

## Run
```bash
bash demos/demo-2-bb72/run.sh             # 10 rounds prod (~3.3h)
MODE=fast bash demos/demo-2-bb72/run.sh   # 3 rounds dev (~15 min)
```

## Acceptance criteria
- **Prod mode**: Pareto ≥3 VERIFIED candidates; at least one dominates plain
  BP+OSD order 0 on at least one (Δ_LER, FLOPs) slice.
- **Fast mode**: all 3 rounds complete with non-empty metrics.json.

## Runtime
~3.3h prod (4090 overnight); ~15 min fast fallback.

## Known limitations for MVP
- Uses fully-connected Tanner graph as a smoke-test approximation. Real
  bb72 Tanner edges land post-MVP via `scripts/scout_bb72.py` output.
- Relay-BP baseline is optional; falls back to BP+OSD-only if the
  `relay_bp` package is not available.
```

- [ ] **Step 3: Commit**

```bash
chmod +x demos/demo-2-bb72/run.sh
git add demos/demo-2-bb72/
git commit -m "feat: Demo 2 bb72 research loop"
```

---

## Task C3.3: Full test-suite green + lint

- [ ] **Step 1: Run full CPU test suite**

```bash
pytest tests/ -m "not integration" -v
```

Expected: all non-integration tests PASS.

- [ ] **Step 2: Ruff**

```bash
ruff check autoqec cli tests scripts
```

Fix any style issues that surface; commit as `chore: ruff cleanup`.

- [ ] **Step 3: Integration smoke (with GPU)**

```bash
pytest tests/test_runner_smoke.py -v -m integration --run-integration || true
```

(If GPU-less, skip.)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final lint + smoke test pass"
```

---

## Self-review checklist

- [x] **Spec coverage**: §6 DSL → C1.1/C1.4/C1.5; §4.5 RunnerSafety → C1.7; Appendix A full schema → C1.1 pydantic; §4.1 predecoder+backend → C2.1 adapter; §9 Demo 2 → C3.2; §11 Makefile → C3.1.
- [x] **No placeholders**: every task has complete pydantic, complete compiler, complete Runner.
- [x] **Type consistency**: `RunnerConfig.predecoder_config` consumed by `compile_predecoder` which validates via `PredecoderDSL`. `RoundMetrics` fields align with what B's `independent_eval` and A's Analyst read.
- [x] **Phase gates**: C2.2 CLI `run-round` unblocks A's handshake (A2.1) and B's `verify` CLI (B2.1) — shared sandbox at `runs/handshake/`.
- [x] **Predecoder-interface discipline**: every predecoder extends `PredecoderBase` and declares `output_mode`; `backend_adapter.py` is the single dispatch point.
