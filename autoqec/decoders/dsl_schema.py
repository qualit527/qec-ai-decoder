from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

EdgeFeature = Literal["syndrome_bit", "round_idx", "stabilizer_type", "distance", "prior_weight"]
MessageFn = Literal[
    "mlp",
    "gated_mlp",
    "attention",
    "gru_cell",
    "edge_attention",
    "geometric_attention",
    "residual_mlp",
    "normalized_mlp",
]
Aggregation = Literal["sum", "mean", "max", "attention_pool", "set_transformer", "gated_sum"]
Normalization = Literal["none", "layer", "batch", "edge_norm", "graph_norm"]
WeightSharing = Literal["none", "per_layer", "per_check"]
Damping = Literal["fixed", "learnable_scalar", "learnable_per_iter"]
LossName = Literal["bce", "focal", "weighted_bce"]
OutputMode = Literal["hard_flip", "soft_priors"]


class CustomFn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["custom"] = "custom"
    code: str
    params_declared: dict[str, str] = Field(default_factory=dict)


class GNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layers: int = Field(ge=1)
    hidden_dim: int = Field(ge=4)
    message_fn: Union[MessageFn, CustomFn]
    aggregation: Union[Aggregation, CustomFn]
    normalization: Normalization
    residual: bool
    edge_features: list[EdgeFeature] = Field(default_factory=list)


class NeuralBPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    iterations: int = Field(ge=1)
    weight_sharing: WeightSharing
    damping: Damping
    attention_aug: bool
    attention_heads: int = Field(ge=1)


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(gt=0)
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    loss: LossName
    profile: Literal["dev", "prod", "benchmark"]


class PredecoderDSL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["gnn", "neural_bp"]
    output_mode: OutputMode
    gnn: GNNConfig | None = None
    neural_bp: NeuralBPConfig | None = None
    head: Union[Literal["linear", "mlp_small"], CustomFn]
    training: TrainingConfig

    @model_validator(mode="after")
    def _check_family_specific_blocks(self) -> "PredecoderDSL":
        if self.type == "gnn":
            if self.gnn is None:
                raise ValueError("gnn spec required when type='gnn'")
            if self.neural_bp is not None:
                raise ValueError("neural_bp block is not allowed when type='gnn'")
        if self.type == "neural_bp":
            if self.neural_bp is None:
                raise ValueError("neural_bp spec required when type='neural_bp'")
            if self.gnn is not None:
                raise ValueError("gnn block is not allowed when type='neural_bp'")
        return self
