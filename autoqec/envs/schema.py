from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class SeedPolicy(BaseModel):
    train: tuple[int, int] = (1, 999)
    val: tuple[int, int] = (1000, 1999)
    holdout: tuple[int, int] = (9000, 9999)


class NoiseSpec(BaseModel):
    type: Literal["depolarizing", "biased", "leakage", "custom_dem"]
    p: list[float]
    seed_policy: SeedPolicy = Field(default_factory=SeedPolicy)


class CodeSpec(BaseModel):
    type: Literal["stim_circuit", "parity_check_matrix", "tanner_graph"]
    source: str


class ConstraintsSpec(BaseModel):
    # None = unlimited. Kept as informational metadata for the Ideator, not
    # enforced anywhere in the runner. The only effective budget is the
    # outer wall-clock budget surfaced via machine_state.
    latency_flops_budget: Optional[int] = None
    param_budget: Optional[int] = None
    target_ler: float = 1e-4
    target_p: float = 1e-3


class EvalProtocol(BaseModel):
    min_shots_train: int = 1_000_000
    min_shots_val: int = 100_000
    min_shots_verify: int = 200_000
    bootstrap_ci: float = 0.95
    osd_orders_reported: list[int] = [0, 10]
    x_z_decoding: Literal["circuit", "x_only"] = "circuit"


class EnvSpec(BaseModel):
    name: str
    code: CodeSpec
    noise: NoiseSpec
    constraints: ConstraintsSpec
    baseline_decoders: list[str]
    classical_backend: Literal["mwpm", "osd"]
    eval_protocol: EvalProtocol = Field(default_factory=EvalProtocol)


def load_env_yaml(path: str | Path) -> EnvSpec:
    path = Path(path).expanduser().resolve()
    # Force UTF-8 so YAMLs with non-ASCII comments load consistently on
    # Windows (system default is GBK/CP936 there, which chokes on em-dashes
    # and other common punctuation).
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "code" in data:
        code = dict(data["code"] or {})
        source = code.get("source")
        if source:
            source_path = Path(source).expanduser()
            if not source_path.is_absolute():
                candidates = [(path.parent / source_path).resolve()]
                repo_root = next((parent for parent in path.parents if (parent / "pyproject.toml").exists()), None)
                if repo_root is not None:
                    candidates.append((repo_root / source_path).resolve())
                existing = [candidate for candidate in candidates if candidate.exists()]
                source_path = existing[0] if existing else candidates[0]
            code["source"] = str(source_path)
        data["code"] = code
    return EnvSpec(**data)
