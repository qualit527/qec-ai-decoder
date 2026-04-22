from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


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

