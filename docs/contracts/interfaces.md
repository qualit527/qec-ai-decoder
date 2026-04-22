# AutoQEC Interface Contracts

This file freezes the Phase-0 contracts referenced by the owner plans.

## 2.1 `EnvSpec`

```python
from pydantic import BaseModel, Field
from typing import Literal


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
    latency_flops_budget: int = 10_000_000
    param_budget: int = 200_000
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
```

## 2.2 `RunnerConfig` + `RoundMetrics`

```python
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

## 2.3 `VerifyReport`

```python
from pydantic import BaseModel
from typing import Literal


class VerifyReport(BaseModel):
    verdict: Literal["VERIFIED", "SUSPICIOUS", "FAILED"]
    ler_holdout: float
    ler_holdout_ci: tuple[float, float]
    delta_ler_holdout: float
    ler_shuffled: float
    ablation_sanity_ok: bool
    holdout_seeds_used: list[int]
    seed_leakage_check_ok: bool
    notes: str
```

## 2.4 Predecoder I/O

```python
from typing import Literal
from torch import Tensor, nn


class PredecoderModule(nn.Module):
    output_mode: Literal["hard_flip", "soft_priors"]

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        """
        syndrome: [batch, n_checks] float (or [batch, T, n_checks] for circuit).
        ctx: dict with keys:
            "tanner_edges": LongTensor [2, n_edges]
            "edge_features": FloatTensor [n_edges, edge_dim]  (optional)
            "prior_p": FloatTensor [n_faults]  (optional)
        Returns:
            hard_flip: LongTensor [batch, n_checks]
            soft_priors: FloatTensor [batch, n_faults]
        """
```

## 2.5 Subagent message format

- Ideator: `{"hypothesis": str, "expected_delta_ler": float, "expected_cost_s": int, "rationale": str, "dsl_hint": dict?}`
- Coder: `{"dsl_config": {...}, "tier": "1" | "2", "rationale": str}`
- Analyst: `{"summary_1line": str, "verdict": "candidate" | "ignore", "next_hypothesis_seed": str}`

## 2.6 Skill surface

Two surfaces, distinguished by who is driving:

### 2.6.1 Claude-Code-driven skills (recipes in `.claude/skills/<name>/SKILL.md`)

The orchestrator is Claude Code chat. It dispatches subagents via the
`Agent` tool and shells out to per-step Python helpers. These are the
skills that need LLM reasoning at runtime.

- `/autoqec-run` — Ideator → Coder → Runner → Analyst → record loop.
  Recipe: `.claude/skills/autoqec-run/SKILL.md`. Shells out to
  `python -m cli.autoqec run-round <env> <config> <round_dir>` once per
  round.
- `/add-env` *(planned)* — dialog-driven env YAML composer.
- `/verify-decoder` *(planned, owned by Xie)* — holdout evaluation of a
  Pareto candidate.
- `/review-log` *(planned, owned by Xie)* — retrospective over a whole
  `runs/<run_id>/`.
- `/diagnose-failure` *(planned, owned by Xie)* — root-cause a broken or
  stalled round.

### 2.6.2 Pure-CLI commands (no LLM in the loop)

These are Python entry points callable from any shell or skill recipe.
All are registered in `cli/autoqec.py`.

- `python -m cli.autoqec run-round <env.yaml> <config.yaml> <round_dir> [--profile dev|prod]`
  — one Runner round from a hand-written DSL config. Primary building
  block the `/autoqec-run` skill calls per round.
- `python -m cli.autoqec run <env.yaml> --rounds N --profile dev --no-llm`
  — N-round random-template smoke loop. **No LLM.** Writes
  `runs/<id>/history.jsonl + history.json`; does **not** write
  `log.md` / `pareto.json` (those are orchestration-side, only produced
  by `/autoqec-run`).
- `python -m cli.autoqec add-env --out <env.yaml>`
  — interactive EnvSpec YAML composer (no LLM; just click prompts).

**Historical note:** earlier drafts of this contract promised a
`python -m autoqec run <env> --rounds N` entrypoint that would drive
the full LLM loop from plain Python. That path is not implemented on
`main`; the `/autoqec-run` skill is the live LLM-driven surface until a
subprocess router replaces the Claude-chat orchestrator.

