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

---

## §15 Worktree additions (contract-change)

Added 2026-04-22. All fields are Optional on the legacy path; validators
make them required on the worktree path. Authoritative source:
`docs/superpowers/specs/2026-04-20-autoqec-design.md` §15.7.

### `RunnerConfig`

- `code_cwd: Optional[str]` — absolute path to worktree checkout; `None` = legacy in-process path.
- `branch: Optional[str]` — `exp/<run_id>/<NN>-<slug>`; **required when `code_cwd` is set**.
- `fork_from: Optional[Union[str, list[str]]]` — list ⇒ compose round.
- `fork_from_canonical: Optional[str]` — sorted, `|`-joined; dedup key for compose.
- `fork_from_ordered: Optional[list[str]]` — merge-sequence order (drives `git merge` base); compose only.
- `compose_mode: Optional[Literal["pure", "with_edit"]]` — **required when `fork_from` is a list**.

Validators: `code_cwd` set ⇒ `branch` required; list-form `fork_from` ⇒
`compose_mode` required.

> **Note:** `round_attempt_id`, `commit_sha`, and `paired_eval_bundle_id` live
> on `RoundMetrics` / `VerifyReport` (the *output* surfaces), not on
> `RunnerConfig` (the *input* surface). The Runner CLI accepts
> `--round-attempt-id` as a flag and writes it into the emitted
> `metrics.json`; it is not an input schema field on `RunnerConfig`.
> See `autoqec/runner/schema.py` for the authoritative shape.

### `RoundMetrics`

- `round_attempt_id: Optional[str]` — **required on worktree path**.
- `reconcile_id: Optional[str]` — UUID from §15.10 startup reconciliation; set **only** for reconciliation-synthetic rows. Mutually exclusive with `round_attempt_id`.
- `branch: Optional[str]`, `commit_sha: Optional[str]` — paired; `commit_sha` required when `branch` set, **except** when `status == "branch_manually_deleted"` (§15.10 follow-up rows reference a branch that no longer exists in git, so the commit_sha is legitimately unavailable).
- `fork_from: Optional[Union[str, list[str]]]`, `fork_from_canonical: Optional[str]`, `fork_from_ordered: Optional[list[str]]`.
- `compose_mode: Optional[Literal["pure", "with_edit"]]`.
- `delta_vs_parent: Optional[float]`, `parent_ler: Optional[float]` — training-regime Δ; search-guidance only.
- `train_seed: Optional[int]` — seed actually used; for Pareto disambiguation.
- `status_reason: Optional[str]` — short explanation for non-ok / orphaned / branch-manually-deleted statuses.
- `conflicting_files: Optional[list[str]]` — set iff `status == "compose_conflict"`.
- `status` Literal gains: `"compose_conflict"`, `"orphaned_branch"`, `"branch_manually_deleted"` (the last two are set by §15.10 startup reconciliation).

> **Note:** `paired_eval_bundle_id` lives on `VerifyReport` (holdout-eval
> surface), not on `RoundMetrics` (training-side surface). Pareto admission
> reads it from the paired `verify_report.json`.

Validators: worktree-path rows (`branch` set, `fork_from` set, or
`status == "compose_conflict"`) need exactly one of `round_attempt_id`
or `reconcile_id`; `branch` set ⇒ `commit_sha` required;
`compose_conflict` rows MUST have `branch == None` and
`commit_sha == None`.

### `VerifyReport`

- `branch: Optional[str]`, `commit_sha: Optional[str]` — paired.
- `delta_vs_baseline_holdout: Optional[float]` — paired-bundle canonical delta (§15.6.4).
- `paired_eval_bundle_id: Optional[str]` — required for compose rounds and any Pareto candidate.

### `IdeatorResponse`

- `fork_from: Union[Literal["baseline"], str, list[str]] = "baseline"` — default unblocks legacy responses.
- `compose_mode: Optional[Literal["pure", "with_edit"]]` — required when `fork_from` is a list.

### `CoderResponse`

- `commit_message: Optional[str]` — filled by Coder on the worktree path; absent tolerated on legacy responses.

