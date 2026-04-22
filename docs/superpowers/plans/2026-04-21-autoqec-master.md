# AutoQEC Master Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Coordinate 3 owners (陈嘉汉 / 谢金谷 / 林腾祥) to ship the AutoQEC MVP per `docs/superpowers/specs/2026-04-20-autoqec-design.md` v2.2 within ~3 days by freezing interfaces early and running scaffolds in parallel.

**Architecture:** 3 owners work in parallel after a Phase-0 contract freeze. Each owns one QEC-core workstream + one delivery-facing workstream (see spec §12.1). A single Python package `autoqec/` is shared; each owner owns a disjoint subtree so merges don't collide.

**Tech Stack:** Python 3.10+, PyTorch, Stim, sinter, PyMatching, ldpc (BP+OSD), pydantic, click, pytest. Optional: LangGraph for orchestration (or plain Python loop).

---

## 0. Reading order

Read in this order before starting work:
1. `docs/superpowers/specs/2026-04-20-autoqec-design.md` (the spec — source of truth)
2. **This file** — master coordination
3. Your own plan:
   - 陈嘉汉 → `docs/superpowers/plans/2026-04-21-autoqec-person-a-chen.md`
   - 谢金谷 → `docs/superpowers/plans/2026-04-21-autoqec-person-b-xie.md`
   - 林腾祥 → `docs/superpowers/plans/2026-04-21-autoqec-person-c-lin.md`
4. Companion knowledge: `knowledge/DECODER_ROADMAP.md` (DSL building blocks), `knowledge/STRATEGIC_ASSESSMENT.md` (novelty/venue), `knowledge/AUTORESEARCH_PATTERNS.md` (what to port from AIDE / open-coscientist).

---

## 1. Logical phases (not calendar days)

The plan is **phase-driven**, not day-driven. Each phase has a gate; downstream phases must not start until gate holds.

| Phase | Gate (done when) | Who | Typical duration |
|---|---|---|---|
| **Phase 0 — Contract freeze** | `docs/contracts/interfaces.md` committed, all 3 owners have signed off on the 6 interfaces in §2 below | All 3 (synchronous, ~90 min) | ~0.5 day |
| **Phase 1 — Parallel scaffolds** | Each owner's Tasks 1..K unit-tested in isolation. `pytest -m "not integration"` green on each owner's subtree | Each owner in parallel | ~1 day |
| **Phase 2 — End-to-end integration** | One full research round completes: `config.yaml → Runner → metrics.json → independent_eval → VerifyReport`, invoked by a minimal orchestrator on `surface_d5_depol` | A+C (round wiring) then B (verify wiring) | ~0.5–1 day |
| **Phase 3 — Demos, skills, polish** | Demo 1 + Demo 4 ship VERIFIED; `/autoqec-run` + `/verify-decoder` callable; remaining 3 skills have SKILL.md; Demo 2 runs at least 3 dev-profile rounds | Each owner finishes their demos + skills | ~1 day |

**Execution principle (from spec §12.1):** no one does only glue. Each owner touches real QEC artifacts and one delivery-facing surface.

## 2. Interface contracts (Phase 0 — frozen deliverable)

After Phase 0 these six interfaces are **frozen**. Subsequent changes require a PR with all 3 owners signing off. Contracts live in a single file `docs/contracts/interfaces.md` (written in Phase 0 Task M0.1).

### 2.1 `EnvSpec` (陈嘉汉 drafts)

```python
# autoqec/envs/schema.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class SeedPolicy(BaseModel):
    train:   tuple[int, int] = (1, 999)
    val:     tuple[int, int] = (1000, 1999)
    holdout: tuple[int, int] = (9000, 9999)

class NoiseSpec(BaseModel):
    type: Literal["depolarizing", "biased", "leakage", "custom_dem"]
    p: list[float]
    seed_policy: SeedPolicy = Field(default_factory=SeedPolicy)

class CodeSpec(BaseModel):
    type: Literal["stim_circuit", "parity_check_matrix", "tanner_graph"]
    source: str  # relative path

class ConstraintsSpec(BaseModel):
    latency_flops_budget: int = 10_000_000
    param_budget: int = 200_000
    target_ler: float = 1e-4
    target_p: float = 1e-3

class EvalProtocol(BaseModel):
    min_shots_train:   int = 1_000_000
    min_shots_val:     int = 100_000
    min_shots_verify:  int = 200_000
    bootstrap_ci:      float = 0.95
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

### 2.2 `RunnerConfig` + `RoundMetrics` (林腾祥 drafts)

```python
# autoqec/runner/schema.py
from pydantic import BaseModel
from typing import Literal, Optional

class RunnerConfig(BaseModel):
    env_name: str              # matches EnvSpec.name
    predecoder_config: dict    # raw Tier-1/Tier-2 DSL dict
    training_profile: Literal["dev", "prod"] = "dev"
    seed: int = 0
    round_dir: str             # absolute path, runner writes here

class RoundMetrics(BaseModel):
    status: Literal["ok", "killed_by_safety", "compile_error", "train_error"]
    status_reason: Optional[str] = None
    ler_plain_classical: Optional[float] = None
    ler_predecoder:      Optional[float] = None
    delta_ler:           Optional[float] = None
    delta_ler_ci_low:    Optional[float] = None
    delta_ler_ci_high:   Optional[float] = None
    flops_per_syndrome:  Optional[int] = None
    n_params:            Optional[int] = None
    train_wallclock_s:   float = 0.0
    eval_wallclock_s:    float = 0.0
    vram_peak_gb:        float = 0.0
    checkpoint_path:     Optional[str] = None
    training_log_path:   Optional[str] = None
```

### 2.3 `VerifyReport` (谢金谷 drafts)

```python
# autoqec/eval/schema.py
from pydantic import BaseModel
from typing import Literal

class VerifyReport(BaseModel):
    verdict: Literal["VERIFIED", "SUSPICIOUS", "FAILED"]
    ler_holdout:    float
    ler_holdout_ci: tuple[float, float]
    delta_ler_holdout: float
    # Ablation sanity: LER after shuffling predecoder weights.
    ler_shuffled:   float
    ablation_sanity_ok: bool   # True if shuffled LER ≳ plain-classical LER
    holdout_seeds_used: list[int]
    seed_leakage_check_ok: bool  # True if no train/val seeds intersect holdout
    notes: str  # free-form, short
```

### 2.4 Predecoder I/O contract (林腾祥 drafts)

Every predecoder is an `nn.Module` implementing:

```python
class PredecoderModule(nn.Module):
    output_mode: Literal["hard_flip", "soft_priors"]

    def forward(self, syndrome: Tensor, ctx: dict) -> Tensor:
        """
        syndrome: [batch, n_checks] float (or [batch, T, n_checks] for circuit).
        ctx: dict with keys:
            "tanner_edges":  LongTensor [2, n_edges] (bipartite var↔check)
            "edge_features": FloatTensor [n_edges, edge_dim]  (optional)
            "prior_p":       FloatTensor [n_faults]  (initial prior, optional)
        Returns:
            hard_flip:   LongTensor [batch, n_checks]  (cleaned syndrome bits)
            soft_priors: FloatTensor [batch, n_faults] (per-fault posterior)
        """
```

The backend adapter (`autoqec/decoders/backend_adapter.py`) consumes the output and calls MWPM or OSD:

```python
def decode_with_predecoder(predecoder_output, env_spec, syndrome_raw) -> corrections:
    if output_mode == "hard_flip":
        corrected_syndrome = predecoder_output
        return pymatching_or_osd(corrected_syndrome, prior=uniform)
    else:  # soft_priors
        return pymatching_or_osd(syndrome_raw, prior=predecoder_output)
```

### 2.5 Subagent message format (陈嘉汉 drafts)

Inline mode calls subagents via Claude Code's `Agent` tool. Each subagent receives a single string prompt; the orchestrator parses its response. All three subagents must respond with a JSON object inside a fenced `` ```json `` block:

- **Ideator →** `{"hypothesis": str, "expected_delta_ler": float, "expected_cost_s": int, "rationale": str, "dsl_hint": dict?}`
- **Coder →** `{"dsl_config": {...}, "tier": "1" | "2", "rationale": str}`
- **Analyst →** `{"summary_1line": str, "verdict": "candidate" | "ignore", "next_hypothesis_seed": str}`

### 2.6 Skill CLI contract (陈嘉汉 drafts, 谢金谷/林腾祥 consume)

Each `/skill-name` wraps a single CLI:
- `/autoqec-run` → `python -m autoqec run <env.yaml> --rounds N`
- `/add-env`    → `python -m autoqec add-env --out <env.yaml>` (interactive)
- `/verify-decoder` → `python -m autoqec verify <round_dir>`
- `/review-log` → `python -m autoqec review-log <run_dir>`
- `/diagnose-failure` → `python -m autoqec diagnose <run_dir>`

All CLIs must be registered in `cli/autoqec.py` via `click`. Each CLI writes structured output to stdout (JSON or markdown) so skills can parse.

---

## 3. Cross-owner dependency graph

```
                    ┌───────────────────────────┐
                    │ Phase 0: contracts frozen │
                    └────────────┬──────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ↓                    ↓                    ↓
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ A: surface_d5   │  │ B: independent_ │  │ C: DSL + Runner │
   │    env + MWPM   │  │    eval + bb72  │  │    + predecoder │
   │    + orchestr.  │  │    + baselines  │  │    templates    │
   │    skeleton     │  │                  │  │                  │
   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
            │                    │                    │
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                                 ↓
                    ┌───────────────────────────┐
                    │ Phase 2: E2E round works  │
                    │  A↔C: orch calls Runner   │
                    │  B↔C: verify reads ckpt   │
                    └────────────┬──────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ↓                    ↓                    ↓
   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
   │ A: Demo 1 +     │  │ B: Demo 4 +     │  │ C: Demo 2 +     │
   │    /autoqec-run │  │    Demo 5 +     │  │    Makefile +   │
   │    /add-env     │  │    /verify-dec  │  │    CLI polish   │
   │                 │  │    /review-log  │  │                 │
   │                 │  │    /diag-failure│  │                 │
   └─────────────────┘  └─────────────────┘  └─────────────────┘
```

## 4. Integration milestones (with acceptance tests)

| ID | Milestone | Acceptance test |
|---|---|---|
| **M1** | Phase 0 done | `docs/contracts/interfaces.md` committed; `pytest tests/test_contracts.py -v` passes (imports pydantic schemas, validates example payloads) |
| **M2** | Phase 1 done | Each owner's `pytest -m "not integration"` green; `surface_d5` baseline benchmark prints LER and FLOPs; DSL compiler compiles 3 GNN + 3 Neural-BP templates into `nn.Module`; `independent_eval` rejects a hand-crafted "always return random" predecoder |
| **M3** | First Runner end-to-end | `python -m autoqec run envs/surface_d5_depol.yaml --rounds 1 --profile dev --no-llm` completes with non-empty `metrics.json` |
| **M4** | First full LLM round | Orchestrator dispatches Ideator/Coder/Analyst for 1 round, writes `log.md`, `history.jsonl` entries |
| **M5** | First VerifIED candidate | `python -m autoqec verify runs/<id>/round_1/` returns `VERIFIED` on the surface_d5 baseline-beater |
| **M6** | Demo 1 + Demo 4 ship | Both demos' `run.sh` produces expected artifacts; `/autoqec-run` and `/verify-decoder` callable from Claude Code |

## 5. Checkpoint protocol

- **End-of-Phase-0 sync** (mandatory, 30 min): each owner walks through their drafted contract; edits applied in-meeting; contract file merged before anyone starts Phase 1.
- **Daily 15-min standup** (async OK): what I did, where I'm blocked, what contract I need from whom.
- **Pre-integration handshake** (Phase 2 kickoff): A and C demo the `Runner` end-to-end on a stub config together before A wires the LLM subagents.
- **Cross-review rule**: each owner reviews at least one other owner's QEC artifact PR (not just prompts or docs).
- **Shared notebook**: `runs/handshake/` is the agreed sandbox where any owner can drop a test config for another to run.

## 6. Risk register (owner → mitigation)

| Risk | P | Owner | Mitigation |
|---|---|---|---|
| Phase-0 contract drift (someone edits without PR) | Med | All | Contract file tracked in Git; CI blocks merges to `autoqec/envs/schema.py`, `autoqec/runner/schema.py`, `autoqec/eval/schema.py` without label `contract-change` |
| bb72 Stim circuit unavailable | Med | 谢金谷 | Three candidate sources already scouted (`qLDPC`, `stimbposd`, Bravyi github); fallback = hand-build (~200 LOC) |
| Coder Tier-2 failure rate too high | Med | 林腾祥 | Start Tier-1 only in first 5 rounds; enable Tier-2 only if plateau detected. Strict AST + smoke-test pipeline |
| Orchestrator ↔ Runner API mismatch | High | 陈嘉汉 + 林腾祥 | Integration handshake at Phase-2 kickoff; use shared stub config in `runs/handshake/` |
| One owner unavailable | Med | All | Backup readers assigned per spec §12.3; contracts + tests let anyone resume another's module |
| Reward-hacking case too easy / too hard | Med | 谢金谷 | Hand-craft 2 cheating predecoders (memorized-syndrome + label-noise); one must fail, one must pass — calibrates `SUSPICIOUS` threshold |
| Demo fails live | Low | All | Pre-record each demo after Phase-3 morning smoke test |

## 7. Branching and commit discipline

- All work on branch `claude/brainstorm-docs` (already open as PR #2) **or** fresh feature branches like `feat/<owner>-<topic>` rebased onto `main` when stable.
- One commit per completed task step group (tests + implementation + passing output). Small, frequent commits.
- Commit message format (from `~/.claude/rules/common/git-workflow.md`):
  ```
  <type>: <description>

  <optional body>
  ```
  Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`.
- Never amend pushed commits. Never force-push to main.

## 8. Tasks (master-level coordination only)

These are tasks only the coordinator (陈嘉汉, as orchestration owner) executes to unblock the team. Per-owner plans contain the bulk of the work.

### Task M0.1: Write `docs/contracts/interfaces.md`

**Files:**
- Create: `docs/contracts/interfaces.md`

- [ ] **Step 1: Draft contract file**

Copy §2.1–§2.6 of this plan verbatim into `docs/contracts/interfaces.md`, adding a frontmatter block:

```markdown
# AutoQEC Interface Contracts

**Frozen on:** 2026-04-21 (Phase-0 sync)
**Change policy:** any edit requires PR with 3-of-3 owner sign-off and label `contract-change`.

... (paste §2.1–§2.6 here)
```

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/interfaces.md
git commit -m "docs: freeze Phase-0 interface contracts"
```

### Task M0.2: Write `tests/test_contracts.py`

**Files:**
- Create: `tests/test_contracts.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_contracts.py
"""Validates that the six Phase-0 contracts load and accept example payloads."""
import pytest

def test_envspec_accepts_example():
    from autoqec.envs.schema import EnvSpec
    example = {
        "name": "surface_d5_depol",
        "code": {"type": "stim_circuit", "source": "circuits/surface_d5.stim"},
        "noise": {"type": "depolarizing", "p": [1e-3, 5e-3, 1e-2]},
        "constraints": {},
        "baseline_decoders": ["pymatching"],
        "classical_backend": "mwpm",
    }
    spec = EnvSpec(**example)
    assert spec.name == "surface_d5_depol"

def test_runner_metrics_schema_roundtrip():
    from autoqec.runner.schema import RoundMetrics
    m = RoundMetrics(status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4,
                    delta_ler=5e-4, flops_per_syndrome=10000, n_params=50000,
                    train_wallclock_s=100, eval_wallclock_s=20, vram_peak_gb=6.0)
    assert m.status == "ok"

def test_verify_report_schema():
    from autoqec.eval.schema import VerifyReport
    r = VerifyReport(verdict="VERIFIED", ler_holdout=5e-4,
                     ler_holdout_ci=(4e-4, 6e-4),
                     delta_ler_holdout=5e-4, ler_shuffled=1e-3,
                     ablation_sanity_ok=True,
                     holdout_seeds_used=list(range(9000, 9100)),
                     seed_leakage_check_ok=True, notes="ok")
    assert r.verdict == "VERIFIED"
```

- [ ] **Step 2: Ensure schemas exist as stubs**

Each owner creates their schema file with the content from §2 of this plan. Run:

```bash
pytest tests/test_contracts.py -v
```

Expected: 3 tests pass (once schema files are committed by their owners).

- [ ] **Step 3: Commit**

```bash
git add tests/test_contracts.py
git commit -m "test: add contract schema smoke tests"
```

### Task M0.3: Create skeleton `autoqec/` package

**Files:**
- Create: `autoqec/__init__.py`, `autoqec/envs/__init__.py`, `autoqec/runner/__init__.py`, `autoqec/eval/__init__.py`, `autoqec/decoders/__init__.py`, `autoqec/orchestration/__init__.py`, `autoqec/agents/__init__.py`, `autoqec/llm/__init__.py`, `autoqec/tools/__init__.py`, `autoqec/logging/__init__.py`, `autoqec/pareto/__init__.py`, `autoqec/example_db/__init__.py`
- Create: `pyproject.toml`, `.gitignore` additions, `cli/__init__.py`, `cli/autoqec.py`

- [ ] **Step 1: Make package dirs and empty `__init__.py`**

```bash
mkdir -p autoqec/{envs,runner,eval,decoders/baselines,orchestration,agents,llm,tools,logging,pareto,example_db,envs/builtin} cli circuits runs tests demos
touch autoqec/__init__.py \
      autoqec/{envs,runner,eval,decoders,decoders/baselines,orchestration,agents,llm,tools,logging,pareto,example_db}/__init__.py \
      cli/__init__.py \
      tests/__init__.py
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
# pyproject.toml
[project]
name = "autoqec"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "torch",
  "stim",
  "sinter",
  "pymatching",
  "ldpc",
  "pydantic>=2",
  "click",
  "pyyaml",
  "numpy",
  "scipy",
  "fvcore",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-xdist", "ruff"]

[project.scripts]
autoqec = "cli.autoqec:main"

[tool.pytest.ini_options]
markers = [
  "integration: tests requiring GPU / model weights / long wall clock",
]
```

- [ ] **Step 3: Create minimal `cli/autoqec.py`**

```python
# cli/autoqec.py
import click

@click.group()
def main():
    """AutoQEC command-line interface."""

@main.command()
@click.argument("env_yaml")
@click.option("--rounds", type=int, default=10)
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option("--no-llm", is_flag=True, help="Skip LLM subagents (stub mode)")
def run(env_yaml, rounds, profile, no_llm):
    click.echo(f"[stub] would run {env_yaml} for {rounds} rounds (profile={profile}, no_llm={no_llm})")

@main.command()
@click.argument("round_dir")
def verify(round_dir):
    click.echo(f"[stub] would verify {round_dir}")

@main.command(name="review-log")
@click.argument("run_dir")
def review_log(run_dir):
    click.echo(f"[stub] would review {run_dir}/log.md")

@main.command()
@click.argument("run_dir")
def diagnose(run_dir):
    click.echo(f"[stub] would diagnose {run_dir}")

@main.command(name="add-env")
@click.option("--out", required=True)
def add_env(out):
    click.echo(f"[stub] would write env YAML to {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke test the CLI**

```bash
pip install -e '.[dev]'
python -m cli.autoqec --help
```

Expected: prints help with 5 subcommands.

- [ ] **Step 5: Commit**

```bash
git add autoqec/ cli/ pyproject.toml tests/__init__.py
git commit -m "chore: scaffold autoqec package + CLI stubs"
```

## 9. Self-review checklist

- [x] **Spec coverage**: every §3 goal G1–G7 maps to a phase milestone (G1→M3+M4, G2→M2 surface + Phase 3 bb72, G3→Phase 3 skills, G4→Phase 3 demos, G5→Phase 3 Demo 4, G6→M3 stub mode, G7→Phase 1 machine_state tool)
- [x] **No placeholders**: all schemas have concrete pydantic code; no "TBD"
- [x] **Type consistency**: `EnvSpec.name` matches `RunnerConfig.env_name` lookup; `RoundMetrics` keys align with what `independent_eval` consumes
- [x] **Contract frozen before scaffolds**: Phase 0 gate enforces this

---

## 10. Execution handoff

**Plan saved to `docs/superpowers/plans/2026-04-21-autoqec-master.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per Phase (Phase 0 first; then 3 subagents in parallel for Phase 1; then integration).

**2. Inline Execution** — Three owners open their respective plan files and execute the tasks themselves using `superpowers:executing-plans`.

**Which approach?**
