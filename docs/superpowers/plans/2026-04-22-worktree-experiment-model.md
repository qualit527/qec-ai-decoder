# Worktree-Based Experiment Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement §15 of the AutoQEC design spec — each research round gets its own git branch + optional `.worktrees/<id>/` checkout; Pareto members are the complete non-dominated set of VERIFIED branches; compose rounds test `git merge parent-A parent-B` as a first-class scientific probe.

**Architecture:** Additive schema layer + new orchestration modules (`worktree.py`, `fork_graph.py`, `subprocess_runner.py`) + surgical updates to existing orchestration / CLI / agent-prompt files. Backward-compatible on the `code_cwd=None` path: Lin's Runner, Demo 2, `cli/autoqec.py run --no-llm`, and 12 existing tests continue to pass unchanged.

**Tech Stack:** Python 3.10+, pydantic v2, pytest, subprocess + git CLI (no libgit2), Stim + PyTorch (existing Runner), click (existing CLI).

**Spec reference:** `docs/superpowers/specs/2026-04-20-autoqec-design.md` §15 (v2.3, post-4-rounds-of-codex-review).

**Migration order** (mirrors §15.9.3 — land tasks in this order to keep CI green at every step):
1. Pydantic schemas (additive fields with sensible defaults)
2. Pareto algorithm in `round_recorder.py` (top-5 sort → non-dominated filter)
3. CLI flags on `cli/autoqec.py run-round`
4. Orchestration prompt shape (`memory.py` → fork_graph; `loop.py` → fork_from)
5. New worktree modules + their tests
6. Runner in-process guard + startup reconciliation
7. Agent-prompt `.md` and skill updates
8. Contract documents

---

## File structure

### New files

| Path | Responsibility | LOC |
|---|---|---|
| `autoqec/orchestration/worktree.py` | Git-worktree CRUD ported from `problem-reductions/scripts/pipeline_worktree.py`; adds `create_round_worktree`, `create_compose_worktree`, `cleanup_round_worktree` | ~230 |
| `autoqec/orchestration/fork_graph.py` | Assembles fork-graph JSON from `git branch --list` + `history.jsonl` + `pareto.json`; computes non-dominated set | ~170 |
| `autoqec/orchestration/subprocess_runner.py` | `run_round_in_subprocess(cfg)` shells out to `python -m cli.autoqec run-round --code-cwd <path> ...`, captures metrics.json, validates, returns `RoundMetrics` | ~100 |
| `autoqec/orchestration/reconcile.py` | Startup `git branch <-> history.jsonl` reconciliation per §15.10 | ~150 |
| `tests/test_worktree.py` | Temp-git-repo tests: create / compose / conflict / cleanup | ~180 |
| `tests/test_fork_graph.py` | Tree serialization, non-dominated filter, empty-run fallback | ~120 |
| `tests/test_subprocess_runner.py` | Dev-profile round in a worktree via `gnn_small.yaml` | ~90 |
| `tests/test_reconcile.py` | Empty-synthetic reap, committed-orphan quarantine, Pareto-commit check | ~150 |

### Modified files

| Path | Change |
|---|---|
| `autoqec/runner/schema.py` | `RunnerConfig` + `RoundMetrics` gain §15.7 fields + validator |
| `autoqec/eval/schema.py` | `VerifyReport` gains `branch`, `commit_sha`, `delta_vs_baseline_holdout`, `paired_eval_bundle_id` |
| `autoqec/agents/schemas.py` | `IdeatorResponse` gains `fork_from` (default `"baseline"`) + `compose_mode`; `CoderResponse` gains `commit_message` (Optional) |
| `autoqec/orchestration/round_recorder.py` | Pareto algorithm: top-5 sort → non-dominated filter; writes `pareto_preview.json` |
| `autoqec/orchestration/memory.py` | `l3_for_ideator` returns `fork_graph` instead of `last_5_hypotheses` |
| `autoqec/orchestration/loop.py` | `run_round_plan` accepts `fork_from`; Coder ctx gets `worktree_dir` |
| `autoqec/tools/machine_state.py` | Adds `active_worktrees: list[str]` |
| `autoqec/runner/runner.py` | Raise `RunnerCallPathError` when `code_cwd` is set in-process |
| `cli/autoqec.py` | `run-round` adds `--code-cwd`, `--branch`, `--fork-from`, `--compose-mode`, `--round-attempt-id` flags |
| `scripts/run_single_round.py` | Accepts `--fork-from`; creates worktree; passes `code_cwd` |
| `.claude/agents/autoqec-ideator.md` | Consumes fork_graph; emits `fork_from` + `compose_mode` |
| `.claude/agents/autoqec-coder.md` | Notes: cwd is a worktree; emits `commit_message` |
| `.claude/agents/autoqec-analyst.md` | Output carries `branch` + `commit_sha` |
| `.claude/skills/autoqec-run/SKILL.md` | Adds fork-from decision + compose subflow + conflict handling |
| `docs/contracts/interfaces.md` | All §15.7 schema additions |
| `docs/contracts/round_dir_layout.md` | New `round_N_pointer.json` schema + provenance requirements |
| `tests/test_orchestration_stub.py` | Updated to expect fork_graph in `l3_for_ideator` output |
| `tests/test_round_recorder.py` | Updated for non-dominated Pareto + `_PARETO_FIELDS` additions |
| `tests/test_loop_helpers.py` | Updated for `fork_from` parameter |
| `tests/test_run_single_round.py` | Updated for `--fork-from` flag |
| `tests/test_run_quick.py` | No change expected (no-LLM smoke path); touch only if break |
| `tests/test_e2e_handshake.py` | Updated to include `round_attempt_id` |

---

## Phase A — Pydantic schemas (migration step 1)

These are pure-additive changes: all new fields have defaults. Existing callers that don't set any new field continue to work. Land these first so later phases have the types available.

### Task 1: Extend `RunnerConfig` and `RoundMetrics`

**Files:**
- Modify: `autoqec/runner/schema.py`
- Test: `tests/test_runner_schema_worktree.py` (new)

- [ ] **Step 1: Write failing tests for new fields + validators**

```python
# tests/test_runner_schema_worktree.py
import pytest
from autoqec.runner.schema import RunnerConfig, RoundMetrics


def test_runner_config_accepts_legacy_form():
    cfg = RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir="/tmp/round_1",
    )
    assert cfg.code_cwd is None
    assert cfg.branch is None
    assert cfg.fork_from is None


def test_runner_config_rejects_code_cwd_without_branch():
    with pytest.raises(ValueError, match="branch is required"):
        RunnerConfig(
            env_name="surface_d5_depol",
            predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
            training_profile="dev",
            seed=0,
            round_dir="/tmp/round_1",
            code_cwd="/abs/path/.worktrees/exp-x",
        )


def test_runner_config_rejects_compose_without_mode():
    with pytest.raises(ValueError, match="compose_mode is required"):
        RunnerConfig(
            env_name="surface_d5_depol",
            predecoder_config={},
            training_profile="dev",
            seed=0,
            round_dir="/tmp/round_1",
            fork_from=["exp/.../02-a", "exp/.../04-b"],
        )


def test_round_metrics_requires_round_attempt_id_on_worktree_path():
    with pytest.raises(ValueError, match="round_attempt_id"):
        RoundMetrics(
            status="ok",
            branch="exp/20260422-140000/01-small",
            commit_sha="abc123",
        )


def test_round_metrics_requires_commit_sha_when_branch_set():
    with pytest.raises(ValueError, match="commit_sha is required"):
        RoundMetrics(
            status="ok",
            branch="exp/20260422-140000/01-small",
            round_attempt_id="8b4f2c1e-9d3a-4f1b-b2e7-56f0ab7c3def",
        )


def test_round_metrics_compose_conflict_rejects_branch():
    with pytest.raises(ValueError, match="compose_conflict rows must have branch=None"):
        RoundMetrics(
            status="compose_conflict",
            branch="exp/20260422-140000/12-compose",
            round_attempt_id="8b4f2c1e-9d3a-4f1b-b2e7-56f0ab7c3def",
        )


def test_round_metrics_mutually_exclusive_ids():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoundMetrics(
            status="orphaned_branch",
            round_attempt_id="8b4f2c1e-9d3a-4f1b-b2e7-56f0ab7c3def",
            reconcile_id="c3def-8b4f2c1e-abc",
        )


def test_round_metrics_legacy_in_process_path_still_validates():
    # No branch / fork_from / round_attempt_id — existing Lin-era callers.
    m = RoundMetrics(status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4, delta_ler=5e-4)
    assert m.branch is None
    assert m.status == "ok"
```

- [ ] **Step 2: Run tests — expect all to fail**

```bash
pytest tests/test_runner_schema_worktree.py -v
```

Expected: 8 tests FAIL (`code_cwd` / `branch` / `fork_from` / etc. don't exist).

- [ ] **Step 3: Implement schema additions + validators**

```python
# autoqec/runner/schema.py — append / modify as shown
from typing import Literal, Optional, Union
from pydantic import BaseModel, model_validator


class RunnerConfig(BaseModel):
    # === existing fields ===
    env_name: str
    predecoder_config: dict
    training_profile: Literal["dev", "prod"] = "dev"
    seed: int = 0
    round_dir: str

    # === §15 additions ===
    code_cwd:            Optional[str] = None
    branch:              Optional[str] = None
    fork_from:           Optional[Union[str, list[str]]] = None
    fork_from_canonical: Optional[str] = None
    fork_from_ordered:   Optional[list[str]] = None
    compose_mode:        Optional[Literal["pure", "with_edit"]] = None

    @model_validator(mode="after")
    def _worktree_fields_consistent(self):
        if self.code_cwd is not None and self.branch is None:
            raise ValueError("branch is required when code_cwd is set")
        if isinstance(self.fork_from, list) and self.compose_mode is None:
            raise ValueError("compose_mode is required for compose rounds (fork_from is a list)")
        return self


class RoundMetrics(BaseModel):
    # === existing fields ===
    status: Literal[
        "ok", "killed_by_safety", "compile_error", "train_error",
        "compose_conflict",            # §15.6 — no commit, no branch
        "orphaned_branch",             # §15.10 reconciliation — commit may exist, no live loop ran
        "branch_manually_deleted",     # §15.10 — follow-up marker
    ]
    status_reason:       Optional[str] = None
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

    # === §15 additions ===
    round_attempt_id:    Optional[str] = None
    reconcile_id:        Optional[str] = None
    branch:              Optional[str] = None
    commit_sha:          Optional[str] = None
    fork_from:           Optional[Union[str, list[str]]] = None
    fork_from_canonical: Optional[str] = None
    fork_from_ordered:   Optional[list[str]] = None
    compose_mode:        Optional[Literal["pure", "with_edit"]] = None
    delta_vs_parent:     Optional[float] = None
    parent_ler:          Optional[float] = None
    conflicting_files:   Optional[list[str]] = None
    train_seed:          Optional[int] = None

    @model_validator(mode="after")
    def _provenance_integrity(self):
        is_worktree_row = (
            self.branch is not None
            or self.fork_from is not None
            or self.status in ("compose_conflict", "orphaned_branch", "branch_manually_deleted")
        )
        if is_worktree_row and self.round_attempt_id is None and self.reconcile_id is None:
            raise ValueError(
                "worktree-path rows need round_attempt_id (normal) or reconcile_id (startup-reconstructed)"
            )
        if self.round_attempt_id is not None and self.reconcile_id is not None:
            raise ValueError("round_attempt_id and reconcile_id are mutually exclusive")
        if self.branch is not None and self.commit_sha is None:
            raise ValueError("commit_sha is required whenever branch is set")
        if self.status == "compose_conflict" and (self.branch is not None or self.commit_sha is not None):
            raise ValueError("compose_conflict rows must have branch=None and commit_sha=None")
        return self
```

- [ ] **Step 4: Run tests — expect all to pass**

```bash
pytest tests/test_runner_schema_worktree.py -v
```

Expected: 8 tests PASS.

- [ ] **Step 5: Run full test suite to confirm no existing test breaks**

```bash
pytest tests/ -m "not integration" -v
```

Expected: all previously-passing tests still pass; the 8 new tests also pass.

- [ ] **Step 6: Commit**

```bash
git add autoqec/runner/schema.py tests/test_runner_schema_worktree.py
git commit -m "feat(schema): RunnerConfig + RoundMetrics worktree fields (§15.7)"
```

---

### Task 2: Extend `VerifyReport`

**Files:**
- Modify: `autoqec/eval/schema.py`
- Test: `tests/test_verify_report_worktree.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_verify_report_worktree.py
import pytest
from autoqec.eval.schema import VerifyReport


def test_verify_report_legacy_form():
    r = VerifyReport(
        verdict="VERIFIED",
        ler_holdout=1e-4,
        ler_holdout_ci=(5e-5, 2e-4),
        delta_ler_holdout=5e-5,
        ler_shuffled=1e-3,
        ablation_sanity_ok=True,
        holdout_seeds_used=[9000, 9001, 9002],
        seed_leakage_check_ok=True,
        notes="",
    )
    assert r.branch is None
    assert r.commit_sha is None


def test_verify_report_with_branch_requires_commit_sha():
    with pytest.raises(ValueError, match="commit_sha is required"):
        VerifyReport(
            verdict="VERIFIED",
            ler_holdout=1e-4,
            ler_holdout_ci=(5e-5, 2e-4),
            delta_ler_holdout=5e-5,
            ler_shuffled=1e-3,
            ablation_sanity_ok=True,
            holdout_seeds_used=[9000],
            seed_leakage_check_ok=True,
            notes="",
            branch="exp/20260422/02-a",
        )


def test_verify_report_paired_bundle_id_accepted():
    r = VerifyReport(
        verdict="VERIFIED",
        ler_holdout=1e-4,
        ler_holdout_ci=(5e-5, 2e-4),
        delta_ler_holdout=5e-5,
        ler_shuffled=1e-3,
        ablation_sanity_ok=True,
        holdout_seeds_used=[9000],
        seed_leakage_check_ok=True,
        notes="",
        branch="exp/20260422/02-a",
        commit_sha="abc123",
        delta_vs_baseline_holdout=5e-5,
        paired_eval_bundle_id="bundle-20260422-1500",
    )
    assert r.paired_eval_bundle_id == "bundle-20260422-1500"
```

- [ ] **Step 2: Run tests — expect fail**

```bash
pytest tests/test_verify_report_worktree.py -v
```

- [ ] **Step 3: Implement additions**

```python
# autoqec/eval/schema.py — add these fields + validator to VerifyReport
from typing import Optional
from pydantic import model_validator

class VerifyReport(BaseModel):
    # === existing fields unchanged ===
    verdict: Literal["VERIFIED", "SUSPICIOUS", "FAILED"]
    ler_holdout: float
    ler_holdout_ci: tuple[float, float]
    delta_ler_holdout: float
    ler_shuffled: float
    ablation_sanity_ok: bool
    holdout_seeds_used: list[int]
    seed_leakage_check_ok: bool
    notes: str

    # === §15 additions ===
    branch:                    Optional[str] = None
    commit_sha:                Optional[str] = None
    delta_vs_baseline_holdout: Optional[float] = None
    paired_eval_bundle_id:     Optional[str] = None

    @model_validator(mode="after")
    def _branch_needs_commit(self):
        if self.branch is not None and self.commit_sha is None:
            raise ValueError("commit_sha is required whenever branch is set")
        return self
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_verify_report_worktree.py -v
pytest tests/ -m "not integration" -v
```

Expected: new tests PASS; nothing else breaks.

- [ ] **Step 5: Commit**

```bash
git add autoqec/eval/schema.py tests/test_verify_report_worktree.py
git commit -m "feat(schema): VerifyReport adds branch/commit_sha/paired bundle (§15.7)"
```

---

### Task 3: Extend agent response schemas

**Files:**
- Modify: `autoqec/agents/schemas.py`
- Test: `tests/test_agent_schemas_worktree.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_agent_schemas_worktree.py
import pytest
from autoqec.agents.schemas import IdeatorResponse, CoderResponse


def test_ideator_legacy_response_gets_baseline_default():
    # Existing responses on main don't emit fork_from; default should be "baseline".
    r = IdeatorResponse(
        hypothesis="try gated MLP",
        expected_delta_ler=1e-4,
        expected_cost_s=600,
        rationale="prior rounds show plateau",
    )
    assert r.fork_from == "baseline"
    assert r.compose_mode is None


def test_ideator_accepts_compose_fork_from():
    r = IdeatorResponse(
        hypothesis="compose 02+04",
        expected_delta_ler=5e-4,
        expected_cost_s=900,
        rationale="test compositionality",
        fork_from=["exp/.../02-a", "exp/.../04-b"],
        compose_mode="pure",
    )
    assert isinstance(r.fork_from, list)
    assert r.compose_mode == "pure"


def test_coder_commit_message_optional():
    # Legacy in-process Coder responses should still validate.
    r = CoderResponse(dsl_config={"type": "gnn"}, tier="1", rationale="baseline GNN")
    assert r.commit_message is None
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_agent_schemas_worktree.py -v
```

- [ ] **Step 3: Implement additions**

```python
# autoqec/agents/schemas.py — modify IdeatorResponse and CoderResponse
from typing import Literal, Optional, Union
from pydantic import BaseModel, ConfigDict


class IdeatorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str
    expected_delta_ler: float
    expected_cost_s: int
    rationale: str
    dsl_hint: Optional[dict] = None

    # §15 additions — fork_from defaults to "baseline" so legacy responses validate.
    fork_from: Union[str, list[str]] = "baseline"
    compose_mode: Optional[Literal["pure", "with_edit"]] = None


class CoderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dsl_config: dict
    tier: Literal["1", "2"]
    rationale: str

    # §15 addition — Coder sets this on the worktree path; Optional so legacy works.
    commit_message: Optional[str] = None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_agent_schemas_worktree.py tests/test_orchestration_stub.py -v
pytest tests/ -m "not integration" -v
```

Expected: new 3 pass; existing agent-schema tests still pass.

- [ ] **Step 5: Commit**

```bash
git add autoqec/agents/schemas.py tests/test_agent_schemas_worktree.py
git commit -m "feat(schema): IdeatorResponse fork_from + CoderResponse commit_message"
```

---

## Phase B — Pareto algorithm (migration step 2)

### Task 4: Non-dominated Pareto filter in `round_recorder.py`

**Files:**
- Modify: `autoqec/orchestration/round_recorder.py`
- Modify: `tests/test_round_recorder.py`

- [ ] **Step 1: Read existing round_recorder.py to understand current shape**

```bash
cat autoqec/orchestration/round_recorder.py
```

Expected finding: `_PARETO_CAP = 5`, `_PARETO_FIELDS = ("round", "delta_ler", "flops_per_syndrome", "n_params", "checkpoint_path", "hypothesis")`, sort-then-truncate logic.

- [ ] **Step 2: Write failing tests for non-dominated behavior**

Append to `tests/test_round_recorder.py`:

```python
# tests/test_round_recorder.py — add these tests
import json
from pathlib import Path

import pytest

from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round


def _admit(mem, **overrides):
    """Helper: simulate a VERIFIED candidate admission."""
    base = {
        "status": "ok",
        "hypothesis": "h",
        "verdict": "candidate",
        "delta_ler": 1e-4,
        "flops_per_syndrome": 100_000,
        "n_params": 50_000,
        "checkpoint_path": "runs/foo/round_1/checkpoint.pt",
        "round": 1,
        "branch": None,
        "commit_sha": None,
        "round_attempt_id": None,
    }
    base.update(overrides)
    record_round(mem, round_metrics=base, verify_verdict="VERIFIED")


def test_pareto_keeps_dominated_points_no_more(tmp_path):
    # Three points: a and b are non-dominated, c is dominated by a.
    mem = RunMemory(tmp_path)
    _admit(mem, round=1, delta_ler=4e-4, flops_per_syndrome=200_000, n_params=40_000,
           branch="exp/t/01-a", commit_sha="a1", round_attempt_id="u1")
    _admit(mem, round=2, delta_ler=2e-4, flops_per_syndrome=50_000, n_params=20_000,
           branch="exp/t/02-b", commit_sha="b1", round_attempt_id="u2")
    _admit(mem, round=3, delta_ler=1e-4, flops_per_syndrome=300_000, n_params=60_000,
           branch="exp/t/03-c", commit_sha="c1", round_attempt_id="u3")
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    branches = {row["branch"] for row in pareto}
    assert branches == {"exp/t/01-a", "exp/t/02-b"}  # c is dominated


def test_pareto_has_no_size_cap(tmp_path):
    # Admit 7 distinct non-dominated points (vary the axes to avoid dominance).
    mem = RunMemory(tmp_path)
    for i in range(7):
        _admit(
            mem,
            round=i + 1,
            delta_ler=1e-4 + i * 1e-5,           # higher delta = better
            flops_per_syndrome=400_000 - i * 50_000,  # lower flops = better
            n_params=10_000 + i * 10_000,        # lower params = better (tradeoff)
            branch=f"exp/t/{i+1:02d}-x",
            commit_sha=f"sha{i}",
            round_attempt_id=f"u{i}",
        )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert len(pareto) == 7  # no truncation to 5


def test_pareto_preview_is_top_5_by_delta(tmp_path):
    mem = RunMemory(tmp_path)
    for i in range(7):
        _admit(
            mem,
            round=i + 1,
            delta_ler=1e-4 + i * 1e-5,
            flops_per_syndrome=400_000 - i * 50_000,
            n_params=10_000 + i * 10_000,
            branch=f"exp/t/{i+1:02d}-x",
            commit_sha=f"sha{i}",
            round_attempt_id=f"u{i}",
        )
    preview = json.loads((tmp_path / "pareto_preview.json").read_text())
    assert len(preview) == 5
    deltas = [row["delta_ler"] for row in preview]
    assert deltas == sorted(deltas, reverse=True)
```

- [ ] **Step 3: Run tests — expect fail**

```bash
pytest tests/test_round_recorder.py -v -k "pareto_keeps_dominated or pareto_has_no_size_cap or pareto_preview_is_top"
```

- [ ] **Step 4: Implement non-dominated filter**

```python
# autoqec/orchestration/round_recorder.py — replace the Pareto maintenance block
import json
from autoqec.orchestration.memory import RunMemory

_PARETO_PREVIEW_CAP = 5

_PARETO_FIELDS = (
    "round",
    "round_attempt_id",
    "branch",
    "commit_sha",
    "fork_from",
    "fork_from_canonical",
    "delta_ler",
    "flops_per_syndrome",
    "n_params",
    "checkpoint_path",
    "hypothesis",
)


def _dominates(a: dict, b: dict) -> bool:
    """Return True iff candidate `a` dominates `b` on (+delta_ler, -flops, -n_params).

    `a` dominates `b` iff `a` is at least as good on every axis AND strictly
    better on at least one.
    """
    a_d, b_d = float(a.get("delta_ler") or 0), float(b.get("delta_ler") or 0)
    a_f, b_f = int(a.get("flops_per_syndrome") or 0), int(b.get("flops_per_syndrome") or 0)
    a_p, b_p = int(a.get("n_params") or 0), int(b.get("n_params") or 0)
    at_least_as_good = (a_d >= b_d) and (a_f <= b_f) and (a_p <= b_p)
    strictly_better = (a_d > b_d) or (a_f < b_f) or (a_p < b_p)
    return at_least_as_good and strictly_better


def _non_dominated_merge(front: list[dict], candidate: dict) -> list[dict]:
    """Admit `candidate` to `front` using Pareto dominance.

    - Drop any existing member dominated by `candidate`.
    - Reject `candidate` if any existing member dominates it.
    - Otherwise append.
    """
    for existing in front:
        if _dominates(existing, candidate):
            return front  # candidate is dominated; reject
    pruned = [p for p in front if not _dominates(candidate, p)]
    pruned.append(candidate)
    return pruned


def _pareto_row(metrics: dict) -> dict:
    return {k: metrics.get(k) for k in _PARETO_FIELDS}


def _write_preview(run_dir, front: list[dict]) -> None:
    preview = sorted(front, key=lambda r: -float(r.get("delta_ler") or 0))[:_PARETO_PREVIEW_CAP]
    (run_dir / "pareto_preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def record_round(
    mem: RunMemory,
    round_metrics: dict,
    verify_verdict: str | None = None,
) -> None:
    """End-of-round bookkeeping: history row + Pareto update + log line."""
    mem.append_round(round_metrics)
    mem.append_log(f"- round {round_metrics.get('round')}: status={round_metrics.get('status')}")

    if verify_verdict != "VERIFIED":
        return

    front = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
    candidate = _pareto_row(round_metrics)
    front = _non_dominated_merge(front, candidate)
    mem.update_pareto(front)
    _write_preview(mem.run_dir, front)
```

- [ ] **Step 5: Run tests — expect new 3 pass, old recorder tests still pass**

```bash
pytest tests/test_round_recorder.py -v
```

Expected: all PASS. If any old test expects truncation-to-5, update it to expect non-dominated behavior (it was testing the wrong thing).

- [ ] **Step 6: Commit**

```bash
git add autoqec/orchestration/round_recorder.py tests/test_round_recorder.py
git commit -m "feat(pareto): non-dominated filter + pareto_preview.json (§15.2)"
```

---

## Phase C — CLI flags (migration step 3)

### Task 5: `run-round` accepts worktree flags

**Files:**
- Modify: `cli/autoqec.py`
- Test: `tests/test_cli_run_round_worktree.py` (new)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli_run_round_worktree.py
import json
import subprocess
import sys
from pathlib import Path


def test_run_round_legacy_positional_still_works(tmp_path):
    """Lin's positional form must keep working."""
    env = Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute()
    cfg = Path("autoqec/example_db/gnn_small.yaml").absolute()
    out = tmp_path / "round_1"
    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "run-round",
         str(env), str(cfg), str(out), "--profile", "dev"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0
    metrics = json.loads(result.stdout)
    assert metrics["status"] in ("ok", "killed_by_safety")


def test_run_round_accepts_worktree_flags_without_running():
    """The --help page lists the new flags so callers can discover them."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "run-round", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    for flag in ("--code-cwd", "--branch", "--fork-from", "--compose-mode", "--round-attempt-id"):
        assert flag in result.stdout, f"missing flag {flag}"
```

- [ ] **Step 2: Run tests — first PASSES (legacy path unchanged), second FAILS (new flags missing)**

```bash
pytest tests/test_cli_run_round_worktree.py -v
```

- [ ] **Step 3: Add flags**

Locate the `@main.command(name="run-round")` block in `cli/autoqec.py` and extend:

```python
# cli/autoqec.py — replace the run-round command body
@main.command(name="run-round")
@click.argument("env_yaml")
@click.argument("config_yaml")
@click.argument("round_dir")
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option("--code-cwd", default=None,
              help="Absolute path to a worktree checkout; when set, Runner runs in that cwd")
@click.option("--branch", default=None,
              help="Branch name (exp/<run_id>/<N>-<slug>); required when --code-cwd is set")
@click.option("--fork-from", default=None,
              help='Parent branch name, or JSON list like \'["exp/.../a", "exp/.../b"]\' for compose rounds')
@click.option("--compose-mode", type=click.Choice(["pure", "with_edit"]), default=None,
              help="Required when --fork-from is a list")
@click.option("--round-attempt-id", default=None,
              help="UUID minted at Ideator emit-time; required on the worktree path")
def run_round_cmd(env_yaml, config_yaml, round_dir, profile,
                  code_cwd, branch, fork_from, compose_mode, round_attempt_id):
    env = load_env_yaml(env_yaml)
    with open(config_yaml) as f:
        cfg_dict = yaml.safe_load(f)

    # Parse fork_from: JSON list → list[str]; bare string → str.
    parsed_fork_from = None
    if fork_from is not None:
        parsed_fork_from = json.loads(fork_from) if fork_from.strip().startswith("[") else fork_from

    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config=cfg_dict,
        training_profile=profile,
        seed=0,
        round_dir=round_dir,
        code_cwd=code_cwd,
        branch=branch,
        fork_from=parsed_fork_from,
        compose_mode=compose_mode,
    )

    # Worktree-path runs go through subprocess_runner; in-process runs use the legacy Runner.
    if code_cwd is not None:
        from autoqec.orchestration.subprocess_runner import run_round_in_subprocess
        metrics = run_round_in_subprocess(cfg, env, round_attempt_id=round_attempt_id)
    else:
        metrics = run_round(cfg, env)

    click.echo(metrics.model_dump_json(indent=2))
```

- [ ] **Step 4: Run tests — expect both pass (subprocess_runner stub tolerated)**

```bash
pytest tests/test_cli_run_round_worktree.py -v
```

If `subprocess_runner` doesn't exist yet (Task 9), create a stub now:

```python
# autoqec/orchestration/subprocess_runner.py — stub; full impl in Task 9
from autoqec.runner.schema import RunnerConfig, RoundMetrics
from autoqec.envs.schema import EnvSpec

def run_round_in_subprocess(cfg: RunnerConfig, env: EnvSpec,
                             round_attempt_id: str | None = None) -> RoundMetrics:
    raise NotImplementedError("implemented in Task 9; stub exists for import resolution")
```

- [ ] **Step 5: Commit**

```bash
git add cli/autoqec.py tests/test_cli_run_round_worktree.py autoqec/orchestration/subprocess_runner.py
git commit -m "feat(cli): run-round --code-cwd/--branch/--fork-from/--compose-mode/--round-attempt-id"
```

---

## Phase D — Orchestration prompt shape (migration step 4)

### Task 6: `memory.py` — `l3_for_ideator` returns fork_graph

**Files:**
- Modify: `autoqec/orchestration/memory.py`
- Modify: `tests/test_orchestration_stub.py`

- [ ] **Step 1: Write failing tests for the new shape**

Append to `tests/test_orchestration_stub.py`:

```python
def test_l3_for_ideator_returns_fork_graph(tmp_path):
    """The Ideator context exposes fork_graph per §15.4, not last_5_hypotheses."""
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path)
    mem.append_round({"round": 1, "status": "ok", "delta_ler": 4e-4,
                       "flops_per_syndrome": 180_000, "n_params": 42_000,
                       "branch": "exp/t/01-a", "commit_sha": "sha1",
                       "round_attempt_id": "u1", "fork_from": "baseline",
                       "hypothesis": "test"})

    ctx = mem.l3_for_ideator(
        env_spec={"name": "surface_d5"},
        kb_excerpt="(excerpt)",
        machine_state={"gpu": {}},
        run_id="t",
    )
    assert "fork_graph" in ctx
    assert "nodes" in ctx["fork_graph"]
    assert any(n.get("branch") == "exp/t/01-a" for n in ctx["fork_graph"]["nodes"])
    assert "baseline" in {n.get("branch") for n in ctx["fork_graph"]["nodes"]}
    # last_5_hypotheses is gone.
    assert "last_5_hypotheses" not in ctx
```

- [ ] **Step 2: Run tests — expect fail**

```bash
pytest tests/test_orchestration_stub.py::test_l3_for_ideator_returns_fork_graph -v
```

- [ ] **Step 3: Implement fork_graph construction in memory.py**

```python
# autoqec/orchestration/memory.py — replace l3_for_ideator
def l3_for_ideator(
    self,
    env_spec: dict,
    kb_excerpt: str,
    machine_state: dict,
    run_id: str | None = None,
) -> dict:
    """Assemble Ideator context including the full fork_graph (§15.4)."""
    from autoqec.orchestration.fork_graph import build_fork_graph

    history = self._load_history()
    pareto = json.loads(self.pareto_path.read_text(encoding="utf-8") or "[]")
    fork_graph = build_fork_graph(history=history, pareto=pareto, run_id=run_id or "")
    return {
        "env_spec": env_spec,
        "fork_graph": fork_graph,
        "knowledge_excerpts": kb_excerpt,
        "machine_state_hint": machine_state,
    }
```

- [ ] **Step 4: Add stub for fork_graph.build_fork_graph (full impl in Task 8)**

```python
# autoqec/orchestration/fork_graph.py — minimal stub; real impl in Task 8
from typing import Any

def build_fork_graph(history: list[dict], pareto: list[dict], run_id: str) -> dict[str, Any]:
    nodes = [{"branch": "baseline", "delta_vs_baseline": 0.0, "status": "baseline"}]
    for row in history:
        if row.get("branch"):
            nodes.append({
                "branch": row["branch"],
                "parent": row.get("fork_from", "baseline"),
                "delta_vs_parent": row.get("delta_ler"),
                "status": (row.get("status") or "ok").upper(),
                "hypothesis_1line": (row.get("hypothesis") or "")[:80],
                "on_pareto": any(p.get("branch") == row["branch"] for p in pareto),
            })
    return {
        "run_id": run_id,
        "nodes": nodes,
        "pareto_front": [p.get("branch") for p in pareto if p.get("branch")],
    }
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_orchestration_stub.py -v
```

Expected: new test PASSES; existing tests may need touch-ups for any reference to `last_5_hypotheses`.

- [ ] **Step 6: Sweep existing tests for stale references**

```bash
grep -rn "last_5_hypotheses" tests/ autoqec/
```

Any hit: replace the assertion with `fork_graph` checks matching the new shape. Update in-place. Do NOT restore the old field.

- [ ] **Step 7: Commit**

```bash
git add autoqec/orchestration/memory.py autoqec/orchestration/fork_graph.py tests/test_orchestration_stub.py
git commit -m "feat(memory): l3_for_ideator returns fork_graph (§15.4)"
```

---

### Task 7: `loop.py` threads `fork_from`

**Files:**
- Modify: `autoqec/orchestration/loop.py`
- Modify: `tests/test_loop_helpers.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_loop_helpers.py — add this
def test_run_round_plan_threads_fork_from(tmp_path):
    from autoqec.envs.schema import EnvSpec, CodeSpec, NoiseSpec, ConstraintsSpec, SeedPolicy
    from autoqec.orchestration.loop import run_round_plan

    env = EnvSpec(
        name="test",
        code=CodeSpec(type="stim_circuit", source="circuits/surface_d5.stim"),
        noise=NoiseSpec(type="depolarizing", p=[1e-3],
                         seed_policy=SeedPolicy()),
        constraints=ConstraintsSpec(),
        baseline_decoders=["pymatching"],
        classical_backend="mwpm",
    )
    plan = run_round_plan(
        env_spec=env, run_dir=tmp_path, round_idx=1,
        machine_state={"gpu": {}}, kb_excerpt="",
        dsl_schema_md="",
        fork_from="exp/t/02-a",  # NEW param
    )
    assert plan["fork_from"] == "exp/t/02-a"
```

- [ ] **Step 2: Run — expect fail**

```bash
pytest tests/test_loop_helpers.py::test_run_round_plan_threads_fork_from -v
```

- [ ] **Step 3: Modify `run_round_plan`**

```python
# autoqec/orchestration/loop.py — update signature
def run_round_plan(
    env_spec: "EnvSpec",
    run_dir: "Path | str",
    round_idx: int,
    machine_state: dict,
    kb_excerpt: str,
    dsl_schema_md: str,
    fork_from: str | list[str] = "baseline",  # NEW
) -> dict:
    run_dir = Path(run_dir)
    mem = RunMemory(run_dir)
    round_dir = run_dir / f"round_{round_idx}"

    ideator_ctx = mem.l3_for_ideator(
        env_spec=env_spec.model_dump(),
        kb_excerpt=kb_excerpt,
        machine_state=machine_state,
        run_id=run_dir.name,
    )
    return {
        "round_idx": round_idx,
        "round_dir": str(round_dir),
        "ideator_prompt": build_prompt("ideator", ideator_ctx),
        "dsl_schema_md": dsl_schema_md,
        "fork_from": fork_from,  # NEW: passed forward to worktree creation
    }
```

- [ ] **Step 4: Also thread worktree_dir to Coder ctx**

```python
# autoqec/orchestration/loop.py — update build_coder_prompt
def build_coder_prompt(
    hypothesis: dict,
    mem: RunMemory,
    dsl_schema_md: str,
    best_so_far: list[dict] | None = None,
    worktree_dir: str | None = None,  # NEW
) -> str:
    if best_so_far is None:
        pareto = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
        best_so_far = pareto[:3]
    ctx = mem.l3_for_coder(
        hypothesis=hypothesis,
        schema_md=dsl_schema_md,
        best_so_far=best_so_far,
    )
    if worktree_dir:
        ctx["worktree_dir"] = worktree_dir
    return build_prompt("coder", ctx)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_loop_helpers.py tests/test_orchestration_stub.py -v
```

- [ ] **Step 6: Commit**

```bash
git add autoqec/orchestration/loop.py tests/test_loop_helpers.py
git commit -m "feat(loop): thread fork_from + worktree_dir through run_round_plan"
```

---

## Phase E — Worktree modules (migration step 5)

### Task 8: `worktree.py` — ported from problem-reductions

**Files:**
- Create: `autoqec/orchestration/worktree.py`
- Create: `tests/test_worktree.py`

- [ ] **Step 1: Write failing tests using a temporary git repo**

```python
# tests/test_worktree.py
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "baseline.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    return repo


def test_create_round_worktree_returns_paths(git_repo):
    from autoqec.orchestration.worktree import create_round_worktree

    plan = create_round_worktree(
        repo_root=git_repo, run_id="20260422-140000", round_idx=1, slug="gated-mlp",
        fork_from="main",
    )
    assert Path(plan["worktree_dir"]).is_dir()
    assert plan["branch"] == "exp/20260422-140000/01-gated-mlp"


def test_create_compose_worktree_detects_conflict(git_repo):
    from autoqec.orchestration.worktree import create_round_worktree, create_compose_worktree

    # Build two divergent branches that both edit the same line of the same file.
    plan_a = create_round_worktree(git_repo, "t", 1, "a", fork_from="main")
    (Path(plan_a["worktree_dir"]) / "baseline.txt").write_text("A\n")
    subprocess.run(["git", "add", "."], cwd=plan_a["worktree_dir"], check=True)
    subprocess.run(["git", "commit", "-m", "a"], cwd=plan_a["worktree_dir"], check=True)

    plan_b = create_round_worktree(git_repo, "t", 2, "b", fork_from="main")
    (Path(plan_b["worktree_dir"]) / "baseline.txt").write_text("B\n")
    subprocess.run(["git", "add", "."], cwd=plan_b["worktree_dir"], check=True)
    subprocess.run(["git", "commit", "-m", "b"], cwd=plan_b["worktree_dir"], check=True)

    result = create_compose_worktree(
        repo_root=git_repo, run_id="t", round_idx=3, slug="compose-a-b",
        parents=[plan_a["branch"], plan_b["branch"]],
    )
    assert result["status"] == "compose_conflict"
    assert "baseline.txt" in result["conflicting_files"]
    # Worktree should be cleaned + branch deleted per §15.6.3
    assert not Path(result["worktree_dir"]).exists()
    branches = subprocess.check_output(
        ["git", "branch", "--list"], cwd=git_repo, text=True,
    )
    assert "compose-a-b" not in branches


def test_cleanup_round_worktree_removes_checkout_keeps_branch(git_repo):
    from autoqec.orchestration.worktree import create_round_worktree, cleanup_round_worktree

    plan = create_round_worktree(git_repo, "t", 1, "a", fork_from="main")
    cleanup_round_worktree(repo_root=git_repo, worktree_dir=plan["worktree_dir"])
    assert not Path(plan["worktree_dir"]).exists()
    branches = subprocess.check_output(
        ["git", "branch", "--list"], cwd=git_repo, text=True,
    )
    assert "exp/t/01-a" in branches
```

- [ ] **Step 2: Run tests — expect fail**

```bash
pytest tests/test_worktree.py -v
```

- [ ] **Step 3: Implement worktree.py**

```python
# autoqec/orchestration/worktree.py
"""Git worktree helpers for AutoQEC research rounds.

Ported from problem-reductions/scripts/pipeline_worktree.py (MIT) with
field renames (issue → round) and a new create_compose_worktree that
handles §15.6 compose rounds with conflict-as-failure semantics.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")[:40] or "round"


def _run_git(cwd: str | Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True)


def _run_git_checked(cwd: str | Path, *args: str) -> None:
    subprocess.check_output(["git", "-C", str(cwd), *args], stderr=subprocess.STDOUT)


def plan_round_worktree(
    repo_root: str | Path,
    *,
    run_id: str,
    round_idx: int,
    slug: str,
) -> dict:
    """Return the deterministic (branch, worktree_dir) paths without creating anything."""
    slug = _sanitize(slug)
    branch = f"exp/{run_id}/{round_idx:02d}-{slug}"
    worktree_dir = Path(repo_root) / ".worktrees" / f"exp-{run_id}-{round_idx:02d}-{slug}"
    return {
        "run_id": run_id, "round_idx": round_idx, "slug": slug,
        "branch": branch, "worktree_dir": str(worktree_dir),
    }


def create_round_worktree(
    repo_root: str | Path,
    run_id: str,
    round_idx: int,
    slug: str,
    fork_from: str = "main",
) -> dict:
    """Create .worktrees/exp-<run_id>-<N>-<slug>/ checked out on a new branch."""
    plan = plan_round_worktree(repo_root, run_id=run_id, round_idx=round_idx, slug=slug)
    Path(plan["worktree_dir"]).parent.mkdir(parents=True, exist_ok=True)
    _run_git_checked(repo_root, "worktree", "add", plan["worktree_dir"],
                      "-b", plan["branch"], fork_from)
    plan["fork_from"] = fork_from
    return plan


def create_compose_worktree(
    repo_root: str | Path,
    run_id: str,
    round_idx: int,
    slug: str,
    parents: list[str],
) -> dict:
    """Create a worktree from parents[0], then `git merge parents[1:]`.

    On conflict: abort merge, remove worktree, delete branch, return
    status=compose_conflict with the list of conflicting files. §15.6.3.
    """
    if len(parents) < 2:
        raise ValueError(f"compose requires ≥2 parents, got {parents}")

    plan = plan_round_worktree(repo_root, run_id=run_id, round_idx=round_idx, slug=slug)
    plan["parents"] = parents
    plan["fork_from_ordered"] = list(parents)
    plan["fork_from_canonical"] = "|".join(sorted(parents))

    Path(plan["worktree_dir"]).parent.mkdir(parents=True, exist_ok=True)
    _run_git_checked(repo_root, "worktree", "add", plan["worktree_dir"],
                      "-b", plan["branch"], parents[0])

    for parent in parents[1:]:
        proc = subprocess.run(
            ["git", "-C", plan["worktree_dir"], "merge", parent, "--no-edit"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            conflicting = _run_git(plan["worktree_dir"], "diff",
                                     "--name-only", "--diff-filter=U").split()
            subprocess.run(["git", "-C", plan["worktree_dir"], "merge", "--abort"],
                           capture_output=True)
            _run_git_checked(repo_root, "worktree", "remove", "--force", plan["worktree_dir"])
            _run_git_checked(repo_root, "branch", "-D", plan["branch"])
            plan.update(status="compose_conflict", conflicting_files=conflicting)
            return plan

    plan["status"] = "ok"
    return plan


def cleanup_round_worktree(
    repo_root: str | Path,
    worktree_dir: str | Path,
) -> None:
    """Remove the checkout; keep the branch. Idempotent."""
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_dir)],
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "worktree", "prune"],
        capture_output=True,
    )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_worktree.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqec/orchestration/worktree.py tests/test_worktree.py
git commit -m "feat(worktree): create_round_worktree + create_compose_worktree + cleanup (§15.6)"
```

---

### Task 9: `subprocess_runner.py` — shell-out Runner

**Files:**
- Modify: `autoqec/orchestration/subprocess_runner.py` (replaces stub)
- Create: `tests/test_subprocess_runner.py`

- [ ] **Step 1: Write failing integration-style test**

```python
# tests/test_subprocess_runner.py
import json
import pytest
from pathlib import Path

from autoqec.envs.schema import load_env_yaml
from autoqec.runner.schema import RunnerConfig


@pytest.mark.integration
def test_subprocess_runner_smoke(tmp_path, monkeypatch):
    """Run one dev-profile round inside a worktree via subprocess."""
    from autoqec.orchestration.worktree import create_round_worktree
    from autoqec.orchestration.subprocess_runner import run_round_in_subprocess

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    repo_root = Path(".").absolute()
    plan = create_round_worktree(repo_root, "smoke", 1, "gnn-small", fork_from="HEAD")

    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config=__import__("yaml").safe_load(
            Path("autoqec/example_db/gnn_small.yaml").read_text()),
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
        code_cwd=plan["worktree_dir"],
        branch=plan["branch"],
    )

    metrics = run_round_in_subprocess(cfg, env, round_attempt_id="test-uuid-1")
    assert metrics.status in ("ok", "killed_by_safety")
    assert metrics.round_attempt_id == "test-uuid-1"
```

- [ ] **Step 2: Run — expect fail (NotImplementedError stub)**

```bash
pytest tests/test_subprocess_runner.py -v --run-integration
```

- [ ] **Step 3: Implement subprocess_runner.py**

```python
# autoqec/orchestration/subprocess_runner.py
"""Shell-out Runner dispatch for §15.8 worktree mode."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from autoqec.envs.schema import EnvSpec
from autoqec.runner.schema import RunnerConfig, RoundMetrics


class RunnerSubprocessError(RuntimeError):
    pass


def run_round_in_subprocess(
    cfg: RunnerConfig,
    env: EnvSpec,
    round_attempt_id: str | None = None,
    timeout_s: int = 3000,
) -> RoundMetrics:
    """Run one round in a subprocess with cwd=cfg.code_cwd and PYTHONPATH pinned to the worktree.

    §15.8: Python's import cache can't hot-reload edited modules/*.py, so we must
    launch a fresh interpreter to pick up worktree-local code.
    """
    if cfg.code_cwd is None:
        raise ValueError("run_round_in_subprocess requires cfg.code_cwd")

    # Write the predecoder config to a temp file the subprocess can read.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg.predecoder_config, f)
        config_path = f.name

    # Env YAML on disk (the subprocess re-loads it; the in-memory EnvSpec is not enough).
    env_yaml_path = env.code.source  # env YAML location is taken from code.source anchor
    # Fall back to the builtin path for the env name if needed.
    env_file = Path("autoqec/envs/builtin") / f"{env.name}.yaml"
    if not env_file.exists():
        # Persist the EnvSpec to a temp YAML the subprocess can load.
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(env.model_dump(), f)
            env_file = Path(f.name)

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = cfg.code_cwd + os.pathsep + child_env.get("PYTHONPATH", "")

    argv = [
        sys.executable, "-m", "cli.autoqec", "run-round",
        str(env_file), str(config_path), cfg.round_dir,
        "--profile", cfg.training_profile,
        "--code-cwd", cfg.code_cwd,
        "--branch", cfg.branch or "",
    ]
    if cfg.fork_from is not None:
        argv += ["--fork-from", json.dumps(cfg.fork_from) if isinstance(cfg.fork_from, list) else cfg.fork_from]
    if cfg.compose_mode is not None:
        argv += ["--compose-mode", cfg.compose_mode]
    if round_attempt_id is not None:
        argv += ["--round-attempt-id", round_attempt_id]

    proc = subprocess.run(
        argv,
        cwd=cfg.code_cwd,
        env=child_env,
        capture_output=True, text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RunnerSubprocessError(
            f"subprocess runner failed: rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )

    # CLI prints metrics.json-shaped JSON to stdout.
    metrics_data = json.loads(proc.stdout)
    metrics_data.setdefault("round_attempt_id", round_attempt_id)
    metrics_data.setdefault("branch", cfg.branch)
    return RoundMetrics(**metrics_data)
```

- [ ] **Step 4: Run test (GPU-dependent; skip on CI)**

```bash
pytest tests/test_subprocess_runner.py -v --run-integration
```

If GPU unavailable: note in code-review that this test requires the integration marker; unit CI does not need to run it.

- [ ] **Step 5: Commit**

```bash
git add autoqec/orchestration/subprocess_runner.py tests/test_subprocess_runner.py
git commit -m "feat(subprocess_runner): shell-out Runner for §15.8 worktree mode"
```

---

### Task 10: Full `fork_graph.py`

**Files:**
- Modify: `autoqec/orchestration/fork_graph.py` (replaces stub from Task 6)
- Create: `tests/test_fork_graph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_fork_graph.py
from autoqec.orchestration.fork_graph import build_fork_graph, non_dominated


def test_empty_run_returns_baseline_only():
    g = build_fork_graph(history=[], pareto=[], run_id="t")
    assert len(g["nodes"]) == 1
    assert g["nodes"][0]["branch"] == "baseline"


def test_compose_conflict_node_included():
    history = [
        {"round": 12, "status": "compose_conflict",
         "fork_from": ["exp/t/02-a", "exp/t/04-b"],
         "fork_from_canonical": "exp/t/02-a|exp/t/04-b",
         "round_attempt_id": "u12",
         "conflicting_files": ["autoqec/decoders/modules/gnn.py"]},
    ]
    g = build_fork_graph(history=history, pareto=[], run_id="t")
    compose_nodes = [n for n in g["nodes"] if n.get("status") == "FAILED_COMPOSE"]
    assert len(compose_nodes) == 1
    assert compose_nodes[0].get("parents") == ["exp/t/02-a", "exp/t/04-b"]


def test_on_pareto_flag():
    history = [
        {"round": 1, "status": "ok", "branch": "exp/t/01-a", "commit_sha": "s1",
         "round_attempt_id": "u1", "fork_from": "baseline", "delta_ler": 1e-4,
         "hypothesis": "test"},
    ]
    pareto = [{"branch": "exp/t/01-a", "commit_sha": "s1", "delta_ler": 1e-4}]
    g = build_fork_graph(history=history, pareto=pareto, run_id="t")
    node = next(n for n in g["nodes"] if n.get("branch") == "exp/t/01-a")
    assert node["on_pareto"] is True


def test_non_dominated_filter():
    points = [
        {"delta_ler": 4e-4, "flops_per_syndrome": 200_000, "n_params": 40_000, "id": "a"},
        {"delta_ler": 2e-4, "flops_per_syndrome": 50_000,  "n_params": 20_000, "id": "b"},
        {"delta_ler": 1e-4, "flops_per_syndrome": 300_000, "n_params": 60_000, "id": "c"},  # dominated by a
    ]
    out = non_dominated(points)
    ids = {p["id"] for p in out}
    assert ids == {"a", "b"}
```

- [ ] **Step 2: Implement**

```python
# autoqec/orchestration/fork_graph.py — replace stub
"""Fork-graph assembly for the Ideator's L3 context (§15.4)."""
from __future__ import annotations

from typing import Any


def non_dominated(points: list[dict]) -> list[dict]:
    """Return the non-dominated subset over (+delta_ler, -flops, -n_params)."""
    def dominates(a: dict, b: dict) -> bool:
        ad, bd = float(a.get("delta_ler") or 0), float(b.get("delta_ler") or 0)
        af, bf = int(a.get("flops_per_syndrome") or 0), int(b.get("flops_per_syndrome") or 0)
        ap, bp = int(a.get("n_params") or 0), int(b.get("n_params") or 0)
        return (ad >= bd and af <= bf and ap <= bp) and (ad > bd or af < bf or ap < bp)
    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]


def build_fork_graph(history: list[dict], pareto: list[dict], run_id: str) -> dict[str, Any]:
    """Serialize the fork graph (§15.4) for the Ideator's L3 context.

    Includes baseline, every committed round, every compose_conflict, and Pareto flags.
    """
    pareto_branches = {p.get("branch") for p in pareto if p.get("branch")}
    nodes: list[dict] = [{
        "branch": "baseline", "delta_vs_baseline": 0.0,
        "ler": None, "flops": 0, "n_params": 0, "status": "baseline",
    }]

    for row in history:
        status = row.get("status") or ""
        if status == "compose_conflict":
            nodes.append({
                "branch": None,
                "round_attempt_id": row.get("round_attempt_id"),
                "parents": row.get("fork_from") if isinstance(row.get("fork_from"), list) else [],
                "fork_from_canonical": row.get("fork_from_canonical"),
                "status": "FAILED_COMPOSE",
                "failure_reason": f"git merge conflict in {', '.join(row.get('conflicting_files') or [])}",
                "hypothesis_1line": (row.get("hypothesis") or "")[:80],
            })
        elif row.get("branch"):
            nodes.append({
                "branch": row["branch"],
                "commit_sha": row.get("commit_sha"),
                "parent": row.get("fork_from", "baseline"),
                "delta_vs_parent": row.get("delta_vs_parent") or row.get("delta_ler"),
                "delta_vs_baseline": row.get("delta_vs_baseline") or row.get("delta_ler"),
                "ler": row.get("ler_predecoder"),
                "flops": row.get("flops_per_syndrome"),
                "params": row.get("n_params"),
                "status": status.upper(),
                "on_pareto": row["branch"] in pareto_branches,
                "hypothesis_1line": (row.get("hypothesis") or "")[:80],
                "failure_reason": row.get("status_reason"),
            })

    return {
        "run_id": run_id,
        "nodes": nodes,
        "pareto_front": [p.get("branch") for p in pareto if p.get("branch")],
    }
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_fork_graph.py tests/test_orchestration_stub.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/orchestration/fork_graph.py tests/test_fork_graph.py
git commit -m "feat(fork_graph): full §15.4 serialization + non_dominated helper"
```

---

## Phase F — Runner guard + reconciliation (migration step 6)

### Task 11: `run_round` raises on `code_cwd != None`

**Files:**
- Modify: `autoqec/runner/runner.py`
- Test: `tests/test_runner_guard.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_runner_guard.py
import pytest

def test_run_round_raises_when_code_cwd_set():
    from autoqec.runner.runner import run_round, RunnerCallPathError
    from autoqec.runner.schema import RunnerConfig
    from autoqec.envs.schema import load_env_yaml

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"type": "gnn", "output_mode": "soft_priors",
                            "gnn": {"layers": 1, "hidden_dim": 8, "message_fn": "mlp",
                                    "aggregation": "sum", "normalization": "none",
                                    "residual": False, "edge_features": []},
                            "head": "linear",
                            "training": {"learning_rate": 1e-3, "batch_size": 16,
                                          "epochs": 1, "loss": "bce", "profile": "dev"}},
        training_profile="dev", seed=0, round_dir="/tmp/r",
        code_cwd="/abs/.worktrees/x", branch="exp/t/1-a",
    )
    with pytest.raises(RunnerCallPathError):
        run_round(cfg, env)
```

- [ ] **Step 2: Implement guard**

Prepend to `autoqec/runner/runner.py`:

```python
class RunnerCallPathError(RuntimeError):
    """Raised when run_round is called in-process with cfg.code_cwd set.

    Worktree-path runs must go through autoqec.orchestration.subprocess_runner
    because Python's import cache would serve main's modules/*.py instead of
    the worktree's edited copies (§15.8).
    """
```

And at the top of `run_round`:

```python
def run_round(config: RunnerConfig, env_spec: EnvSpec, safety: RunnerSafety | None = None) -> RoundMetrics:
    if config.code_cwd is not None:
        raise RunnerCallPathError(
            "code_cwd is set — use autoqec.orchestration.subprocess_runner.run_round_in_subprocess "
            "instead of run_round (§15.8)"
        )
    # ... rest unchanged
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_runner_guard.py tests/test_runner_smoke.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/runner/runner.py tests/test_runner_guard.py
git commit -m "feat(runner): RunnerCallPathError guards in-process path from worktree misuse (§15.8)"
```

---

### Task 12: `reconcile.py` — startup reconciliation

**Files:**
- Create: `autoqec/orchestration/reconcile.py`
- Create: `tests/test_reconcile.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_reconcile.py
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def repo_with_run(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "baseline.txt").write_text("baseline\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    run_dir = repo / "runs" / "t"
    run_dir.mkdir(parents=True)
    (run_dir / "history.jsonl").write_text("")
    (run_dir / "pareto.json").write_text("[]")
    return repo, run_dir


def test_empty_synthetic_branch_is_reaped(repo_with_run):
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Create an empty synthetic branch with zero commits beyond main.
    subprocess.run(["git", "worktree", "add", ".worktrees/exp-t-01-x",
                    "-b", "exp/t/01-x", "main"], cwd=repo, check=True)
    subprocess.run(["git", "worktree", "remove", "--force", ".worktrees/exp-t-01-x"],
                   cwd=repo, check=True)
    # Branch exists; history is empty → B\H = {exp/t/01-x} and it's empty.
    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    reaped = [a for a in actions if a["kind"] == "reaped"]
    assert any("exp/t/01-x" in a["branch"] for a in reaped)
    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "exp/t/01-x" not in branches


def test_committed_orphan_is_quarantined(repo_with_run):
    from autoqec.orchestration.reconcile import reconcile_at_startup

    repo, run_dir = repo_with_run
    # Create a branch with a real commit (simulating a crash after step 5).
    subprocess.run(["git", "worktree", "add", ".worktrees/exp-t-02-y",
                    "-b", "exp/t/02-y", "main"], cwd=repo, check=True)
    wt = repo / ".worktrees" / "exp-t-02-y"
    (wt / "new.txt").write_text("work in progress\n")
    subprocess.run(["git", "add", "."], cwd=wt, check=True)
    subprocess.run(["git", "commit", "-m", "wip"], cwd=wt, check=True)
    subprocess.run(["git", "worktree", "remove", "--force", ".worktrees/exp-t-02-y"],
                   cwd=repo, check=True)

    actions = reconcile_at_startup(repo_root=repo, run_id="t", run_dir=run_dir)
    quarantined = [a for a in actions if a["kind"] == "quarantined"]
    assert any("exp/t/02-y" in a["original_branch"] for a in quarantined)

    branches = subprocess.check_output(["git", "branch", "--list"], cwd=repo, text=True)
    assert "quarantine/t/02-y" in branches
```

- [ ] **Step 2: Implement**

```python
# autoqec/orchestration/reconcile.py
"""§15.10 startup reconciliation.

Auto-heal empty synthetic branches; quarantine branches with real commits but
no history row. Pause only for ambiguous cases (no recoverable round_attempt_id,
or missing Pareto commit).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _run_git(cwd: str | Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True)


def _run_git_checked(cwd: str | Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True)


def _list_exp_branches(repo_root: Path, run_id: str) -> set[str]:
    out = _run_git(repo_root, "branch", "--list", f"exp/{run_id}/*")
    return {line.strip().lstrip("* ").strip() for line in out.splitlines() if line.strip()}


def _history_branches(run_dir: Path) -> set[str]:
    path = run_dir / "history.jsonl"
    if not path.exists():
        return set()
    branches = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("branch"):
            branches.add(row["branch"])
    return branches


def _is_empty_synthetic(repo_root: Path, branch: str) -> bool:
    """A branch is empty-synthetic iff its tip equals its merge-base with main."""
    try:
        tip = _run_git(repo_root, "rev-parse", branch).strip()
        base = _run_git(repo_root, "merge-base", branch, "main").strip()
        return tip == base
    except subprocess.CalledProcessError:
        return False  # if something's broken, don't treat as empty


def reconcile_at_startup(repo_root: str | Path, run_id: str, run_dir: str | Path) -> list[dict]:
    """Reconcile git-branch state vs history.jsonl per §15.10. Returns a list of actions taken."""
    repo_root = Path(repo_root)
    run_dir = Path(run_dir)
    actions: list[dict] = []

    _run_git_checked(repo_root, "worktree", "prune")

    B = _list_exp_branches(repo_root, run_id)
    H = _history_branches(run_dir)

    # 3. Branches without history rows.
    for b in B - H:
        if _is_empty_synthetic(repo_root, b):
            _run_git_checked(repo_root, "branch", "-D", b)
            actions.append({"kind": "reaped", "branch": b})
        else:
            # Quarantine: rename to quarantine/<run_id>/<remainder>
            remainder = b.removeprefix(f"exp/{run_id}/")
            new = f"quarantine/{run_id}/{remainder}"
            _run_git_checked(repo_root, "branch", "-m", b, new)
            actions.append({"kind": "quarantined", "original_branch": b, "quarantine_branch": new})
            # Emit an orphaned_branch history row (reconcile_id, no round_attempt_id).
            import uuid
            row = {
                "status": "orphaned_branch",
                "round_attempt_id": None,
                "reconcile_id": str(uuid.uuid4()),
                "branch": new,
                "commit_sha": _run_git(repo_root, "rev-parse", new).strip(),
                "status_reason": "branch had real commits but no history row at startup",
            }
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 4. History rows without live branches.
    for h in H - B:
        # Append a branch_manually_deleted follow-up marker.
        row = {
            "status": "branch_manually_deleted",
            "branch": h,
            "round_attempt_id": None,
            "reconcile_id": None,  # follow-up marker references the original row's attempt_id via status_reason
            "status_reason": f"branch {h} was listed in history but not found in git at startup",
        }
        # Follow-ups don't need an id; they reference by branch.
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            # Small shim: Optional[str] so the row validates without an id.
            row["reconcile_id"] = f"followup-{h.replace('/', '-')}"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        actions.append({"kind": "follow_up", "branch": h})

    # 5. Pareto-commit reachability.
    pareto = json.loads((run_dir / "pareto.json").read_text(encoding="utf-8") or "[]")
    for p in pareto:
        sha = p.get("commit_sha")
        if not sha:
            continue
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{sha}^{{commit}}"],
            capture_output=True,
        )
        if proc.returncode != 0:
            actions.append({"kind": "pause", "reason": f"Pareto commit {sha} unreachable",
                             "branch": p.get("branch")})

    return actions
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_reconcile.py -v
```

- [ ] **Step 4: Commit**

```bash
git add autoqec/orchestration/reconcile.py tests/test_reconcile.py
git commit -m "feat(reconcile): §15.10 startup reconciliation with auto-heal/pause policy"
```

---

## Phase G — Agent / skill updates (migration step 7)

### Task 13: Update `.claude/agents/autoqec-ideator.md`

**Files:**
- Modify: `.claude/agents/autoqec-ideator.md`

- [ ] **Step 1: Read the current file**

```bash
cat .claude/agents/autoqec-ideator.md
```

- [ ] **Step 2: Replace "Input" section to consume fork_graph**

Find the block describing inputs (likely mentions `last_5_hypotheses`, `pareto_front`) and replace with:

```markdown
# Inputs (provided in your prompt)
- `env_spec`: pydantic dump of the EnvSpec
- `fork_graph` (§15.4): the full tree of this run's branches — `nodes` (every round, including FAILED and compose_conflict) + `pareto_front` (list of currently-admitted branch names). Read this to decide `fork_from`.
- `knowledge_excerpts`: short excerpt from knowledge/DECODER_ROADMAP.md §5
- `machine_state_hint`: result of `machine_state(run_dir)` — includes `active_worktrees`
```

Update the **Output format** block to require `fork_from`:

```markdown
# Output format
Exactly one fenced JSON block:

```json
{
  "hypothesis": "<1 sentence what to try>",
  "fork_from": "baseline",
  "compose_mode": null,
  "expected_delta_ler": 5e-5,
  "expected_cost_s": 900,
  "rationale": "<why this over alternatives; reference prior rounds by branch name>",
  "dsl_hint": {"type": "gnn", "message_fn": "gated_mlp", "layers": 4}
}
```

`fork_from` values:
- `"baseline"` — new line of attack, fork from main
- `"exp/<run_id>/<N>-<slug>"` — stack on a prior VERIFIED branch (check `fork_graph.pareto_front` first)
- `["exp/.../A", "exp/.../B"]` — compose round; requires `compose_mode ∈ {"pure", "with_edit"}`
```

Add a **Hard rules** bullet:

```markdown
- Do not re-propose a `fork_from_canonical` that already has `status=FAILED_compose` in the fork graph.
- When proposing a compose round, check that both parents are VERIFIED (present in `fork_graph.pareto_front` or have `status=VERIFIED` nodes) — composing a FAILED with anything is wasted compute.
```

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/autoqec-ideator.md
git commit -m "docs(agents): Ideator consumes fork_graph + emits fork_from/compose_mode"
```

---

### Task 14: Update `.claude/agents/autoqec-coder.md`

**Files:**
- Modify: `.claude/agents/autoqec-coder.md`

- [ ] **Step 1: Add worktree awareness block**

Insert after the existing **Inputs** section:

```markdown
# Worktree awareness

When the orchestrator invokes you on the §15 worktree path, your cwd is
`.worktrees/exp-<run_id>-<N>-<slug>/`, which is an isolated git checkout
on branch `exp/<run_id>/<N>-<slug>`. Edits you make land in that branch;
a subsequent commit and Runner invocation happen automatically.

Most rounds are Tier-1 (DSL YAML only). For Tier-1:
- Write the config to the filename the orchestrator specifies (typically `config.yaml` in the worktree root, or a name under `autoqec/example_db/`).
- Do NOT edit `autoqec/decoders/modules/*.py` — Tier-1 uses the main-branch versions.

For Tier-2 (custom_fn), you may edit `autoqec/decoders/modules/*.py` within the worktree.
The subprocess Runner picks up your edits because `PYTHONPATH=<worktree>:$PYTHONPATH`.
```

Update the **Output format** to include `commit_message`:

```markdown
# Output format
```json
{
  "dsl_config": {...},
  "tier": "1",
  "rationale": "<why this shape>",
  "commit_message": "exp(<run_id>/<N>): <one-line description of the change>"
}
```
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/autoqec-coder.md
git commit -m "docs(agents): Coder worktree awareness + commit_message output"
```

---

### Task 15: Update `.claude/agents/autoqec-analyst.md`

**Files:**
- Modify: `.claude/agents/autoqec-analyst.md`

- [ ] **Step 1: Append branch/commit_sha to output format**

Replace the **Output** block with:

```markdown
# Output
```json
{
  "summary_1line": "<one line summarizing this round>",
  "verdict": "candidate",
  "next_hypothesis_seed": "<suggestion for Ideator>",
  "branch": "<carry from metrics.json if present>",
  "commit_sha": "<carry from metrics.json if present>"
}
```

The `branch` and `commit_sha` values come verbatim from `metrics.json`; do not invent them.
If `metrics.json` lacks them (legacy in-process path), emit `null` for both.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/agents/autoqec-analyst.md
git commit -m "docs(agents): Analyst output echoes branch + commit_sha"
```

---

### Task 16: Update `.claude/skills/autoqec-run/SKILL.md`

**Files:**
- Modify: `.claude/skills/autoqec-run/SKILL.md`

- [ ] **Step 1: Add fork-from decision + compose subflow**

Insert a new section before the existing Runner invocation step:

```markdown
## Fork-from decision (§15.2)

After the Ideator responds, read its `fork_from` field:

| `fork_from` value | Action |
|---|---|
| `"baseline"` | Call `create_round_worktree(repo_root, run_id, N, slug, fork_from="main")`. |
| `"exp/<run_id>/<M>-<slug>"` (string) | Call `create_round_worktree(..., fork_from=<that branch name>)`. |
| list of ≥2 branch names | **Compose round**: call `create_compose_worktree(repo_root, run_id, N, slug, parents=<list>)`. If it returns `status="compose_conflict"`: write a `history.jsonl` row via `record_round` with `status=compose_conflict`, `round_attempt_id=<UUID>`, `conflicting_files=<list>`, and skip to the next round (no Runner invocation). |

Mint `round_attempt_id` as `str(uuid.uuid4())` **before** calling the Ideator prompt is assembled, and pass it through every subsequent call (Coder prompt, Runner subprocess, Analyst prompt, record_round).

## Runner invocation (worktree path)

Use `subprocess_runner.run_round_in_subprocess(cfg, env, round_attempt_id=<uuid>)` with:
- `cfg.code_cwd = <worktree_dir>` (from the worktree plan dict)
- `cfg.branch = <branch>` (from the plan)
- `cfg.fork_from = <Ideator's fork_from value>`
- `cfg.compose_mode = <Ideator's compose_mode>` (only for compose rounds)

On success (metrics.status == "ok"), the subprocess has already committed the pointer file in the worktree. Proceed to Analyst dispatch.

On `status=compose_conflict` or any non-ok status, call `cleanup_round_worktree` (removes the checkout; branch persists unless it was a compose_conflict, in which case `create_compose_worktree` already deleted the branch).
```

- [ ] **Step 2: Add a note in the failure-handling section**

```markdown
## Compose conflict handling (§15.6.3)

If a compose round returns `status="compose_conflict"`:
- `record_round` writes the `compose_conflict` row BEFORE any cleanup
- The synthetic branch is already deleted by `create_compose_worktree`
- Ideator on the next round will see the `FAILED_compose` node in `fork_graph` and must not re-propose the same `fork_from_canonical` set
- Do NOT invoke `/verify-decoder` for compose_conflict rounds — there's no checkpoint
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/autoqec-run/SKILL.md
git commit -m "docs(skill): /autoqec-run fork-from decision + compose subflow (§15.3, §15.6)"
```

---

## Phase H — Contracts (migration step 8)

### Task 17: Update `docs/contracts/interfaces.md`

**Files:**
- Modify: `docs/contracts/interfaces.md`

- [ ] **Step 1: Copy the §15.7 schema blocks into the contract file**

Open `docs/contracts/interfaces.md` and add a new section at the end:

```markdown
---

## §15 Worktree additions (contract-change)

Added 2026-04-22. All fields are Optional on the legacy path; validators make them required on the worktree path.

### `RunnerConfig`
- `code_cwd: Optional[str]`
- `branch: Optional[str]`  (required when `code_cwd` set)
- `fork_from: Optional[Union[str, list[str]]]`  (list ⇒ compose round)
- `fork_from_canonical: Optional[str]`
- `fork_from_ordered: Optional[list[str]]`
- `compose_mode: Optional[Literal["pure", "with_edit"]]`  (required when `fork_from` is a list)

### `RoundMetrics`
- `round_attempt_id: Optional[str]`  (required on worktree path)
- `reconcile_id: Optional[str]`  (set only for reconciliation-synthetic rows)
- `branch`, `commit_sha` (paired; `commit_sha` required when `branch` set)
- `fork_from`, `fork_from_canonical`, `fork_from_ordered`
- `compose_mode`
- `delta_vs_parent`, `parent_ler`, `train_seed`, `status_reason`
- `conflicting_files: Optional[list[str]]` (set iff status=compose_conflict)
- `status` gains: `compose_conflict`, `orphaned_branch`, `branch_manually_deleted`

### `VerifyReport`
- `branch`, `commit_sha` (paired)
- `delta_vs_baseline_holdout: Optional[float]`
- `paired_eval_bundle_id: Optional[str]`

### `IdeatorResponse`
- `fork_from: Union[str, list[str]] = "baseline"`
- `compose_mode: Optional[Literal["pure", "with_edit"]]`

### `CoderResponse`
- `commit_message: Optional[str]`
```

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/interfaces.md
git commit -m "docs(contracts): §15 worktree schema additions (contract-change)"
```

---

### Task 18: Update `docs/contracts/round_dir_layout.md`

**Files:**
- Modify: `docs/contracts/round_dir_layout.md`

- [ ] **Step 1: Append the pointer-file schema and worktree artifacts**

Add at the end:

```markdown
---

## §15 Additions

### `round_N_pointer.json` (committed inside the worktree)

Provenance fields are all REQUIRED; consumers reject pointer files missing any.

```json
{
  "run_id": "<YYYYMMDD-HHMMSS>",
  "round_idx": <int>,
  "round_attempt_id": "<UUID>",
  "branch": "exp/<run_id>/<N>-<slug>",
  "commit_sha": "<full SHA>",
  "fork_from": "baseline" | "<branch>" | [<branches>],
  "fork_from_canonical": "<sorted|joined>",
  "fork_from_ordered": <list or null>,
  "provenance": {
    "env_yaml_sha256": "<sha256>",
    "dsl_config_sha256": "<sha256>",
    "requirements_fingerprint": "<short>",
    "repo_root_resolved": "<absolute path>"
  },
  "metrics_summary": {
    "delta_vs_parent": <float>,
    "flops_per_syndrome": <int>,
    "n_params": <int>,
    "status": "ok" | "killed_by_safety" | ...
  },
  "artifact_paths": {
    "checkpoint": "<absolute>",
    "metrics":    "<absolute>",
    "train_log":  "<absolute>"
  }
}
```

### `pareto.json` — complete non-dominated archive (NOT capped)

See `docs/superpowers/specs/2026-04-20-autoqec-design.md` §15.2 for the `round_recorder.py` update from top-5 sort to non-dominated filter.

### `pareto_preview.json` — derived top-5 (for L2 Ideator context)

Regenerated after every Pareto mutation; sorted by `-delta_vs_baseline_holdout`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/contracts/round_dir_layout.md
git commit -m "docs(contracts): round_N_pointer.json + pareto/preview split (§15.5)"
```

---

## Phase I — Verification

### Task 19: Full test suite green + ruff clean

- [ ] **Step 1: Run unit tests**

```bash
pytest tests/ -m "not integration" -v
```

Expected: all PASS. Any failure indicates a contract mismatch between phases — fix in place.

- [ ] **Step 2: Run ruff**

```bash
ruff check autoqec/ cli/ scripts/ tests/
```

Fix any issues with `ruff check --fix` where safe; otherwise edit by hand. Commit cleanup as `chore: ruff cleanup`.

- [ ] **Step 3: Run the no-LLM smoke path to prove legacy still works**

```bash
python scripts/run_quick.py --rounds 2 --profile dev
```

Expected: 2 rounds complete; `runs/<id>/history.jsonl` has 2 rows; `pareto.json` exists.

- [ ] **Step 4: Run the integration worktree smoke test (GPU required)**

```bash
pytest tests/test_subprocess_runner.py -v --run-integration
```

Expected: PASS on GPU host; skipped otherwise.

- [ ] **Step 5: Commit any cleanup**

```bash
git add -A
git commit -m "chore: final ruff cleanup + verified full-suite green"
```

---

### Task 20: Document the §15 rollout completion

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a bullet under "What's implemented"**

Find the existing feature/status table in README and add:

```markdown
| **F6** | Worktree-based experiment model (branches-as-Pareto; compose rounds; startup reconciliation) | Full team | **implemented** |
```

- [ ] **Step 2: Link to the spec**

Add a pointer in the "Architecture" or "Logical phases" section:

```markdown
Per-round isolation and the Pareto-as-branches model are specified in
[`spec §15`](docs/superpowers/specs/2026-04-20-autoqec-design.md#15-worktree-based-experiment-model).
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): announce §15 worktree rollout"
```

---

## Self-review checklist

- [x] **Spec coverage**: §15.1 motivation → Phase B (non-dominated Pareto) + §15.10 reconciliation; §15.2 data model → Tasks 1, 4, 10; §15.3 lifecycle → Tasks 7, 8, 9; §15.4 fork_graph → Task 10; §15.5 pointer file + scope → Task 18; §15.6 compose rounds → Task 8 (create_compose_worktree); §15.7 schema delta → Tasks 1, 2, 3; §15.8 runner paths → Tasks 9, 11; §15.9 migration order = Phases A–H in order; §15.10 safety + reconciliation → Tasks 11, 12; §15.11 MVP scope → whole plan.
- [x] **No placeholders**: every task has complete code blocks, exact commands, expected outputs. No "TBD", "similar to Task N" shortcuts.
- [x] **Type consistency**: `RunnerConfig` / `RoundMetrics` field names are identical across Tasks 1, 4, 5, 7, 9, 11. `fork_from` / `fork_from_canonical` / `fork_from_ordered` appear together in every schema and row example. `round_attempt_id` and `reconcile_id` handled identically in memory, reconcile, fork_graph.
- [x] **Ordering**: Tasks respect the §15.9.3 8-step migration — schemas land before callers break; reconciliation lands before agent prompts that might trigger crashes.
- [x] **Legacy compat**: every task that touches a legacy file adds a backward-compat test first (e.g. Task 5's `test_run_round_legacy_positional_still_works`).

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-22-worktree-experiment-model.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using superpowers:executing-plans, batch execution with checkpoints.

**Which approach?**
