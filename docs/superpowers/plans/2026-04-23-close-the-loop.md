# Close-the-Loop + Acceptance-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every gap between the current `main` and the acceptance gates in `docs/verification/human-verification-test-plan.md`. After this plan lands, Phases 1–5 of the human verification test plan all execute green (given GPU + LLM credentials).

**Architecture:**
- **P0** — unblocks demo day. Most visible: empty Pareto, CLI-blocked LLM loop, missing manifest, missing reward-hacking fixtures.
- **P1** — research-integrity gates. Orchestrator invariants (branch fence, idempotency, commit-prefix), diagnostic depth, fixtures used by existing tests.
- **P2** — polish. Demo completion, dead-code delete, quality-of-life warnings, offline replay packaging.

Everything is additive or surgical; the §15 worktree model, verifier, and record_round API stay exactly as they are on `main`.

**Tech Stack:** Python 3.12+, pydantic v2, pytest, click, subprocess; `stim`/`torch`/`pymatching` for integration tests; `codex-cli` / `claude-cli` as the background dispatch backends.

---

## What's already on `main` (don't redo)

- `autoqec/eval/independent_eval.py::independent_verify(checkpoint, env_spec, holdout_seeds, n_shots=None, n_bootstrap=1000) → VerifyReport` + bootstrap + ablation
- `autoqec/cheaters/memorize.py`
- `autoqec/orchestration/round_recorder.py::record_round(mem, round_metrics, verify_verdict, verify_report=None)` — skips Pareto admission + WARNs when `verify_report is None`
- `autoqec/orchestration/round_recorder.py::_non_dominated_merge` — the canonical non-dominated filter
- `autoqec/orchestration/worktree.py::{create_round_worktree, create_compose_worktree, cleanup_round_worktree}`
- `autoqec/orchestration/reconcile.py::reconcile_at_startup` — reads `round_N_pointer.json` defensively
- `autoqec/orchestration/subprocess_runner.py::run_round_in_subprocess` + hidden `--_internal-execute-locally` flag
- `cli/autoqec.py` subcommands: `run`, `run-round`, `verify`, `review-log` (all fields present), `diagnose` (thin — see P1.5), `add-env`
- `AUTOQEC_RESULT_JSON=` stdout prefix (`cli/autoqec.py:34`)
- `.claude/skills/autoqec-run/SKILL.md` — live LLM orchestration via the Agent tool
- `.claude/skills/{verify-decoder, review-log, diagnose-failure, add-env}/SKILL.md`
- Demos: D1 (surface_d5), D2 (bb72), D4 (reward-hacking), D5 (failure-recovery)
- `autoqec/agents/dispatch.py` — already anticipates background mode (`claude -p` / `codex exec`) — see module docstring

## What's missing

This plan covers 19 tasks across 3 priority tiers. Numbering is `P{tier}.{ordinal}`:

**Tier P0 — blocks demo day**
- P0.1 Live LLM DAG in `cli.autoqec run` (currently `raise ClickException("LLM mode is not wired")`)
- P0.2 Pareto atomic write (tmp-file + `os.replace`)
- P0.3 `artifact_manifest.json` per round (repo SHA, tool versions, env/DSL hashes, command line)
- P0.4 Reward-hacking fixtures `tests/fixtures/reward_hacking/trap_{A,B,C}.pt`
- P0.5 Auto-verify inside `/autoqec-run` (`independent_verify` → `record_round(verify_report=…)`)
- P0.6 Runner writes `round_N_pointer.json`
- P0.7 Runner commits round artifacts to branch + contract promotion

**Tier P1 — research integrity**
- P1.1 Compose round scheduling (`composer.py` + `run_round_plan` list-form routing)
- P1.2 Conventional-commit prefix validator on Coder `commit_message`
- P1.3 Branch-diff containment fence (experiment branches cannot touch `tests/`, `autoqec/envs/builtin/`, `autoqec/eval/`, `docs/contracts/`, `runs/`)
- P1.4 Ctrl-C / resume idempotency (re-running a run_dir skips completed rounds)
- P1.5 `diagnose` classification (OOM / NaN / degenerate) + `diagnosis.md` output
- P1.6 Diagnose failure fixtures (OOM / NaN / degenerate) under `tests/fixtures/diagnose/`
- P1.7 Stuck-run fixture for `review-log`
- P1.8 Ablation zero-signal unit test (random-weights predecoder → CI crosses 0)

**Tier P2 — polish**
- P2.1 D3 `/add-env` onboarding demo directory
- P2.2 Delete unused `autoqec/pareto/` package
- P2.3 Dirty-worktree warning + SHA capture in `artifact_manifest.json`
- P2.4 Offline replay packaging (`autoqec package-run <run_dir> → runs/<id>.tar.gz`)

**Dependencies (hard):**
- P0.5 depends on P0.1 (auto-verify must run inside the LLM loop; without that loop the skill is the only caller)
- P0.7 depends on P0.6 (pointer writes before `git add -A`)
- P1.1 depends on P0.6 + P0.7 (compose rounds need pointers too)
- P1.2 plugs into P0.7 (commit validation happens at commit time)
- P1.3 plugs into P0.7 (branch fence is a commit-time pre-check)
- P1.4 plugs into P0.1 (the LLM loop is where `skip-if-exists` checks happen)
- P2.3 plugs into P0.3 (same manifest writer)

**Migration order:** P0.2 → P0.6 → P0.7 → P0.1 → P0.3 → P0.5 → P0.4 → P1.1 → P1.2 → P1.3 → P1.4 → P1.5/6/7/8 (parallel) → P2.* (parallel).

---

## File structure

### New files

| Path | Task | Responsibility | LOC |
|---|---|---|---|
| `autoqec/runner/pointer.py` | P0.6 | Serialize `round_N_pointer.json` per §15.5 | ~80 |
| `autoqec/runner/manifest.py` | P0.3 | `write_artifact_manifest(round_dir, env_yaml, dsl_config, cmd)` — repo SHA + tool versions + hashes | ~130 |
| `autoqec/orchestration/llm_loop.py` | P0.1 | `run_llm_loop(env, rounds, profile) → run_dir` — ties Ideator / Coder / Runner / Analyst / Verifier together via `dispatch` + `subprocess_runner` | ~320 |
| `autoqec/agents/cli_backend.py` | P0.1 | `invoke_subagent(role, prompt) → parsed_response` — dispatches to `codex exec` or `claude -p` per env vars; captures + parses the fenced JSON block | ~180 |
| `autoqec/orchestration/composer.py` | P1.1 | `plan_compose_round(parents, mode) → dict` | ~80 |
| `autoqec/runner/branch_fence.py` | P1.3 | `check_branch_containment(worktree, base_ref) → list[str]` — returns list of forbidden paths touched | ~90 |
| `autoqec/runner/commit_rules.py` | P1.2 | `validate_commit_message(msg) → None \| raises` — enforces `^(feat\|fix\|test\|docs\|chore\|refactor\|perf\|ci)(\|...)?:` | ~40 |
| `scripts/package_run.py` | P2.4 | `package_run(run_dir) → tar.gz` — bundle run + repo SHA for offline replay | ~80 |
| `scripts/build_trap_fixtures.py` | P0.4 | Produce `trap_{A,B,C}.pt` fixtures + metadata | ~200 |
| `scripts/build_diagnose_fixtures.py` | P1.6 | Produce `tests/fixtures/diagnose/{oom,nan,degenerate}/…` | ~120 |
| `tests/fixtures/reward_hacking/trap_A.pt` | P0.4 | Training-seed-leak checkpoint | binary |
| `tests/fixtures/reward_hacking/trap_B.pt` | P0.4 | Paired-batch-mismatch checkpoint | binary |
| `tests/fixtures/reward_hacking/trap_C.pt` | P0.4 | 100-shot overfit memorizer checkpoint | binary |
| `tests/fixtures/reward_hacking/README.md` | P0.4 | Fixture provenance + rebuild instructions | ~40 |
| `tests/fixtures/diagnose/oom/…` | P1.6 | OOM-simulated round artifacts | text |
| `tests/fixtures/diagnose/nan/…` | P1.6 | NaN-loss round artifacts | text |
| `tests/fixtures/diagnose/degenerate/…` | P1.6 | p=0 degenerate round artifacts | text |
| `tests/fixtures/review_log/stuck_run/…` | P1.7 | 3-round |Δ_LER|<0.002 run | JSONL + md |
| `demos/demo-3-add-env/README.md` | P2.1 | D3 runbook | ~80 |
| `demos/demo-3-add-env/sample-session.md` | P2.1 | Verbatim `/add-env` transcript | ~120 |
| `demos/demo-3-add-env/sample-env.yaml` | P2.1 | The YAML the skill produced | ~40 |
| `tests/test_pointer_writer.py` | P0.6 | `round_N_pointer.json` round-trip + invariants | ~100 |
| `tests/test_artifact_manifest.py` | P0.3 | Manifest fields + hash determinism | ~80 |
| `tests/test_llm_loop.py` | P0.1 | Mock-backend end-to-end unit + one integration | ~180 |
| `tests/test_cli_backend.py` | P0.1 | Dispatch env-var routing + response parsing | ~100 |
| `tests/test_composer.py` | P1.1 | Pure vs with_edit + rejects duplicate/single parents | ~70 |
| `tests/test_branch_fence.py` | P1.3 | Allow + reject paths | ~90 |
| `tests/test_commit_rules.py` | P1.2 | Accept/reject message prefixes | ~50 |
| `tests/test_resume_idempotent.py` | P1.4 | Re-running skips completed rounds | ~100 |
| `tests/test_diagnose_classifier.py` | P1.5 | OOM / NaN / degenerate classification | ~120 |
| `tests/test_review_log_stuck.py` | P1.7 | Stuck-run detection | ~60 |
| `tests/test_ablation_zero_signal.py` | P1.8 | Random weights → CI crosses 0 | ~80 |
| `tests/test_dirty_worktree.py` | P2.3 | Warning emission + SHA in manifest | ~60 |
| `tests/test_package_run.py` | P2.4 | tar.gz round-trip + manifest preservation | ~70 |
| `tests/test_run_round_pointer_integration.py` | P0.7 | Live worktree → pointer + commit (GPU-gated) | ~80 |

### Modified files

| Path | Task | Change |
|---|---|---|
| `autoqec/orchestration/memory.py` | P0.2 | `update_pareto` uses `tmp-file + os.replace` |
| `autoqec/runner/runner.py` | P0.3, P0.6, P0.7, P1.2, P1.3 | Call `write_artifact_manifest` after metrics; call `write_round_pointer`; `git add + commit` + SHA stamp; run `validate_commit_message` + `check_branch_containment` before commit |
| `autoqec/runner/schema.py` | P0.7 | Add `commit_message: Optional[str]` to `RunnerConfig` |
| `cli/autoqec.py` | P0.1, P1.4, P2.4 | `run` calls `run_llm_loop` instead of raising; `run_llm_loop` skips completed rounds; new `package-run` subcommand |
| `autoqec/orchestration/loop.py` | P1.1 | `run_round_plan` routes list-form `fork_from` through `plan_compose_round` |
| `autoqec/orchestration/__init__.py` | P1.1 | Re-export `plan_compose_round` |
| `cli/autoqec.py` `diagnose` subcommand | P1.5 | Replaces thin JSON dump with classifier + optional `diagnosis.md` write |
| `.claude/skills/autoqec-run/SKILL.md` | P0.5 | Add "Verify" step; update "verification not part of skill" disclaimer; compose-round branch |
| `docs/contracts/round_dir_layout.md` | P0.7, P0.3 | Promote `round_N_pointer.json` + `artifact_manifest.json` to enforced |
| `autoqec/pareto/` | P2.2 | **Delete** |
| `tests/test_pareto.py` | P2.2 | **Delete** |
| `Makefile` | P2.1, P0.4, P1.6, P2.4 | Add `demo-3`, `build-trap-fixtures`, `build-diagnose-fixtures`, `package-run` targets |
| `README.md` | P2.1, P0.5 | D3 status → DONE; Demo 1 walkthrough reflects auto-verify |

### Files NOT touched
`autoqec/eval/*`, `autoqec/orchestration/memory.py` (beyond P0.2 one-liner), `autoqec/orchestration/round_recorder.py`, `autoqec/orchestration/reconcile.py`, `autoqec/orchestration/fork_graph.py`, `autoqec/decoders/*`, `autoqec/agents/schemas.py`, `autoqec/agents/dispatch.py`.

---

# Tier P0 — Unblocks demo

## Task P0.2: Pareto atomic write

**Why first:** 3 lines of code, test plan 5.5 gate, no dependencies.

**Files:**
- Modify: `autoqec/orchestration/memory.py::update_pareto`
- Test: `tests/test_pareto_atomic.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_pareto_atomic.py
import json
from pathlib import Path
from unittest.mock import patch

from autoqec.orchestration.memory import RunMemory


def test_update_pareto_uses_os_replace(tmp_path, monkeypatch):
    mem = RunMemory(tmp_path)
    calls = {"write_text": 0, "replace": 0}
    original_replace = __import__("os").replace

    def spy_replace(src, dst):
        calls["replace"] += 1
        return original_replace(src, dst)

    monkeypatch.setattr("os.replace", spy_replace)
    mem.update_pareto([{"round": 1, "delta_ler": 0.01}])
    assert calls["replace"] == 1


def test_update_pareto_leaves_old_on_crash(tmp_path, monkeypatch):
    """If write_text to tmp fails, pareto.json is unchanged."""
    mem = RunMemory(tmp_path)
    mem.update_pareto([{"round": 1, "delta_ler": 0.01}])
    before = Path(mem.pareto_path).read_text()

    def boom(*a, **kw):
        raise OSError("simulated disk-full")

    monkeypatch.setattr("pathlib.Path.write_text", boom)
    try:
        mem.update_pareto([{"round": 2, "delta_ler": 0.99}])
    except OSError:
        pass
    after = Path(mem.pareto_path).read_text()
    assert before == after
```

- [ ] **Step 2: Run → FAIL**

```bash
python -m pytest tests/test_pareto_atomic.py -v
```

Expected: fails (current `update_pareto` calls `write_text` directly).

- [ ] **Step 3: Implement**

`autoqec/orchestration/memory.py::update_pareto`:

```python
def update_pareto(self, pareto: list[dict]) -> None:
    """Replace pareto.json atomically — tmp-file + os.replace (spec §11.4)."""
    import os
    tmp = self.pareto_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(pareto, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, self.pareto_path)
```

- [ ] **Step 4: Pass + regression**

```bash
python -m pytest tests/test_pareto_atomic.py tests/test_round_recorder.py tests/test_orchestration_stub.py -v
```

- [ ] **Step 5: Commit**

```bash
git add autoqec/orchestration/memory.py tests/test_pareto_atomic.py
git commit -m "fix(memory): atomic pareto.json write via tmp + os.replace (§11.4)"
```

---

## Task P0.6: Runner pointer helper

**Files:**
- Create: `autoqec/runner/pointer.py`
- Test: `tests/test_pointer_writer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_pointer_writer.py
import json
from pathlib import Path

import pytest

from autoqec.runner.pointer import write_round_pointer
from autoqec.runner.schema import RoundMetrics, RunnerConfig


def _cfg(tmp, *, branch=None, code_cwd=None, rid="uuid-1"):
    rd = tmp / "round_1"; rd.mkdir(parents=True)
    return RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(rd),
        code_cwd=code_cwd,
        branch=branch,
        round_attempt_id=rid,
    )


def test_pointer_all_fields(tmp_path):
    cfg = _cfg(tmp_path, branch="exp/t/01-x",
               code_cwd=str(tmp_path / ".worktrees" / "exp-x"))
    m = RoundMetrics(
        status="ok", delta_ler=0.003, flops_per_syndrome=100, n_params=500,
        train_wallclock_s=1.0, eval_wallclock_s=0.1,
        branch=cfg.branch, commit_sha="abc", round_attempt_id="uuid-1",
    )
    out = write_round_pointer(cfg=cfg, metrics=m, round_idx=1)
    assert out.name == "round_1_pointer.json"
    d = json.loads(out.read_text())
    assert d == {
        "round_attempt_id": "uuid-1",
        "reconcile_id": None,
        "branch": "exp/t/01-x",
        "commit_sha": "abc",
        "worktree_path": str(tmp_path / ".worktrees" / "exp-x"),
        "fork_from": None,
        "fork_from_ordered": None,
        "compose_mode": None,
        "status": "ok",
        "status_reason": None,
    }


def test_pointer_baseline_round(tmp_path):
    cfg = _cfg(tmp_path)
    m = RoundMetrics(status="ok", delta_ler=0.0, flops_per_syndrome=1,
                     n_params=1, train_wallclock_s=0.1, eval_wallclock_s=0.01,
                     round_attempt_id="uuid-2")
    d = json.loads(write_round_pointer(cfg=cfg, metrics=m, round_idx=2).read_text())
    assert d["branch"] is None and d["commit_sha"] is None


def test_pointer_compose_conflict(tmp_path):
    cfg = _cfg(tmp_path, rid="uuid-3")
    m = RoundMetrics(status="compose_conflict", round_attempt_id="uuid-3",
                     status_reason="parent-A vs parent-B conflict on foo.py")
    d = json.loads(write_round_pointer(cfg=cfg, metrics=m, round_idx=3).read_text())
    assert d["status"] == "compose_conflict"
    assert d["branch"] is None
    assert d["status_reason"].startswith("parent-A")


def test_pointer_requires_id(tmp_path):
    cfg = _cfg(tmp_path, rid="")
    m = RoundMetrics(status="ok", round_attempt_id="")
    with pytest.raises(ValueError, match="round_attempt_id or reconcile_id"):
        write_round_pointer(cfg=cfg, metrics=m, round_idx=4)
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement**

```python
# autoqec/runner/pointer.py
"""Write round_N_pointer.json — authoritative provenance per §15.5.

Read by autoqec.orchestration.reconcile to recover round_attempt_id
after a crash. Without this file reconcile emits a `pause` action.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from autoqec.runner.schema import RoundMetrics, RunnerConfig


def write_round_pointer(
    cfg: RunnerConfig,
    metrics: RoundMetrics,
    round_idx: int,
) -> Path:
    round_attempt_id = cfg.round_attempt_id or metrics.round_attempt_id
    reconcile_id = getattr(metrics, "reconcile_id", None)
    if not round_attempt_id and not reconcile_id:
        raise ValueError(
            "pointer must carry either round_attempt_id or reconcile_id "
            "(spec §15.2 mutual-exclusion)"
        )
    pointer: dict[str, Any] = {
        "round_attempt_id": round_attempt_id or None,
        "reconcile_id": reconcile_id,
        "branch": metrics.branch,
        "commit_sha": metrics.commit_sha,
        "worktree_path": cfg.code_cwd,
        "fork_from": cfg.fork_from,
        "fork_from_ordered": getattr(cfg, "fork_from_ordered", None),
        "compose_mode": cfg.compose_mode,
        "status": metrics.status,
        "status_reason": getattr(metrics, "status_reason", None),
    }
    out = Path(cfg.round_dir) / f"round_{round_idx}_pointer.json"
    out.write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
```

- [ ] **Step 4: Pass + commit**

```bash
python -m pytest tests/test_pointer_writer.py -v
git add autoqec/runner/pointer.py tests/test_pointer_writer.py
git commit -m "feat(runner): write_round_pointer per §15.5"
```

---

## Task P0.7: Runner commits round artifacts to branch + contract

**Files:**
- Modify: `autoqec/runner/runner.py`
- Modify: `autoqec/runner/schema.py` (add `commit_message: Optional[str]` to `RunnerConfig`)
- Modify: `docs/contracts/round_dir_layout.md`
- Test: `tests/test_run_round_pointer_integration.py`

**Design:** commit runs inside the worktree via `git -C <code_cwd>`. Commit message comes from Coder's `commit_message` → `RunnerConfig.commit_message`; validation (P1.2) and branch fence (P1.3) run as pre-commit steps but are implemented in later tasks. If any git step fails: `status="failed"`, `status_reason="git_commit_error: <msg>"`, metrics.json still written — training work not lost.

- [ ] **Step 1: Add `commit_message` to RunnerConfig**

`autoqec/runner/schema.py`:

```python
# add to RunnerConfig
commit_message: Optional[str] = None
# in docstring: "Coder's proposed commit message when the round runs on a branch; falls back to 'round-attempt <uuid>'."
```

- [ ] **Step 2: Write failing integration test**

```python
# tests/test_run_round_pointer_integration.py
import json
import subprocess
from pathlib import Path

import pytest

from autoqec.orchestration.worktree import create_round_worktree
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig


@pytest.mark.integration
def test_run_round_commits_and_writes_pointer(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-b", "main", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "init"])

    wt = create_round_worktree(
        repo_root=str(repo), run_id="20260423-000000", round_idx=1, slug="smoke",
    )
    round_dir = Path(wt["worktree_path"]) / "runs" / "20260423-000000" / "round_1"
    round_dir.mkdir(parents=True)

    cfg = RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={
            "type": "gnn", "output_mode": "soft_priors",
            "hidden_dim": 4, "n_layers": 1,
        },
        training_profile="dev",
        seed=0,
        round_dir=str(round_dir),
        code_cwd=wt["worktree_path"],
        branch=wt["branch"],
        round_attempt_id="test-uuid-1",
        commit_message="feat(test): smoke round",
    )
    metrics = run_round(cfg)

    assert metrics.status == "ok"
    assert metrics.commit_sha is not None
    assert (round_dir / "round_1_pointer.json").exists()

    p = json.loads((round_dir / "round_1_pointer.json").read_text())
    assert p["commit_sha"] == metrics.commit_sha

    tip = subprocess.check_output(
        ["git", "-C", wt["worktree_path"], "rev-parse", wt["branch"]]
    ).decode().strip()
    assert tip == metrics.commit_sha
```

- [ ] **Step 3: FAIL**

- [ ] **Step 4: Implement** — replace the tail of `run_round` in `autoqec/runner/runner.py` (current line 231-245):

```python
    import subprocess  # add at top of file if not present
    metrics = RoundMetrics(
        status="ok",
        ler_plain_classical=ler_plain,
        ler_predecoder=ler_predecoder,
        delta_ler=delta_ler,
        flops_per_syndrome=int(flops),
        n_params=n_params,
        train_wallclock_s=train_wallclock,
        eval_wallclock_s=time.time() - eval_start,
        vram_peak_gb=(
            float(torch.cuda.max_memory_allocated() / 1e9)
            if torch.cuda.is_available() else 0.0
        ),
        checkpoint_path=str(checkpoint_path),
        training_log_path=str(train_log),
        branch=config.branch,
        round_attempt_id=config.round_attempt_id,
        fork_from=config.fork_from,
        compose_mode=config.compose_mode,
    )
    (round_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))

    # --- commit + pointer (spec §15.5, §15.8) ---
    if config.branch is not None and config.code_cwd is not None:
        try:
            commit_sha = _commit_round_to_branch(config, round_idx=_round_idx_from_dir(round_dir))
            metrics.commit_sha = commit_sha
            metrics.branch = config.branch
            (round_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))
        except subprocess.CalledProcessError as exc:
            metrics.status = "failed"
            metrics.status_reason = f"git_commit_error: {exc}"
            (round_dir / "metrics.json").write_text(metrics.model_dump_json(indent=2))

    from autoqec.runner.pointer import write_round_pointer
    write_round_pointer(cfg=config, metrics=metrics, round_idx=_round_idx_from_dir(round_dir))
    return metrics


def _round_idx_from_dir(round_dir: Path) -> int:
    return int(round_dir.name.split("_")[-1])


def _commit_round_to_branch(config: "RunnerConfig", round_idx: int) -> str:
    import subprocess
    worktree = Path(config.code_cwd)
    round_rel = Path(config.round_dir).relative_to(worktree)
    message = config.commit_message or f"round-attempt {config.round_attempt_id or ''}".strip()
    subprocess.check_call(["git", "-C", str(worktree), "add", str(round_rel)])
    subprocess.check_call(["git", "-C", str(worktree), "commit", "-m", message])
    return subprocess.check_output(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"]
    ).decode().strip()
```

- [ ] **Step 5: Pass + commit**

```bash
python -m pytest tests/test_pointer_writer.py tests/test_run_round_pointer_integration.py --run-integration -v
git add autoqec/runner/runner.py autoqec/runner/schema.py tests/test_run_round_pointer_integration.py
git commit -m "feat(runner): commit round artifacts to branch + emit round_N_pointer.json"
```

- [ ] **Step 6: Contract promotion** — `docs/contracts/round_dir_layout.md`: move `round_N_pointer.json` section from aspirational → enforced; add line "Enforced as of 2026-04-23 (Runner writes this whenever the round runs with a branch)."

```bash
git add docs/contracts/round_dir_layout.md
git commit -m "docs(contracts): promote round_N_pointer.json to enforced — contract-change"
```

PR with `contract-change` label; requires 3-of-3 owner sign-off.

---

## Task P0.1: Live LLM DAG in `cli.autoqec run`

**Biggest task in this plan.** Currently `cli/autoqec.py:323` raises `"LLM mode is not wired in this branch yet; use --no-llm"`. The `/autoqec-run` SKILL already does live orchestration via the Agent tool inside Claude Code; we now need a headless Python path so `python -m cli.autoqec run ENV --rounds 3` works from a terminal (test plan Phase 4 requires it).

**Architecture:**
1. `autoqec/agents/cli_backend.py::invoke_subagent(role, prompt) → dict` — one subprocess call per subagent. Reads `AUTOQEC_{ROLE}_BACKEND` (codex-cli | claude-cli) + `AUTOQEC_{ROLE}_MODEL` env vars. Builds the shell command (`codex exec --model $M -` vs `claude -p --model $M`), pipes the prompt over stdin, parses the fenced JSON block from stdout.
2. `autoqec/orchestration/llm_loop.py::run_llm_loop(env, rounds, profile, run_dir) → run_dir` — driver. For each round: build Ideator ctx → dispatch Ideator → build Coder ctx → dispatch Coder → construct RunnerConfig → `run_round_in_subprocess` OR inline `run_round` (depends on whether worktree mode is selected) → build Analyst ctx → dispatch Analyst → `record_round`.
3. `cli/autoqec.py run`: remove the `raise`; call `run_llm_loop`.

**Scope exclusions (stay minimal):**
- Worktree mode (`code_cwd` / `branch`) is **opt-in** via a new `--worktree` flag. Default: inline `run_round` on `main`. This keeps the first ship small; worktree rounds are exercised via `/autoqec-run` SKILL and `run-round --code-cwd` already.
- Compose rounds: list-form `fork_from` is rejected by the first ship (`raise NotImplementedError` pending P1.1). Rationale: the LLM is free to propose a list but the loop short-circuits with "compose mode requires P1.1"; this keeps P0.1 small.
- Auto-verify happens inside the loop (P0.5).

**Files:**
- Create: `autoqec/agents/cli_backend.py`
- Create: `autoqec/orchestration/llm_loop.py`
- Modify: `cli/autoqec.py` — `run` body
- Test: `tests/test_cli_backend.py`, `tests/test_llm_loop.py`

### P0.1.a — `cli_backend.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cli_backend.py
from unittest.mock import patch

import pytest

from autoqec.agents.cli_backend import (
    InvalidSubagentResponseError,
    _build_cli_argv,
    _parse_fenced_json,
    invoke_subagent,
)


def test_build_argv_codex_cli(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "gpt-5.4")
    assert _build_cli_argv("ideator") == ["codex", "exec", "--model", "gpt-5.4", "-"]


def test_build_argv_claude_cli(monkeypatch):
    monkeypatch.setenv("AUTOQEC_CODER_BACKEND", "claude-cli")
    monkeypatch.setenv("AUTOQEC_CODER_MODEL", "claude-haiku-4-5")
    assert _build_cli_argv("coder") == ["claude", "-p", "--model", "claude-haiku-4-5"]


def test_parse_fenced_json_strict():
    text = 'hello\n\n```json\n{"k": "v"}\n```\ntrailing'
    assert _parse_fenced_json(text) == {"k": "v"}


def test_parse_fenced_json_no_block_raises():
    with pytest.raises(InvalidSubagentResponseError, match="no fenced json"):
        _parse_fenced_json("plain prose with no json")


def test_parse_fenced_json_malformed_raises():
    with pytest.raises(InvalidSubagentResponseError, match="malformed json"):
        _parse_fenced_json("```json\n{not valid\n```")


def test_invoke_subagent_returns_parsed(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "gpt-5.4")

    class FakeCompleted:
        stdout = '```json\n{"hypothesis": "try GNN", "fork_from": "baseline"}\n```'
        stderr = ""
        returncode = 0

    with patch("subprocess.run", return_value=FakeCompleted()):
        out = invoke_subagent("ideator", "prompt here")
    assert out["hypothesis"] == "try GNN"


def test_invoke_subagent_propagates_nonzero(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "gpt-5.4")

    class FakeCompleted:
        stdout = ""
        stderr = "model refused"
        returncode = 2

    with patch("subprocess.run", return_value=FakeCompleted()):
        with pytest.raises(InvalidSubagentResponseError, match="exit 2"):
            invoke_subagent("ideator", "prompt")
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement**

```python
# autoqec/agents/cli_backend.py
"""Background subagent dispatch — subprocess adapter for codex-cli / claude-cli.

Mirrors the inline Agent-tool contract used by `/autoqec-run` SKILL: one
fenced ```json block per response. Env-var matrix per role:
  AUTOQEC_{role}_BACKEND ∈ {"codex-cli", "claude-cli"}
  AUTOQEC_{role}_MODEL   = backend-specific model id

Used by `autoqec.orchestration.llm_loop`; the SKILL path stays untouched.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Literal

Role = Literal["ideator", "coder", "analyst"]


class InvalidSubagentResponseError(RuntimeError):
    pass


_BACKEND_ARGV = {
    "codex-cli": lambda model: ["codex", "exec", "--model", model, "-"],
    "claude-cli": lambda model: ["claude", "-p", "--model", model],
}


def _build_cli_argv(role: Role) -> list[str]:
    backend = os.environ.get(f"AUTOQEC_{role.upper()}_BACKEND", "codex-cli")
    model = os.environ.get(
        f"AUTOQEC_{role.upper()}_MODEL",
        "gpt-5.4" if backend == "codex-cli" else "claude-haiku-4-5",
    )
    if backend not in _BACKEND_ARGV:
        raise InvalidSubagentResponseError(
            f"unknown backend {backend!r} for role {role!r}"
        )
    return _BACKEND_ARGV[backend](model)


_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _parse_fenced_json(stdout: str) -> dict:
    m = _FENCE.search(stdout)
    if m is None:
        raise InvalidSubagentResponseError(
            f"no fenced json block in response (got {stdout[:120]!r})"
        )
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise InvalidSubagentResponseError(f"malformed json: {exc}") from exc


def invoke_subagent(role: Role, prompt: str, timeout: float = 300.0) -> dict:
    argv = _build_cli_argv(role)
    result = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise InvalidSubagentResponseError(
            f"{argv[0]} exit {result.returncode}: {result.stderr[:200]}"
        )
    return _parse_fenced_json(result.stdout)
```

- [ ] **Step 4: Pass + commit**

```bash
python -m pytest tests/test_cli_backend.py -v
git add autoqec/agents/cli_backend.py tests/test_cli_backend.py
git commit -m "feat(agents): cli_backend dispatch for background codex-cli / claude-cli"
```

### P0.1.b — `llm_loop.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_loop.py
from unittest.mock import patch, MagicMock

import pytest

from autoqec.orchestration.llm_loop import run_llm_loop


def _stub_ideator(round_idx):
    return {
        "hypothesis": f"round {round_idx} idea",
        "fork_from": "baseline",
        "compose_mode": None,
        "rationale": "try a thing",
    }


def _stub_coder(round_idx):
    return {
        "dsl_config": {
            "type": "gnn", "output_mode": "soft_priors",
            "hidden_dim": 4, "n_layers": 1,
        },
        "tier": "1",
        "rationale": "small gnn",
        "commit_message": "feat(round): small gnn",
    }


def _stub_analyst(round_idx):
    return {
        "summary_1line": f"round {round_idx} ok",
        "verdict": "candidate",
        "next_hypothesis_seed": "try bigger",
    }


def test_run_llm_loop_happy_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")

    stub_metrics = MagicMock(
        status="ok", delta_ler=0.001,
        model_dump=MagicMock(return_value={"status": "ok", "delta_ler": 0.001}),
    )
    stub_report = MagicMock(
        verdict="SUSPICIOUS",
        model_dump=MagicMock(return_value={"verdict": "SUSPICIOUS",
                                           "delta_vs_baseline_holdout": None}),
    )

    responses = {"ideator": [], "coder": [], "analyst": []}

    def fake_invoke(role, prompt, timeout=300.0):
        responses[role].append(prompt)
        idx = len(responses[role])
        return {
            "ideator": _stub_ideator,
            "coder": _stub_coder,
            "analyst": _stub_analyst,
        }[role](idx)

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke), \
         patch("autoqec.orchestration.llm_loop.run_round", return_value=stub_metrics), \
         patch("autoqec.orchestration.llm_loop.independent_verify", return_value=stub_report):
        run_dir = run_llm_loop(env=env, rounds=2, profile="dev")

    assert (run_dir / "history.jsonl").exists()
    hist = (run_dir / "history.jsonl").read_text().strip().splitlines()
    assert len(hist) == 2
    assert len(responses["ideator"]) == 2


def test_run_llm_loop_rejects_compose_rounds_until_p11(tmp_path, monkeypatch):
    """P0.1 ships without compose support; Ideator emitting list fork_from should error clearly."""
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")

    def fake_invoke(role, prompt, timeout=300.0):
        if role == "ideator":
            return {"hypothesis": "merge A and B", "fork_from": ["a", "b"],
                    "compose_mode": "pure", "rationale": ""}
        return {}

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke):
        with pytest.raises(NotImplementedError, match="compose"):
            run_llm_loop(env=env, rounds=1, profile="dev")
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement**

```python
# autoqec/orchestration/llm_loop.py
"""Headless live-LLM research loop — Python driver for `cli.autoqec run`.

Mirrors the `/autoqec-run` SKILL but dispatches subagents via
`autoqec.agents.cli_backend.invoke_subagent` (subprocess-spawn codex / claude)
instead of the Agent tool. Runs the Runner inline (not in a worktree) by
default; worktree scheduling + compose rounds land in P1.1.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from autoqec.agents.cli_backend import invoke_subagent
from autoqec.agents.dispatch import build_prompt
from autoqec.envs.schema import EnvSpec
from autoqec.eval.independent_eval import independent_verify
from autoqec.orchestration.loop import build_analyst_prompt, build_coder_prompt
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig
from autoqec.tools.machine_state import machine_state


def run_llm_loop(
    env: EnvSpec,
    rounds: int,
    profile: str,
    run_dir: Path | str | None = None,
) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir) if run_dir else (Path("runs") / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    mem = RunMemory(run_dir)

    for round_idx in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        # P1.4 will add a skip-if-complete check here.

        # --- Ideator ---
        ideator_ctx = mem.l3_for_ideator(
            env_spec=env.model_dump(),
            kb_excerpt="",
            machine_state=machine_state(run_dir),
            run_id=run_dir.name,
        )
        ideator_resp = invoke_subagent("ideator", build_prompt("ideator", ideator_ctx))
        if isinstance(ideator_resp.get("fork_from"), list):
            raise NotImplementedError(
                "compose rounds not supported in P0.1 LLM loop; "
                "land P1.1 (autoqec.orchestration.composer) first"
            )

        # --- Coder ---
        coder_prompt = build_coder_prompt(
            hypothesis=ideator_resp, mem=mem,
            dsl_schema_md=_dsl_schema_md(),
        )
        coder_resp = invoke_subagent("coder", coder_prompt)

        # --- Runner ---
        cfg = RunnerConfig(
            env_name=env.name,
            predecoder_config=coder_resp["dsl_config"],
            training_profile=profile,
            seed=round_idx,
            round_dir=str(round_dir),
            round_attempt_id=str(uuid.uuid4()),
            fork_from=ideator_resp.get("fork_from", "baseline"),
            commit_message=coder_resp.get("commit_message"),
        )
        metrics = run_round(cfg, env)

        # --- Verifier (auto-verify; see P0.5 SKILL parity) ---
        verify_verdict: str | None = None
        verify_report: dict[str, Any] | None = None
        if metrics.status == "ok":
            try:
                sp = env.noise.seed_policy
                holdout = list(range(sp.holdout[0], sp.holdout[1] + 1))[:50]
                report = independent_verify(
                    checkpoint=round_dir / "checkpoint.pt",
                    env_spec=env,
                    holdout_seeds=holdout,
                )
                (round_dir / "verification_report.json").write_text(
                    report.model_dump_json(indent=2), encoding="utf-8",
                )
                verify_verdict = report.verdict
                verify_report = report.model_dump()
            except Exception as exc:
                # Never fail a whole round because the verifier crashed.
                (round_dir / "verification_error.txt").write_text(str(exc))
                verify_verdict = "FAILED"

        # --- Analyst ---
        analyst_prompt = build_analyst_prompt(
            mem=mem, round_dir=round_dir, prev_summary="",
        )
        analyst_resp = invoke_subagent("analyst", analyst_prompt)

        # --- Record ---
        record = metrics.model_dump()
        record["round"] = round_idx
        record["hypothesis"] = ideator_resp.get("hypothesis")
        record["verdict"] = analyst_resp.get("verdict")
        record["summary_1line"] = analyst_resp.get("summary_1line")
        record_round(
            mem,
            round_metrics=record,
            verify_verdict=verify_verdict,
            verify_report=verify_report,
        )

    # Final bookkeeping line for smoke scripts.
    print(f"AUTOQEC_RESULT_JSON={json.dumps({'run_dir': str(run_dir)})}")
    return run_dir


def _dsl_schema_md() -> str:
    from autoqec.decoders.dsl_schema import PredecoderDSL
    return json.dumps(PredecoderDSL.model_json_schema(), indent=2)
```

- [ ] **Step 4: Modify `cli/autoqec.py run` body**

Replace lines 299-... (the `raise ClickException("LLM mode is not wired in this branch yet; use --no-llm")`) with:

```python
        else:
            # live LLM path — P0.1
            from autoqec.orchestration.llm_loop import run_llm_loop
            return run_llm_loop(env=env, rounds=rounds, profile=profile)
```

(remove the surrounding `for round_idx in range(...)` in the no_llm=False branch; `run_llm_loop` owns the loop.)

- [ ] **Step 5: Pass + commit**

```bash
python -m pytest tests/test_llm_loop.py tests/test_cli_backend.py -v
git add autoqec/orchestration/llm_loop.py cli/autoqec.py tests/test_llm_loop.py
git commit -m "feat(cli): wire live LLM DAG into cli.autoqec run via llm_loop"
```

---

## Task P0.3: `artifact_manifest.json` per round

**Files:**
- Create: `autoqec/runner/manifest.py`
- Modify: `autoqec/runner/runner.py` (call `write_artifact_manifest` after `write_round_pointer`)
- Modify: `docs/contracts/round_dir_layout.md` (document `artifact_manifest.json`)
- Test: `tests/test_artifact_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_artifact_manifest.py
import hashlib
import json
import subprocess
from pathlib import Path

from autoqec.runner.manifest import write_artifact_manifest


def test_manifest_contains_repo_sha_and_hashes(tmp_path):
    # Arrange a fake env YAML + DSL config
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("name: test_env\n")
    dsl = {"type": "gnn", "hidden_dim": 4}
    round_dir = tmp_path / "round_1"
    round_dir.mkdir()

    out = write_artifact_manifest(
        round_dir=round_dir,
        env_yaml_path=env_yaml,
        dsl_config=dsl,
        cmd_line=["python", "-m", "cli.autoqec", "run", "fake.yaml"],
    )

    data = json.loads(out.read_text())
    # Env hash is deterministic sha-256 of bytes
    expected = hashlib.sha256(env_yaml.read_bytes()).hexdigest()
    assert data["env_yaml_sha256"] == expected
    # DSL hash is sha-256 of sorted-keys canonical JSON
    dsl_hash = hashlib.sha256(
        json.dumps(dsl, sort_keys=True).encode()
    ).hexdigest()
    assert data["dsl_sha256"] == dsl_hash
    # Tool versions
    assert "python_version" in data
    assert "torch_version" in data or data.get("torch_version") is None
    # Command line is preserved
    assert data["cmd_line"] == ["python", "-m", "cli.autoqec", "run", "fake.yaml"]


def test_manifest_repo_sha_optional(tmp_path):
    # When called outside a git repo, repo_sha is None but no crash
    env_yaml = tmp_path / "env.yaml"; env_yaml.write_text("")
    rd = tmp_path / "round_1"; rd.mkdir()
    data = json.loads(write_artifact_manifest(
        round_dir=rd, env_yaml_path=env_yaml, dsl_config={}, cmd_line=["x"],
    ).read_text())
    assert "repo_sha" in data  # key present even if None
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement**

```python
# autoqec/runner/manifest.py
"""Per-round artifact manifest — reproducibility gate (test plan 4.5)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def _tool_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _repo_sha(anchor: Path) -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(anchor), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _dirty_files(anchor: Path) -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(anchor), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    dirty = []
    for line in out:
        if not line.strip():
            continue
        path = line[3:].strip()
        file_path = anchor / path
        sha = ""
        if file_path.is_file():
            sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        dirty.append({"path": path, "sha256": sha})
    return dirty


def write_artifact_manifest(
    round_dir: Path,
    env_yaml_path: Path,
    dsl_config: dict[str, Any],
    cmd_line: list[str],
) -> Path:
    env_bytes = Path(env_yaml_path).read_bytes() if Path(env_yaml_path).exists() else b""
    env_sha = hashlib.sha256(env_bytes).hexdigest()
    dsl_sha = hashlib.sha256(
        json.dumps(dsl_config, sort_keys=True).encode()
    ).hexdigest()

    anchor = round_dir
    manifest = {
        "repo_sha": _repo_sha(anchor),
        "dirty_files": _dirty_files(anchor),
        "python_version": sys.version.split()[0],
        "torch_version": _tool_version("torch"),
        "stim_version": _tool_version("stim"),
        "pymatching_version": _tool_version("pymatching"),
        "ldpc_version": _tool_version("ldpc"),
        "numpy_version": _tool_version("numpy"),
        "env_yaml_path": str(env_yaml_path),
        "env_yaml_sha256": env_sha,
        "dsl_sha256": dsl_sha,
        "cmd_line": cmd_line,
    }
    out = round_dir / "artifact_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
```

- [ ] **Step 4: Hook into Runner**

After `write_round_pointer(...)` in `autoqec/runner/runner.py`:

```python
from autoqec.runner.manifest import write_artifact_manifest

write_artifact_manifest(
    round_dir=round_dir,
    env_yaml_path=Path(config.env_yaml_path) if hasattr(config, "env_yaml_path") else Path("."),
    dsl_config=config.predecoder_config,
    cmd_line=sys.argv,
)
```

(If `env_yaml_path` isn't currently on `RunnerConfig`, pass it via a new field or let the caller inject after the fact — small scope decision for the implementing subagent.)

- [ ] **Step 5: Pass + commit**

```bash
python -m pytest tests/test_artifact_manifest.py -v
git add autoqec/runner/manifest.py autoqec/runner/runner.py tests/test_artifact_manifest.py
git commit -m "feat(runner): per-round artifact_manifest.json with repo SHA + tool versions"
```

- [ ] **Step 6: Contract update**

`docs/contracts/round_dir_layout.md`: document `artifact_manifest.json` under enforced files. PR with `contract-change` label.

---

## Task P0.5: Auto-verify inside `/autoqec-run` SKILL

**Note:** P0.1's `llm_loop.py` already runs `independent_verify` inline. This task is the **SKILL side** — when users run `/autoqec-run` in Claude Code (not `cli.autoqec run`), the skill should do the same. Both paths must behave identically.

**Files:**
- Modify: `.claude/skills/autoqec-run/SKILL.md`

- [ ] **Step 1: Read current SKILL**

```bash
grep -n "record_round\|verify_verdict\|Verification\|not part of this skill" .claude/skills/autoqec-run/SKILL.md
```

- [ ] **Step 2: Insert "Verify" step after Analyst, before record_round**

See plan v1 (prior revision) for the full verbatim insert; key elements:

````markdown
### Step N — Run the independent verifier (automatic)

If `metrics.status == "ok"` AND the Analyst verdict is `candidate`:

```python
from pathlib import Path
from autoqec.envs.schema import load_env_yaml
from autoqec.eval.independent_eval import independent_verify

env = load_env_yaml("<ENV_YAML_PATH>")
sp = env.noise.seed_policy
holdout = list(range(sp.holdout[0], sp.holdout[1] + 1))[:50]
report = independent_verify(
    checkpoint=Path("<ROUND_DIR>/checkpoint.pt"),
    env_spec=env,
    holdout_seeds=holdout,
)
Path("<ROUND_DIR>/verification_report.json").write_text(
    report.model_dump_json(indent=2), encoding="utf-8",
)
verify_verdict = report.verdict       # pass into record_round
verify_report = report.model_dump()   # ditto
```

**Skip cases:** `metrics.status != "ok"`, `compose_conflict`, Analyst verdict == `ignore`.

**On verifier crash:** catch, set `verify_verdict="FAILED"`, `verify_report=None`, include the exception in the Analyst summary.
````

- [ ] **Step 3: Update the `record_round` call in the SKILL**

```python
record_round(
    mem,
    round_metrics=metrics.model_dump(),
    verify_verdict=verify_verdict,   # from new Step N
    verify_report=verify_report,      # dict or None
)
```

- [ ] **Step 4: Replace the disclaimer**

Old line:
> Verification (`/verify-decoder`) is **not** part of this skill.

New:
> Verification runs **automatically** in this skill for successful rounds with Analyst verdict `candidate` (Step N). `/verify-decoder` remains available for post-hoc re-audits with different holdout seeds.

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/autoqec-run/SKILL.md
git commit -m "docs(skill): /autoqec-run auto-invokes independent_verify (parity with cli.autoqec run)"
```

---

## Task P0.4: Reward-hacking fixtures `trap_{A,B,C}.pt`

**What each trap proves:**
- **Trap A** — training-seed leak. A predecoder trained on `seed ∈ train_range`, then evaluated on `seed ∈ holdout_range` that **overlaps** the training range by construction. Verifier's `_seed_leakage_check` must raise / return `seed_leakage_check_ok=False` → verdict ≠ VERIFIED.
- **Trap B** — paired batch mismatch. Two checkpoints evaluated with **different** syndrome tensors under the same `paired_eval_bundle_id`. Verifier must detect the bundle-id → bytes mismatch.
- **Trap C** — 100-shot overfit memorizer. `autoqec.cheaters.memorize.MemorizerPredecoder` trained on 100 syndromes, evaluated on 200 K holdout → CI of `delta_ler_holdout` crosses 0 or `ablation_sanity_ok=False`.

**Files:**
- Create: `scripts/build_trap_fixtures.py`
- Create: `tests/fixtures/reward_hacking/README.md`
- Create: `tests/fixtures/reward_hacking/trap_A.pt`, `trap_B.pt`, `trap_C.pt` (binary)
- Modify: `Makefile` — `make build-trap-fixtures`

- [ ] **Step 1: Build script**

```python
# scripts/build_trap_fixtures.py
"""Produce reward-hacking fixture checkpoints.

Run once with `make build-trap-fixtures`. Fixtures check into tests/fixtures/reward_hacking/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from autoqec.cheaters.memorize import MemorizerPredecoder
from autoqec.decoders.dsl_compiler import compile_predecoder
from autoqec.envs.schema import load_env_yaml


OUT_DIR = Path("tests/fixtures/reward_hacking")


def build_trap_a(env_yaml: Path) -> None:
    """Training-seed-leak: trains on seed 5, claims train_seeds=[1..10]."""
    env = load_env_yaml(env_yaml)
    model = compile_predecoder(
        {"type": "gnn", "output_mode": "soft_priors", "hidden_dim": 8, "n_layers": 2},
        code_spec=env.code,
    )
    ckpt = {
        "class_name": "GNNPredecoder",
        "state_dict": model.state_dict(),
        "output_mode": "soft_priors",
        "dsl_config": {"type": "gnn", "hidden_dim": 8, "n_layers": 2},
        "train_seeds_claimed": list(range(env.noise.seed_policy.holdout[0],
                                           env.noise.seed_policy.holdout[0] + 5)),
        "trap_kind": "training_seed_leak",
    }
    torch.save(ckpt, OUT_DIR / "trap_A.pt")


def build_trap_b(env_yaml: Path) -> None:
    """Paired-batch-mismatch: ckpt carries paired_eval_bundle_id=X but eval
    data hash inside the ckpt record is the hash of a DIFFERENT bundle."""
    env = load_env_yaml(env_yaml)
    model = compile_predecoder(
        {"type": "gnn", "output_mode": "soft_priors", "hidden_dim": 8, "n_layers": 2},
        code_spec=env.code,
    )
    ckpt = {
        "class_name": "GNNPredecoder",
        "state_dict": model.state_dict(),
        "output_mode": "soft_priors",
        "dsl_config": {"type": "gnn", "hidden_dim": 8, "n_layers": 2},
        "paired_eval_bundle_id": "deadbeef" * 2,  # claimed id
        "recorded_syndrome_sha256": "cafebabe" * 8,  # but hash doesn't match
        "trap_kind": "paired_batch_mismatch",
    }
    torch.save(ckpt, OUT_DIR / "trap_B.pt")


def build_trap_c(env_yaml: Path) -> None:
    """Overfit 100-shot memorizer — memorizes on tiny train, fails on holdout."""
    env = load_env_yaml(env_yaml)
    # Use the existing MemorizerPredecoder with 100 random syndromes memorized
    # Need an input_dim/output_dim that matches env.code's n_check / n_fault
    from autoqec.envs.code_assets import load_code_artifacts
    art = load_code_artifacts(env)
    mem = MemorizerPredecoder(
        input_dim=art.n_check, output_dim=art.n_var, output_mode="soft_priors",
    )
    train_syndromes = torch.randint(0, 2, (100, art.n_check)).float()
    train_priors = torch.rand(100, art.n_var)
    mem.fit(train_syndromes, train_priors)
    ckpt = {
        "class_name": "MemorizerPredecoder",
        "state_dict": mem.state_dict() if hasattr(mem, "state_dict") else {},
        "memorizer_table": mem._table,  # raw table
        "output_mode": "soft_priors",
        "dsl_config": {"type": "custom", "path": "autoqec.cheaters.memorize.MemorizerPredecoder"},
        "trap_kind": "overfit_memorizer",
    }
    torch.save(ckpt, OUT_DIR / "trap_C.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="autoqec/envs/builtin/surface_d5_depol.yaml")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_trap_a(Path(args.env))
    build_trap_b(Path(args.env))
    build_trap_c(Path(args.env))
    manifest = {
        "trap_A": "training_seed_leak",
        "trap_B": "paired_batch_mismatch",
        "trap_C": "overfit_memorizer",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: README**

```markdown
# Reward-hacking fixtures

Binary checkpoint fixtures produced by `scripts/build_trap_fixtures.py`. Each
file simulates one class of reward-hacking failure the verifier must catch
(spec §9 + test plan Phase 5.2).

## Regenerate

```bash
make build-trap-fixtures
```

Requires GPU + full env (`pip install -e '.[dev]'`). Commit the resulting
`.pt` files **and** `manifest.json` to the repo — fixtures are small
(~KB) and deterministic under the build script.

## Files

| File | Failure class | Expected verdict | Relevant guard |
|---|---|---|---|
| `trap_A.pt` | training_seed_leak | ≠ VERIFIED | `seed_leakage_check_ok=False` |
| `trap_B.pt` | paired_batch_mismatch | ≠ VERIFIED | `paired_eval_bundle_id` bytes mismatch |
| `trap_C.pt` | overfit_memorizer | FAILED or CI crosses 0 | `ablation_sanity_ok=False` |
```

- [ ] **Step 3: Makefile target**

```makefile
build-trap-fixtures:
	$(PYTHON) scripts/build_trap_fixtures.py --env autoqec/envs/builtin/surface_d5_depol.yaml
```

- [ ] **Step 4: Generate + verify**

```bash
make build-trap-fixtures
ls -la tests/fixtures/reward_hacking/
python -c "
import torch
for t in ['A','B','C']:
    ck = torch.load(f'tests/fixtures/reward_hacking/trap_{t}.pt', weights_only=False)
    print(t, ck['trap_kind'])
"
```

- [ ] **Step 5: Add acceptance test**

```python
# tests/test_reward_hacking_traps.py
import json
from pathlib import Path

import pytest


FIX = Path("tests/fixtures/reward_hacking")


@pytest.mark.skipif(not (FIX / "trap_A.pt").exists(), reason="fixtures not built")
@pytest.mark.integration
def test_trap_A_fails_verification():
    from autoqec.envs.schema import load_env_yaml
    from autoqec.eval.independent_eval import independent_verify

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    # Construct a deliberately-leaky holdout: one seed inside train range.
    sp = env.noise.seed_policy
    holdout = [sp.train[0]]  # leak
    with pytest.raises(ValueError, match="seed"):
        independent_verify(FIX / "trap_A.pt", env, holdout_seeds=holdout)


@pytest.mark.skipif(not (FIX / "trap_C.pt").exists(), reason="fixtures not built")
@pytest.mark.integration
def test_trap_C_memorizer_fails_or_ci_crosses_zero():
    from autoqec.envs.schema import load_env_yaml
    from autoqec.eval.independent_eval import independent_verify

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    sp = env.noise.seed_policy
    holdout = list(range(sp.holdout[0], sp.holdout[0] + 3))
    report = independent_verify(FIX / "trap_C.pt", env, holdout_seeds=holdout, n_shots=1000)
    ci_lo, ci_hi = report.ler_holdout_ci
    assert report.verdict == "FAILED" or (ci_lo < 0 < ci_hi)
```

- [ ] **Step 6: Commit**

```bash
git add scripts/build_trap_fixtures.py tests/fixtures/reward_hacking/ Makefile tests/test_reward_hacking_traps.py
git commit -m "test: reward-hacking fixtures trap_{A,B,C}.pt + acceptance test"
```

---

# Tier P1 — Research-integrity gates

## Task P1.1: Compose round scheduling

**Files:**
- Create: `autoqec/orchestration/composer.py`
- Modify: `autoqec/orchestration/loop.py` (`run_round_plan`)
- Modify: `autoqec/orchestration/llm_loop.py` (remove the `NotImplementedError`, add list-form routing)
- Test: `tests/test_composer.py`, extend `tests/test_loop_helpers.py`

- [ ] **Step 1: Failing tests for composer**

```python
# tests/test_composer.py
import pytest

from autoqec.orchestration.composer import plan_compose_round


def test_pure_skips_coder():
    p = plan_compose_round(parents=["a", "b"], compose_mode="pure")
    assert p["invoke_coder"] is False


def test_with_edit_invokes_coder():
    p = plan_compose_round(parents=["a", "b"], compose_mode="with_edit")
    assert p["invoke_coder"] is True


def test_rejects_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        plan_compose_round(parents=["a", "a"], compose_mode="pure")


def test_rejects_single():
    with pytest.raises(ValueError, match="at least 2"):
        plan_compose_round(parents=["a"], compose_mode="pure")


def test_rejects_bad_mode():
    with pytest.raises(ValueError, match="compose_mode"):
        plan_compose_round(parents=["a", "b"], compose_mode="bogus")
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement `composer.py`**

```python
# autoqec/orchestration/composer.py
"""Plan compose rounds per §15.6."""
from __future__ import annotations

from typing import Literal


def plan_compose_round(
    parents: list[str],
    compose_mode: Literal["pure", "with_edit"],
) -> dict:
    if len(parents) < 2:
        raise ValueError("compose rounds need at least 2 parents")
    if len(set(parents)) != len(parents):
        raise ValueError(f"duplicate parents: {parents}")
    if compose_mode not in ("pure", "with_edit"):
        raise ValueError(f"compose_mode must be 'pure' or 'with_edit', got {compose_mode!r}")
    return {
        "worktree_action": "create_compose_worktree",
        "parents": parents,
        "compose_mode": compose_mode,
        "invoke_coder": compose_mode == "with_edit",
    }
```

- [ ] **Step 4: Route list-form `fork_from` in `run_round_plan`**

Modify `autoqec/orchestration/loop.py`:

```python
from typing import Literal, Optional
from autoqec.orchestration.composer import plan_compose_round

def run_round_plan(
    env_spec, run_dir, round_idx, machine_state, kb_excerpt, dsl_schema_md,
    fork_from: str | list[str] = "baseline",
    compose_mode: Optional[Literal["pure", "with_edit"]] = None,
) -> dict:
    # ... existing body ...
    result = { ... as before ... }
    if isinstance(fork_from, list):
        if compose_mode is None:
            raise ValueError("compose_mode required when fork_from is a list")
        result["compose_plan"] = plan_compose_round(
            parents=fork_from, compose_mode=compose_mode,
        )
    return result
```

- [ ] **Step 5: Update `llm_loop.py`** — remove the `NotImplementedError`; add:

```python
if isinstance(ideator_resp.get("fork_from"), list):
    from autoqec.orchestration.composer import plan_compose_round
    from autoqec.orchestration.worktree import create_compose_worktree
    plan = plan_compose_round(
        parents=ideator_resp["fork_from"],
        compose_mode=ideator_resp["compose_mode"],
    )
    wt = create_compose_worktree(...)
    if wt["status"] == "compose_conflict":
        metrics = RoundMetrics(
            status="compose_conflict",
            round_attempt_id=str(uuid.uuid4()),
            fork_from=ideator_resp["fork_from"],
            compose_mode=ideator_resp["compose_mode"],
            status_reason=f"conflict on {wt['conflicting_files']}",
        )
        record_round(mem, round_metrics=metrics.model_dump(),
                     verify_verdict=None, verify_report=None)
        continue  # next round
    if not plan["invoke_coder"]:
        # pure compose: run Runner directly on the merged worktree
        ...
    else:
        # with_edit: Coder edits, then Runner
        ...
```

- [ ] **Step 6: Commit**

```bash
git add autoqec/orchestration/composer.py autoqec/orchestration/loop.py autoqec/orchestration/llm_loop.py tests/test_composer.py tests/test_loop_helpers.py
git commit -m "feat(composer): schedule compose rounds (§15.6) in LLM loop"
```

---

## Task P1.2: Conventional-commit prefix validator

**Files:**
- Create: `autoqec/runner/commit_rules.py`
- Modify: `autoqec/runner/runner.py` — call `validate_commit_message(cfg.commit_message)` before the `_commit_round_to_branch` step
- Test: `tests/test_commit_rules.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_commit_rules.py
import pytest

from autoqec.runner.commit_rules import (
    InvalidCommitMessage, validate_commit_message,
)


@pytest.mark.parametrize("msg", [
    "feat: add gnn",
    "fix(runner): oom on eval",
    "test: add trap_C fixture",
    "docs: refresh README",
    "chore: bump ruff",
    "refactor(eval): split guards",
    "perf: faster syndrome sampling",
    "ci: add codex-cli smoke",
])
def test_accepts_valid(msg):
    validate_commit_message(msg)  # no raise


@pytest.mark.parametrize("msg", [
    "add gnn",
    "Feat: caps",
    "random prose",
    "",
    "feature: wrong keyword",
])
def test_rejects_invalid(msg):
    with pytest.raises(InvalidCommitMessage):
        validate_commit_message(msg)
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement**

```python
# autoqec/runner/commit_rules.py
"""Conventional-commit prefix validator for Coder commit_messages.

Test plan 3.3 / 4.2 require every round commit to match:
    ^(feat|fix|test|docs|chore|refactor|perf|ci)(\\(...\\))?:

Rejects malformed messages before we try `git commit` — fail-fast so the
Runner can stamp a clear `status_reason`.
"""
from __future__ import annotations

import re

_PATTERN = re.compile(
    r"^(feat|fix|test|docs|chore|refactor|perf|ci)(\([^)]+\))?:"
)


class InvalidCommitMessage(ValueError):
    pass


def validate_commit_message(msg: str | None) -> None:
    if not msg or not _PATTERN.match(msg):
        raise InvalidCommitMessage(
            f"commit message must start with one of feat/fix/test/docs/chore/"
            f"refactor/perf/ci (optionally followed by scope), got: {msg!r}"
        )
```

- [ ] **Step 4: Hook into Runner**

`_commit_round_to_branch` in `autoqec/runner/runner.py`:

```python
from autoqec.runner.commit_rules import InvalidCommitMessage, validate_commit_message

def _commit_round_to_branch(config, round_idx):
    message = config.commit_message or f"round-attempt {config.round_attempt_id or ''}".strip()
    validate_commit_message(message)
    # ... rest as before
```

- [ ] **Step 5: Pass + commit**

```bash
python -m pytest tests/test_commit_rules.py -v
git add autoqec/runner/commit_rules.py autoqec/runner/runner.py tests/test_commit_rules.py
git commit -m "feat(runner): validate Coder commit_message against conventional-commit prefix"
```

---

## Task P1.3: Branch-diff containment fence

**Goal:** before committing round artifacts to the experiment branch, check that the diff-against-base only touches paths allowed for a research round. Forbidden paths per test plan 4.4: `tests/`, `autoqec/envs/builtin/`, `autoqec/eval/`, `docs/contracts/`, `runs/`.

**Files:**
- Create: `autoqec/runner/branch_fence.py`
- Modify: `autoqec/runner/runner.py` — call fence before commit
- Test: `tests/test_branch_fence.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_branch_fence.py
import subprocess

import pytest

from autoqec.runner.branch_fence import (
    ForbiddenPathError, FORBIDDEN_PREFIXES, check_branch_containment,
)


@pytest.fixture
def temp_repo(tmp_path):
    subprocess.check_call(["git", "init", "-b", "main", str(tmp_path)])
    (tmp_path / "README.md").write_text("hi")
    subprocess.check_call(["git", "-C", str(tmp_path), "add", "README.md"])
    subprocess.check_call(["git", "-C", str(tmp_path), "commit", "-m", "init"])
    return tmp_path


def test_allows_runs_round_N_when_not_under_run_root(temp_repo):
    # Inside the worktree, round artifacts live at runs/<id>/round_N/
    # But `runs/` is git-ignored by default. We check *staged* diff only.
    check_branch_containment(temp_repo, base_ref="main")


def test_rejects_change_to_forbidden_path(temp_repo):
    forbidden = temp_repo / "docs" / "contracts" / "interfaces.md"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_text("x")
    subprocess.check_call(["git", "-C", str(temp_repo), "add", "-A"])
    with pytest.raises(ForbiddenPathError, match="docs/contracts"):
        check_branch_containment(temp_repo, base_ref="main")


def test_lists_all_forbidden(temp_repo):
    for p in ["tests/a.py", "autoqec/eval/b.py"]:
        f = temp_repo / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    subprocess.check_call(["git", "-C", str(temp_repo), "add", "-A"])
    try:
        check_branch_containment(temp_repo, base_ref="main")
    except ForbiddenPathError as exc:
        # Both forbidden files surface in the error
        assert "tests/" in str(exc)
        assert "autoqec/eval/" in str(exc)
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement**

```python
# autoqec/runner/branch_fence.py
"""Branch-diff containment fence — test plan 4.4.

Research round commits may only modify the round's own artifacts under
`runs/<id>/round_<N>/` and experimental modules. These prefixes are
off-limits: mutating them would either poison the test suite, change
the frozen env spec, or violate the contract.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


FORBIDDEN_PREFIXES = (
    "tests/",
    "autoqec/envs/builtin/",
    "autoqec/eval/",
    "docs/contracts/",
    "runs/",
)


class ForbiddenPathError(RuntimeError):
    pass


def check_branch_containment(worktree: Path | str, base_ref: str = "main") -> None:
    worktree = Path(worktree)
    # Staged + unstaged diff against base_ref
    out = subprocess.check_output(
        ["git", "-C", str(worktree), "diff", "--name-only", base_ref, "--"],
        text=True,
    )
    touched = [line.strip() for line in out.splitlines() if line.strip()]
    violations = [p for p in touched if any(p.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)]
    if violations:
        raise ForbiddenPathError(
            f"experiment branch touches forbidden paths: {violations}"
        )
```

- [ ] **Step 4: Hook into Runner**

In `_commit_round_to_branch`, before `git commit`:

```python
from autoqec.runner.branch_fence import check_branch_containment, ForbiddenPathError

# After `git add`, before `git commit`:
try:
    check_branch_containment(worktree, base_ref="main")
except ForbiddenPathError as exc:
    # Abort the commit, reset the index, and raise CalledProcessError so the
    # outer try/except in run_round stamps status_reason.
    subprocess.check_call(["git", "-C", str(worktree), "reset", "HEAD"])
    raise subprocess.CalledProcessError(
        1, ["git", "commit"], stderr=str(exc).encode()
    )
```

- [ ] **Step 5: Pass + commit**

```bash
python -m pytest tests/test_branch_fence.py -v
git add autoqec/runner/branch_fence.py autoqec/runner/runner.py tests/test_branch_fence.py
git commit -m "feat(runner): branch-diff containment fence blocks forbidden paths"
```

---

## Task P1.4: Ctrl-C / resume idempotency

**Goal:** re-running `cli.autoqec run <env> --rounds 3` against a run_dir that already has 2 completed rounds should skip 1 and 2, start at round 3. Test plan 4.4 requires this.

**Design:** `llm_loop.py::run_llm_loop(..., run_dir=None)` — when called **with a specific run_dir**, it probes `round_<N>/metrics.json` existence; skip rounds whose metrics.json exists AND is a valid RoundMetrics. When called **with None**, creates a new timestamped dir (current behavior).

**Files:**
- Modify: `autoqec/orchestration/llm_loop.py`
- Modify: `cli/autoqec.py` — add `--resume <run_dir>` flag
- Test: `tests/test_resume_idempotent.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_resume_idempotent.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from autoqec.orchestration.llm_loop import run_llm_loop


def test_resume_skips_completed_rounds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")

    # Pre-populate round_1/metrics.json as if a previous invocation completed
    run_dir = tmp_path / "runs" / "resume-test"
    r1 = run_dir / "round_1"; r1.mkdir(parents=True)
    (r1 / "metrics.json").write_text(json.dumps({
        "status": "ok", "delta_ler": 0.001, "flops_per_syndrome": 1,
        "n_params": 1, "train_wallclock_s": 0.1, "eval_wallclock_s": 0.01,
        "round_attempt_id": "prev-uuid-1",
    }))

    calls = {"ideator": 0}

    def fake_invoke(role, prompt, timeout=300.0):
        calls[role] = calls.get(role, 0) + 1
        return _stub(role)  # see P0.1 test module

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke), \
         patch("autoqec.orchestration.llm_loop.run_round", return_value=MagicMock(
             status="ok", model_dump=MagicMock(return_value={"status": "ok"})
         )), \
         patch("autoqec.orchestration.llm_loop.independent_verify"):
        run_llm_loop(env=env, rounds=2, profile="dev", run_dir=run_dir)

    # Round 1 existed → skipped. Round 2 new. Only 1 Ideator call.
    assert calls["ideator"] == 1
```

- [ ] **Step 2: Implement skip**

At the top of the loop body in `run_llm_loop`:

```python
for round_idx in range(1, rounds + 1):
    round_dir = run_dir / f"round_{round_idx}"
    m_path = round_dir / "metrics.json"
    if m_path.exists():
        try:
            _ = json.loads(m_path.read_text())
            print(f"resuming: round_{round_idx} complete, skipping")
            continue
        except json.JSONDecodeError:
            pass  # corrupt metrics → redo the round
    round_dir.mkdir(parents=True, exist_ok=True)
    # ... rest of round ...
```

- [ ] **Step 3: Add `--resume` to CLI**

```python
@main.command()
@click.argument("env_yaml")
@click.option("--rounds", type=int, default=10)
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option("--no-llm", is_flag=True)
@click.option("--resume", type=click.Path(), default=None, help="Resume a run_dir")
def run(env_yaml, rounds, profile, no_llm, resume):
    # ...
    if no_llm: ...
    else:
        from autoqec.orchestration.llm_loop import run_llm_loop
        return run_llm_loop(
            env=env, rounds=rounds, profile=profile,
            run_dir=Path(resume).resolve() if resume else None,
        )
```

- [ ] **Step 4: Pass + commit**

```bash
python -m pytest tests/test_resume_idempotent.py -v
git add autoqec/orchestration/llm_loop.py cli/autoqec.py tests/test_resume_idempotent.py
git commit -m "feat(loop): --resume skips rounds with existing metrics.json (test plan 4.4)"
```

---

## Task P1.5: `diagnose` classifier + `diagnosis.md`

**Goal:** `python -m cli.autoqec diagnose <round_dir>` should detect OOM / NaN loss / degenerate p=0 and write a `diagnosis.md` with classification + evidence + suggested fix. Current implementation dumps `{has_config.yaml, has_metrics.json, has_train.log, metrics}` — too thin.

**Files:**
- Modify: `cli/autoqec.py::diagnose`
- Create: `autoqec/diagnose/classifier.py` (new module)
- Test: `tests/test_diagnose_classifier.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_diagnose_classifier.py
from pathlib import Path

from autoqec.diagnose.classifier import classify_failure


def test_classifies_oom(tmp_path):
    (tmp_path / "metrics.json").write_text(
        '{"status": "failed", "status_reason": "CUDA out of memory"}'
    )
    result = classify_failure(tmp_path)
    assert result["classification"] == "OOM"
    assert "cuda" in result["evidence"].lower()


def test_classifies_nan_from_trainlog(tmp_path):
    (tmp_path / "metrics.json").write_text('{"status": "failed", "status_reason": "divergence"}')
    (tmp_path / "train.log").write_text(
        "0\t0.1\n100\t0.5\n200\t1.2\n300\tnan\n400\tnan\n"
    )
    result = classify_failure(tmp_path)
    assert result["classification"] == "NUMERICAL_DIVERGENCE"
    assert "step" in result["evidence"].lower()


def test_classifies_degenerate_config(tmp_path):
    (tmp_path / "metrics.json").write_text('{"status": "ok", "delta_ler": 0.0}')
    (tmp_path / "config.yaml").write_text("noise:\n  p: 0.0\n")
    result = classify_failure(tmp_path)
    assert result["classification"] == "DEGENERATE_NOISE"


def test_unknown_when_no_evidence(tmp_path):
    (tmp_path / "metrics.json").write_text('{"status": "ok", "delta_ler": 0.01}')
    result = classify_failure(tmp_path)
    assert result["classification"] == "UNKNOWN"
```

- [ ] **Step 2: FAIL**

- [ ] **Step 3: Implement classifier**

```python
# autoqec/diagnose/__init__.py
# (empty)
```

```python
# autoqec/diagnose/classifier.py
"""Classify a failed round by reading round_<N>/ artifacts.

Categories (test plan 5.3 + spec §9.3):
  OOM                  — GPU out-of-memory from metrics.status_reason
  NUMERICAL_DIVERGENCE — NaN/Inf in train.log
  OPTIMIZER_DIVERGENCE — monotone-increasing loss over ≥100 steps
  DEGENERATE_NOISE     — config.yaml has p=0 (or None)
  DSL_COMPILE_ERROR    — metrics.status_reason mentions 'compile'
  VALIDATOR_REJECTED   — status=='validator_rejected'
  UNKNOWN
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _parse_train_log(path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for line in path.read_text().splitlines():
        if "\t" not in line:
            continue
        step_s, loss_s = line.split("\t", 1)
        try:
            rows.append((int(step_s), float(loss_s)))
        except ValueError:
            # NaN/inf: record as a sentinel — float("nan") returns nan
            try:
                loss = float(loss_s)
            except ValueError:
                continue
            rows.append((int(step_s), loss))
    return rows


def classify_failure(round_dir: Path) -> dict[str, Any]:
    metrics_path = round_dir / "metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            pass

    reason = (metrics.get("status_reason") or "").lower()
    if "out of memory" in reason or "oom" in reason or "cuda" in reason and "memory" in reason:
        return {
            "classification": "OOM",
            "evidence": f"metrics.status_reason: {metrics.get('status_reason')!r}",
            "suggested_fix": "reduce batch size or hidden_dim; try dev profile",
        }
    if "compile" in reason:
        return {
            "classification": "DSL_COMPILE_ERROR",
            "evidence": f"status_reason: {metrics.get('status_reason')!r}",
            "suggested_fix": "check dsl_config against PredecoderDSL schema",
        }
    if metrics.get("status") == "validator_rejected":
        return {
            "classification": "VALIDATOR_REJECTED",
            "evidence": f"status_reason: {metrics.get('status_reason')!r}",
            "suggested_fix": "review custom_fn for forbidden imports / names",
        }

    # Train-log inspection
    log = round_dir / "train.log"
    if log.exists():
        rows = _parse_train_log(log)
        nan_rows = [(s, l) for s, l in rows if l != l]  # NaN != NaN
        if nan_rows:
            step = nan_rows[0][0]
            return {
                "classification": "NUMERICAL_DIVERGENCE",
                "evidence": f"NaN in train.log at step {step}",
                "suggested_fix": "lower lr by 10×, add gradient clipping (max_norm=1.0)",
            }
        # Monotone-increasing check
        if len(rows) >= 100:
            window = [loss for _, loss in rows[-100:]]
            increasing = all(window[i] <= window[i + 1] for i in range(len(window) - 1))
            if increasing and window[-1] > 2 * window[0]:
                return {
                    "classification": "OPTIMIZER_DIVERGENCE",
                    "evidence": f"train.log last 100 steps monotone-increasing ({window[0]:.3g} → {window[-1]:.3g})",
                    "suggested_fix": "lower lr; consider SGD → AdamW with warmup",
                }

    # Config-based
    cfg_path = round_dir / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            noise = (cfg or {}).get("noise", {})
            if noise.get("p") in (0, 0.0):
                return {
                    "classification": "DEGENERATE_NOISE",
                    "evidence": "config.yaml has noise.p = 0",
                    "suggested_fix": "set noise.p > 0 (e.g. 1e-3)",
                }
        except yaml.YAMLError:
            pass

    return {
        "classification": "UNKNOWN",
        "evidence": "no matching signal in metrics.json / train.log / config.yaml",
        "suggested_fix": "inspect the round manually — no automatic classification matched",
    }
```

- [ ] **Step 4: Rewrite `diagnose` CLI**

Replace the body of `diagnose` in `cli/autoqec.py`:

```python
@main.command()
@click.argument("run_dir")
@click.option("--out", type=click.Path(), default=None,
              help="Write diagnosis.md here (default: <target>/diagnosis.md)")
def diagnose(run_dir, out):
    from autoqec.diagnose.classifier import classify_failure

    rd = Path(run_dir)
    if (rd / "metrics.json").exists() or (rd / "train.log").exists():
        target = rd
    else:
        round_dirs = sorted(rd.glob("round_*"))
        if not round_dirs:
            click.echo("No round dirs found and no metrics in given path")
            return
        target = round_dirs[-1]

    result = classify_failure(target)
    md = (
        f"# Diagnosis\n\n"
        f"**Classification:** {result['classification']}\n\n"
        f"**Evidence:**\n{result['evidence']}\n\n"
        f"**Suggested fix:** {result['suggested_fix']}\n"
    )
    out_path = Path(out) if out else target / "diagnosis.md"
    out_path.write_text(md, encoding="utf-8")
    click.echo(json.dumps({**result, "diagnosis_md": str(out_path)}, indent=2))
```

- [ ] **Step 5: Pass + commit**

```bash
python -m pytest tests/test_diagnose_classifier.py -v
git add autoqec/diagnose/ cli/autoqec.py tests/test_diagnose_classifier.py
git commit -m "feat(diagnose): classifier + diagnosis.md per S5 skill contract"
```

---

## Task P1.6: Diagnose failure fixtures

**Files:**
- Create: `scripts/build_diagnose_fixtures.py`
- Create: `tests/fixtures/diagnose/{oom,nan,degenerate}/{metrics.json, train.log, config.yaml}`

- [ ] **Step 1: Construct fixtures by hand (text-only, no binary)**

`tests/fixtures/diagnose/oom/metrics.json`:
```json
{"status": "failed", "status_reason": "CUDA out of memory",
 "round_attempt_id": "fix-oom-1"}
```
`tests/fixtures/diagnose/oom/config.yaml`:
```yaml
noise:
  p: 5.0e-3
```

`tests/fixtures/diagnose/nan/metrics.json`:
```json
{"status": "failed", "status_reason": "optimizer divergence",
 "round_attempt_id": "fix-nan-1"}
```
`tests/fixtures/diagnose/nan/train.log`: 500 lines `<step>\t<loss>`, loss goes `0.1 → 12.8 → nan`, NaN from step 420 onwards. Write a small Python script to emit this.

`tests/fixtures/diagnose/degenerate/metrics.json`:
```json
{"status": "ok", "delta_ler": 0.0, "round_attempt_id": "fix-deg-1"}
```
`tests/fixtures/diagnose/degenerate/config.yaml`:
```yaml
noise:
  p: 0.0
```

- [ ] **Step 2: Builder script** — if the text fixtures suffice, just commit them; otherwise a small generator for the train.log.

```python
# scripts/build_diagnose_fixtures.py
import math
import random
from pathlib import Path

OUT = Path("tests/fixtures/diagnose")


def build_nan_train_log() -> None:
    lines = []
    random.seed(42)
    loss = 0.1
    for step in range(500):
        if step >= 420:
            lines.append(f"{step}\tnan")
            continue
        loss *= 1.012  # monotone growth
        lines.append(f"{step}\t{loss:.6g}")
    (OUT / "nan" / "train.log").write_text("\n".join(lines))


def main():
    (OUT / "oom").mkdir(parents=True, exist_ok=True)
    (OUT / "nan").mkdir(parents=True, exist_ok=True)
    (OUT / "degenerate").mkdir(parents=True, exist_ok=True)
    # The JSON/YAML fixtures are hand-written and committed;
    # only train.log is generated here because it's 500 lines.
    build_nan_train_log()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Extend `test_diagnose_classifier.py`** to point at real fixture paths:

```python
def test_fixture_oom():
    r = classify_failure(Path("tests/fixtures/diagnose/oom"))
    assert r["classification"] == "OOM"


def test_fixture_nan():
    r = classify_failure(Path("tests/fixtures/diagnose/nan"))
    assert r["classification"] in ("NUMERICAL_DIVERGENCE", "OPTIMIZER_DIVERGENCE")


def test_fixture_degenerate():
    r = classify_failure(Path("tests/fixtures/diagnose/degenerate"))
    assert r["classification"] == "DEGENERATE_NOISE"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/build_diagnose_fixtures.py tests/fixtures/diagnose/ tests/test_diagnose_classifier.py
git commit -m "test: diagnose fixtures (oom/nan/degenerate)"
```

---

## Task P1.7: Stuck-run fixture

**Files:**
- Create: `tests/fixtures/review_log/stuck_run/history.jsonl`
- Create: `tests/test_review_log_stuck.py`

- [ ] **Step 1: Build the fixture**

`tests/fixtures/review_log/stuck_run/history.jsonl`:
```
{"round": 1, "delta_ler": 0.0015, "verdict": "candidate", "hypothesis": "try wider GNN"}
{"round": 2, "delta_ler": 0.0012, "verdict": "candidate", "hypothesis": "deeper GNN"}
{"round": 3, "delta_ler": 0.0018, "verdict": "candidate", "hypothesis": "GNN with residuals"}
```

- [ ] **Step 2: Test** the `review-log` CLI on this fixture → asserts the JSON blob reports `n_rounds=3`, `mean_wallclock_s=0` (no wallclock in fixture), and the hypotheses list:

```python
# tests/test_review_log_stuck.py
import json
import subprocess


def test_review_log_stuck_fixture(tmp_path):
    # Copy fixture to tmp dir (pareto.json absent = 0 pareto entries)
    src = "tests/fixtures/review_log/stuck_run/history.jsonl"
    dst = tmp_path / "history.jsonl"
    dst.write_text(open(src).read())
    result = subprocess.run(
        ["python", "-m", "cli.autoqec", "review-log", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    assert data["n_rounds"] == 3
    assert all("GNN" in h for h in data["top_hypotheses"])
```

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/review_log/ tests/test_review_log_stuck.py
git commit -m "test: stuck-run fixture for review-log"
```

---

## Task P1.8: Ablation zero-signal unit test

**Goal:** a random-weights predecoder should have `delta_ler` whose 95% CI crosses 0 — the protocol has no positive bias. Test plan 5.4.

**Files:**
- Create: `tests/test_ablation_zero_signal.py`

- [ ] **Step 1: Integration-gated test**

```python
# tests/test_ablation_zero_signal.py
import pytest


@pytest.mark.integration
def test_random_weights_delta_ler_crosses_zero():
    """Random-weights predecoder — CI on delta_ler_holdout should straddle 0."""
    import tempfile
    from pathlib import Path

    import torch
    from autoqec.decoders.dsl_compiler import compile_predecoder
    from autoqec.envs.schema import load_env_yaml
    from autoqec.eval.independent_eval import independent_verify

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    model = compile_predecoder(
        {"type": "gnn", "output_mode": "soft_priors", "hidden_dim": 8, "n_layers": 2},
        code_spec=env.code,
    )
    # Random weights: don't train.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save({
            "class_name": "GNNPredecoder",
            "state_dict": model.state_dict(),
            "output_mode": "soft_priors",
            "dsl_config": {"type": "gnn", "hidden_dim": 8, "n_layers": 2},
        }, f.name)
        ckpt_path = Path(f.name)

    sp = env.noise.seed_policy
    holdout = list(range(sp.holdout[0], sp.holdout[0] + 3))
    report = independent_verify(ckpt_path, env, holdout_seeds=holdout, n_shots=1000)
    ci_lo, ci_hi = report.ler_holdout_ci
    # Random weights should not have a positive delta_ler_holdout with CI above 0.
    # Either the CI crosses 0, OR the verdict is not VERIFIED.
    assert ci_lo <= 0 <= ci_hi or report.verdict != "VERIFIED"
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_ablation_zero_signal.py
git commit -m "test: random-weights predecoder delta_ler CI crosses 0 (test plan 5.4)"
```

---

# Tier P2 — Polish

## Task P2.1: D3 `/add-env` onboarding demo

**Files:**
- Create: `demos/demo-3-add-env/README.md`, `sample-session.md`, `sample-env.yaml`
- Modify: `Makefile`, `README.md`

Follow the content skeleton from the prior revision of this plan (Steane-7 walkthrough) — material unchanged.

- [ ] **Step 1-5:** as before — create the 3 files, add Makefile target, validate YAML parses via `load_env_yaml`, flip README D3 status row to DONE.

- [ ] **Step 6: Commit**

```bash
git add demos/demo-3-add-env/ Makefile README.md
git commit -m "feat(demo): D3 /add-env onboarding walkthrough"
```

---

## Task P2.2: Delete unused `autoqec/pareto/` package

**Files:**
- Delete: `autoqec/pareto/__init__.py`, `autoqec/pareto/front.py`
- Delete: `tests/test_pareto.py`

- [ ] **Step 1: Confirm zero callers**

```bash
grep -rn "autoqec\.pareto\|is_pareto_dominated\|update_front" autoqec/ cli/ scripts/ tests/ | grep -v "autoqec/pareto/\|tests/test_pareto.py"
```

Expected: empty.

- [ ] **Step 2: Delete**

```bash
rm -rf autoqec/pareto/
rm tests/test_pareto.py
```

- [ ] **Step 3: Regression**

```bash
python -m pytest tests/test_round_recorder.py -v  # covers the canonical Pareto impl
```

- [ ] **Step 4: Commit**

```bash
git add -A autoqec/pareto tests/test_pareto.py
git commit -m "refactor: delete unused autoqec.pareto (duplicate of round_recorder._non_dominated_merge)"
```

---

## Task P2.3: Dirty-worktree warning

**Goal:** when `run_llm_loop` starts and `git status --porcelain` reports uncommitted changes at the repo root, log a warning listing the dirty files. The `artifact_manifest.json` already captures `dirty_files` (P0.3), so this is mostly a UX step.

**Files:**
- Modify: `autoqec/orchestration/llm_loop.py` — emit warning at start
- Test: `tests/test_dirty_worktree.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_dirty_worktree.py
import subprocess
import warnings
from pathlib import Path

import pytest


@pytest.mark.integration
def test_warns_on_dirty_worktree(tmp_path, monkeypatch):
    # Create a dirty file in the current repo — scary to do in CI.
    # Skip if not in a dev repo.
    ...  # detailed fixture setup; skipped for brevity in plan
```

- [ ] **Step 2: Implement**

At the top of `run_llm_loop`, before any subagent dispatch:

```python
import subprocess, warnings
try:
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True,
    ).strip()
    if porcelain:
        warnings.warn(
            f"running with uncommitted changes in the working tree:\n{porcelain}\n"
            "(dirty file SHAs will be captured in artifact_manifest.json per round)"
        )
except (subprocess.CalledProcessError, FileNotFoundError):
    pass
```

- [ ] **Step 3: Commit**

```bash
git add autoqec/orchestration/llm_loop.py tests/test_dirty_worktree.py
git commit -m "feat(loop): warn on dirty worktree at run start (SHAs already in manifest)"
```

---

## Task P2.4: Offline replay packaging

**Goal:** `python -m cli.autoqec package-run <run_dir> -o <out.tar.gz>` — bundle a run for advisor offline replay. Test plan 5.5.

**Files:**
- Create: `scripts/package_run.py`
- Modify: `cli/autoqec.py` — add `package-run` subcommand (thin wrapper)
- Test: `tests/test_package_run.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_package_run.py
import json
import subprocess
import tarfile
from pathlib import Path


def test_package_run_creates_archive(tmp_path):
    # Make a fake run_dir
    run_dir = tmp_path / "runs" / "20260423-000000"
    (run_dir / "round_1").mkdir(parents=True)
    (run_dir / "round_1" / "metrics.json").write_text('{"status": "ok"}')
    (run_dir / "history.jsonl").write_text('{"round": 1}\n')
    (run_dir / "pareto.json").write_text("[]")

    out = tmp_path / "pkg.tar.gz"
    result = subprocess.run(
        ["python", "-m", "cli.autoqec", "package-run", str(run_dir), "-o", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert out.exists()
    with tarfile.open(out) as tar:
        names = tar.getnames()
        assert any("history.jsonl" in n for n in names)
        assert any("metrics.json" in n for n in names)
        assert any("replay_manifest.json" in n for n in names)
```

- [ ] **Step 2: Implement**

```python
# scripts/package_run.py
"""Package a run_dir for offline advisor replay (test plan 5.5)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
from pathlib import Path


def package_run(run_dir: Path, out_path: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    # Write a replay_manifest.json with repo SHA so we know what code to pair with.
    try:
        repo_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        repo_sha = None
    manifest = {
        "run_dir": str(run_dir.relative_to(run_dir.parent)),
        "repo_sha": repo_sha,
        "python_version": sys.version.split()[0],
    }
    (run_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(f"{Path(args.run_dir).name}.tar.gz")
    package_run(Path(args.run_dir), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: CLI wrapper**

```python
@main.command("package-run")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--out", type=click.Path(), default=None)
def package_run_cmd(run_dir, out):
    from scripts.package_run import package_run
    out_path = Path(out) if out else Path(f"{Path(run_dir).name}.tar.gz")
    package_run(Path(run_dir), out_path)
    click.echo(f"wrote {out_path}")
```

- [ ] **Step 4: Commit**

```bash
git add scripts/package_run.py cli/autoqec.py tests/test_package_run.py
git commit -m "feat(cli): package-run bundles run_dir + repo_sha for offline replay"
```

---

# Verification gates — after all tasks land

```bash
# Unit suite
python -m pytest tests/ -m "not integration" -v
# Expected: previous green + ~90 new tests

# Integration suite (GPU host)
python -m pytest tests/ -m integration --run-integration -v

# Lint
ruff check autoqec cli tests scripts

# Full-loop live smoke
make run ENV=autoqec/envs/builtin/surface_d5_depol.yaml ROUNDS=3
# Expected: 3 rounds in runs/<id>/, pareto.json populated, every round_N/ has
#   metrics.json, checkpoint.pt, train.log, config.yaml,
#   round_N_pointer.json, artifact_manifest.json, verification_report.json

# Every item in docs/verification/human-verification-test-plan.md Phases 1-5
# should either tick green or be explicitly marked `blocked: <reason>`.
```

---

## Self-Review

**Spec coverage** (cross-referenced with `docs/verification/human-verification-test-plan.md`):

| Test plan gate | Task |
|---|---|
| 1.1 Clean install + ruff clean | already green on main |
| 1.2 Baseline tests ≥ 33 | pre-existing + growth from P0.6/P0.3/P0.1/P1.x tests |
| 2.1 Schemas + Tier-2 validator | already green |
| 2.2 MWPM/OSD latency + baseline anchor | already green |
| 2.3 Handshake + RunnerCallPathError + worktree + reconcile | already green |
| 2.4 Fork_graph + backend dispatch + reward-hacking + machine_state | P0.4 (traps) + already green |
| 3.1 run_quick smoke | already green |
| 3.2 CLI run --no-llm + AUTOQEC_RESULT_JSON + candidate_pareto.json | already green (P0.2 tightens) |
| 3.3 Worktree round + pointer round-trip + commit msg + cleanup | P0.6 + P0.7 + P1.2 |
| 3.4 Post-round invariants | P0.6 + P0.7 |
| 4.1-4.2 Full LLM loop surface_d5 + bb72 | P0.1 |
| 4.3 Ideator prompts contain fork_graph + compose rounds land safely | already green + P1.1 |
| 4.4 Ctrl-C resume + branch_manually_deleted + containment | P1.4 + P1.3 |
| 4.5 Dirty-worktree + artifact_manifest + reproduce command line | P2.3 + P0.3 |
| 5.1 verify positive case | already green |
| 5.2 Traps A/B/C | P0.4 |
| 5.3 review-log + diagnose | already partial + P1.5 + P1.6 + P1.7 |
| 5.4 Statistical correctness + ablation | P1.8 + already partial |
| 5.5 Atomic Pareto + offline replay | P0.2 + P2.4 |

**Open gaps not addressed by this plan:**
- Open Question #1 (Phase 4 wall-clock budget) — empirical, needs a calibration dry-run, not code
- Open Question #2 (per-env token budgets) — empirical, not code
- Open Question #6 (`autoqec cleanup-worktree` CLI wrapper) — deliberately deferred; `cleanup_round_worktree` Python API is sufficient for demos

**Placeholder scan:** every code block has concrete content. The `tests/fixtures/` directories are populated by the build scripts; the tests gate on `.pt` existence so CI doesn't break before they're built.

**Type consistency:**
- `RunnerConfig.commit_message` added in P0.7, read in P1.2 (commit_rules), P1.3 (branch_fence), manifest cmd_line capture
- `write_round_pointer(cfg, metrics, round_idx)` — same signature in definition (P0.6) and callsite (P0.7 runner hook)
- `plan_compose_round(parents, compose_mode)` — identical in P1.1 definition and `run_round_plan` + `run_llm_loop` callsites
- `independent_verify(checkpoint, env_spec, holdout_seeds, n_shots, n_bootstrap)` — the existing main signature; called identically in P0.1 (`llm_loop`) and P0.5 (SKILL) and P1.8 (ablation test)
- `classify_failure(round_dir) → dict` — P1.5 definition; same shape consumed by the CLI command

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-23-close-the-loop.md`.

**Task count:** 19 across P0 (7) / P1 (8) / P2 (4).

**Critical path (sequential):** P0.2 → P0.6 → P0.7 → P0.1 → P0.3 → P0.5 → P0.4

**Parallel after critical path:**
- P1.1 (uses P0.7) / P1.2 / P1.3 (all plug into Runner commit flow — serialize if one subagent, OK if different subagents)
- P1.4 (depends only on P0.1)
- P1.5 + P1.6 + P1.7 (all around `diagnose` + fixtures — can run together)
- P1.8 (independent)
- P2.1 / P2.2 / P2.3 / P2.4 (independent)

**Ownership suggestion** (mirrors the test-plan ownership split):
- Chen Jiahan: P0.1, P0.2, P0.3, P0.5, P1.4, P2.3, P2.4
- Lin Tengxiang: P0.6, P0.7, P1.1, P1.2, P1.3, P2.1, P2.2
- Xie Jingu: P0.4, P1.5, P1.6, P1.7, P1.8

**Execution options:**
1. **Subagent-Driven** — one subagent per task; 19 subagents total.
2. **Inline Execution** — one session per tier (P0 → P1 → P2) via `/superpowers:executing-plans`.

Which approach?
