# Owner-C Remaining Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining C-owner gaps by making the CLI run paths robust outside the repo root, emitting a run-level `pareto.json` artifact for Demo 2, and hardening regression coverage around the C-owned execution path.

**Architecture:** Reuse the existing `RunMemory` L1 artifact contract instead of inventing a parallel storage path. Fix path robustness at the loader/CLI boundary so `run` and `run-round` behave the same from foreign working directories, then add targeted regression tests for CLI artifact creation, OSD backend decoding, and Runner output contracts.

**Tech Stack:** Python 3.12, click, pytest, PyYAML, Stim, PyMatching, ldpc, torch

---

### Task 1: Make env and CLI paths robust outside the repo root

**Files:**
- Modify: `autoqec/envs/schema.py`
- Modify: `cli/autoqec.py`
- Test: `tests/test_cli_run_paths.py`

- [ ] **Step 1: Write the failing regression test for `run` from a foreign cwd**

Create `tests/test_cli_run_paths.py` with a subprocess-based regression that runs the CLI from a temporary directory:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_cli_works_from_foreign_cwd_and_writes_pareto(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.autoqec",
            "run",
            str(env_yaml),
            "--rounds",
            "1",
            "--profile",
            "dev",
            "--no-llm",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    run_dir = tmp_path / payload["run_dir"]
    assert run_dir.exists()
    assert (run_dir / "history.json").exists()
    assert (run_dir / "history.jsonl").exists()
    assert (run_dir / "pareto.json").exists()
```

- [ ] **Step 2: Run the new test and confirm it fails on current code**

Run:

```bash
./.venv/bin/pytest tests/test_cli_run_paths.py::test_run_cli_works_from_foreign_cwd_and_writes_pareto -v
```

Expected: FAIL, currently because `cli.autoqec run` cannot find templates from a foreign cwd and does not write `pareto.json`.

- [ ] **Step 3: Implement path resolution in the env loader**

Update `autoqec/envs/schema.py` so relative `code.source` values are resolved against the YAML file location:

```python
def load_env_yaml(path: str | Path) -> EnvSpec:
    path = Path(path).resolve()
    with path.open() as f:
        data = yaml.safe_load(f)
    code = data.get("code", {})
    source = code.get("source")
    if source and not Path(source).is_absolute():
        code["source"] = str((path.parent / source).resolve())
    data["code"] = code
    return EnvSpec(**data)
```

- [ ] **Step 4: Implement repo-root template discovery and run artifact writing**

Update `cli/autoqec.py`:

```python
from autoqec.orchestration.memory import RunMemory


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_pareto(records: list[dict]) -> list[dict]:
    candidates = []
    for idx, record in enumerate(records, start=1):
        if record.get("status") != "ok":
            continue
        if any(record.get(k) is None for k in ("delta_ler", "flops_per_syndrome", "n_params")):
            continue
        candidates.append(
            {
                "round": idx,
                "delta_ler": record["delta_ler"],
                "flops_per_syndrome": record["flops_per_syndrome"],
                "n_params": record["n_params"],
                "checkpoint_path": record.get("checkpoint_path"),
                "verified": False,
            }
        )

    front = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            no_worse = (
                other["delta_ler"] >= cand["delta_ler"]
                and other["flops_per_syndrome"] <= cand["flops_per_syndrome"]
                and other["n_params"] <= cand["n_params"]
            )
            strictly_better = (
                other["delta_ler"] > cand["delta_ler"]
                or other["flops_per_syndrome"] < cand["flops_per_syndrome"]
                or other["n_params"] < cand["n_params"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(cand)

    front.sort(key=lambda item: (-item["delta_ler"], item["flops_per_syndrome"], item["n_params"]))
    return front
```

And inside `run(...)`:

```python
    repo_root = _repo_root()
    templates = sorted((repo_root / "autoqec/example_db").glob("*.yaml"))
    mem = RunMemory(run_dir)
```

After each round:

```python
        record = metrics.model_dump()
        history.append(record)
        mem.append_round(record)
        mem.update_pareto(_candidate_pareto(history))
```

At the end:

```python
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    click.echo(json.dumps({"run_dir": str(run_dir), "rounds": rounds, "pareto_path": str(run_dir / "pareto.json")}, indent=2))
```

- [ ] **Step 5: Re-run the CLI regression test**

Run:

```bash
./.venv/bin/pytest tests/test_cli_run_paths.py::test_run_cli_works_from_foreign_cwd_and_writes_pareto -v
```

Expected: PASS.

### Task 2: Harden backend and runner contract tests

**Files:**
- Modify: `tests/test_backend_adapter.py`
- Modify: `tests/test_runner_smoke.py`

- [ ] **Step 1: Add OSD backend coverage**

Extend `tests/test_backend_adapter.py` with two tests:

```python
import numpy as np


def test_soft_priors_drive_osd_decode_shape() -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    parity = np.load(env.code.source)
    syndrome = np.zeros((3, parity.shape[0]), dtype=np.uint8)
    priors = np.full((3, parity.shape[1]), 0.05, dtype=float)
    out = decode_with_predecoder(priors, env, syndrome, parity, "soft_priors")
    assert out.shape == (3, parity.shape[1])


def test_hard_flip_drives_osd_decode_shape() -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    parity = np.load(env.code.source)
    cleaned = np.zeros((2, parity.shape[0]), dtype=np.uint8)
    out = decode_with_predecoder(cleaned, env, cleaned, parity, "hard_flip")
    assert out.shape == (2, parity.shape[1])
```

- [ ] **Step 2: Add runner artifact assertions**

Extend `tests/test_runner_smoke.py`:

```python
    round_dir = tmp_path / "round_0"
    metrics = run_round(cfg, env)
    assert metrics.status == "ok"
    assert round_dir.joinpath("metrics.json").exists()
    assert round_dir.joinpath("checkpoint.pt").exists()
    assert round_dir.joinpath("train.log").exists()
    assert metrics.checkpoint_path is not None
    assert metrics.training_log_path is not None
```

- [ ] **Step 3: Run the focused tests**

Run:

```bash
./.venv/bin/pytest tests/test_backend_adapter.py tests/test_runner_smoke.py -v --run-integration
```

Expected: PASS.

### Task 3: Polish Demo 2 docs and run the full verification set

**Files:**
- Modify: `demos/demo-2-bb72/README.md`

- [ ] **Step 1: Update Demo 2 README to match actual artifact**

Adjust the README so it describes the branch honestly:

```markdown
- Fast mode: all 3 rounds complete and `pareto.json` is emitted.
- The current branch writes an unverified candidate Pareto summary for demo/reporting.
- Verification-admitted Pareto maintenance remains the verification slice's responsibility.
```

- [ ] **Step 2: Run the main verification commands**

Run:

```bash
make lint
make test
./.venv/bin/pytest tests/test_runner_smoke.py -q -m integration --run-integration
./.venv/bin/pytest tests/test_cli_run_paths.py::test_run_cli_works_from_foreign_cwd_and_writes_pareto -v
MODE=fast bash demos/demo-2-bb72/run.sh
```

Expected:

- lint passes
- non-integration tests pass
- integration smoke passes
- CLI foreign-cwd regression passes
- Demo 2 fast mode prints a real `pareto.json` instead of `(no pareto yet)`

- [ ] **Step 3: Commit**

```bash
git add autoqec/envs/schema.py cli/autoqec.py demos/demo-2-bb72/README.md tests/test_backend_adapter.py tests/test_runner_smoke.py tests/test_cli_run_paths.py docs/superpowers/plans/2026-04-22-owner-c-remaining-completion.md
git commit -m "feat: finish owner-c cli and demo polish"
```
