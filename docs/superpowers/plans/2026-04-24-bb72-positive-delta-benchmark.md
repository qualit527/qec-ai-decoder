# BB72 Positive Delta Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible BB72/OSD benchmark track that produces compact evidence for positive `delta_ler` across a fixed round schedule.

**Architecture:** Add a new experiment directory under `experiments/bb72-positive-delta/` and keep the Runner as the source of truth for training and metrics. The new driver sequences fixed YAML configs, calls `run_round()`, writes `summary.json` and `report.md`, and fails if it cannot show a strictly positive best round.

**Tech Stack:** Python 3.12, pytest, PyYAML, Pydantic `EnvSpec`, `RunnerConfig`, `RoundMetrics`, existing `autoqec.runner.runner.run_round()`.

---

## File Structure

- Create `autoqec/envs/builtin/bb72_perf.yaml` for the performance environment.
- Create `experiments/bb72-positive-delta/configs/round_1_gnn_small.yaml` for the deliberately small first round.
- Create `experiments/bb72-positive-delta/configs/round_2_gnn_gated.yaml` for a stronger GNN round.
- Create `experiments/bb72-positive-delta/configs/round_3_neural_bp.yaml` for a different-family final round.
- Create `experiments/bb72-positive-delta/run.py` for benchmark orchestration and summary/report generation.
- Create `experiments/bb72-positive-delta/README.md` for reviewer-facing usage and interpretation.
- Create `experiments/bb72-positive-delta/expected_output/summary.json` after a successful manual benchmark run.
- Create `tests/test_bb72_positive_delta_assets.py` for env/config/doc/snapshot assets.
- Create `tests/test_bb72_positive_delta_benchmark.py` for mocked driver behavior and failure gates.

---

### Task 1: Add Benchmark Environment And Config Assets

**Files:**
- Create: `autoqec/envs/builtin/bb72_perf.yaml`
- Create: `experiments/bb72-positive-delta/configs/round_1_gnn_small.yaml`
- Create: `experiments/bb72-positive-delta/configs/round_2_gnn_gated.yaml`
- Create: `experiments/bb72-positive-delta/configs/round_3_neural_bp.yaml`
- Create: `tests/test_bb72_positive_delta_assets.py`

- [ ] **Step 1: Write the failing asset tests**

Create `tests/test_bb72_positive_delta_assets.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from autoqec.decoders.dsl_compiler import compile_predecoder
from autoqec.envs.schema import load_env_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / "autoqec/envs/builtin/bb72_perf.yaml"
CONFIG_DIR = REPO_ROOT / "experiments/bb72-positive-delta/configs"


def test_bb72_perf_env_uses_osd_and_nonzero_error_budget() -> None:
    env = load_env_yaml(ENV_PATH)

    assert env.name == "bb72_perf"
    assert env.code.type == "parity_check_matrix"
    assert Path(env.code.source).name == "bb72_Hx.npy"
    assert env.classical_backend == "osd"
    assert env.noise.p[0] == 0.05
    assert env.eval_protocol.osd_orders_reported == [0]
    assert env.eval_protocol.min_shots_val >= 4096


def test_positive_delta_round_configs_compile() -> None:
    config_paths = sorted(CONFIG_DIR.glob("round_*.yaml"))

    assert [path.name for path in config_paths] == [
        "round_1_gnn_small.yaml",
        "round_2_gnn_gated.yaml",
        "round_3_neural_bp.yaml",
    ]
    for path in config_paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = compile_predecoder(cfg, n_var=72, n_check=36)
        assert model.output_mode == "soft_priors"
        assert sum(parameter.numel() for parameter in model.parameters()) > 0
        assert cfg["training"]["epochs"] >= 3
```

- [ ] **Step 2: Run the asset tests and verify they fail**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_assets.py -v
```

Expected: FAIL because `bb72_perf.yaml` and the `experiments/bb72-positive-delta/configs/` directory do not exist.

- [ ] **Step 3: Add the benchmark env**

Create `autoqec/envs/builtin/bb72_perf.yaml`:

```yaml
name: bb72_perf
code:
  type: parity_check_matrix
  source: circuits/bb72_Hx.npy
noise:
  type: depolarizing
  p: [5.0e-2]
  seed_policy:
    train: [1, 999]
    val: [1000, 1999]
    holdout: [9000, 9999]
constraints:
  latency_flops_budget: 50000000
  param_budget: 500000
  target_ler: 1.0e-4
  target_p: 5.0e-2
baseline_decoders:
  - bposd
classical_backend: osd
eval_protocol:
  min_shots_train: 8192
  min_shots_val: 4096
  min_shots_verify: 200000
  bootstrap_ci: 0.95
  osd_orders_reported: [0]
  x_z_decoding: x_only
```

- [ ] **Step 4: Add the fixed round configs**

Create `experiments/bb72-positive-delta/configs/round_1_gnn_small.yaml`:

```yaml
type: gnn
output_mode: soft_priors
gnn:
  layers: 1
  hidden_dim: 16
  message_fn: mlp
  aggregation: sum
  normalization: layer
  residual: false
  edge_features: [syndrome_bit]
head: linear
training:
  learning_rate: 1.0e-3
  batch_size: 64
  epochs: 3
  loss: bce
  profile: benchmark
```

Create `experiments/bb72-positive-delta/configs/round_2_gnn_gated.yaml`:

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
head: mlp_small
training:
  learning_rate: 1.0e-3
  batch_size: 64
  epochs: 6
  loss: bce
  profile: benchmark
```

Create `experiments/bb72-positive-delta/configs/round_3_neural_bp.yaml`:

```yaml
type: neural_bp
output_mode: soft_priors
neural_bp:
  iterations: 5
  weight_sharing: per_layer
  damping: learnable_per_iter
  attention_aug: false
  attention_heads: 1
head: linear
training:
  learning_rate: 3.0e-4
  batch_size: 64
  epochs: 6
  loss: bce
  profile: benchmark
```

- [ ] **Step 5: Run the asset tests and verify they pass**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_assets.py -v
```

Expected: PASS for both tests in `tests/test_bb72_positive_delta_assets.py`.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add autoqec/envs/builtin/bb72_perf.yaml experiments/bb72-positive-delta/configs tests/test_bb72_positive_delta_assets.py
git commit -m "feat: add bb72 positive delta benchmark assets"
```

---

### Task 2: Add Benchmark Driver Success Path

**Files:**
- Create: `experiments/bb72-positive-delta/run.py`
- Modify: `tests/test_bb72_positive_delta_benchmark.py`

- [ ] **Step 1: Write the failing driver success test**

Create `tests/test_bb72_positive_delta_benchmark.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from autoqec.runner.schema import RoundMetrics


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "experiments/bb72-positive-delta/run.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("bb72_positive_delta_run", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_benchmark_writes_positive_summary_and_report(monkeypatch, tmp_path: Path) -> None:
    module = _load_benchmark_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for idx in range(1, 4):
        (config_dir / f"round_{idx}.yaml").write_text(
            "type: gnn\n"
            "output_mode: soft_priors\n"
            "gnn:\n"
            "  layers: 1\n"
            "  hidden_dim: 4\n"
            "  message_fn: mlp\n"
            "  aggregation: sum\n"
            "  normalization: layer\n"
            "  residual: false\n"
            "  edge_features: [syndrome_bit]\n"
            "head: linear\n"
            "training:\n"
            "  learning_rate: 0.001\n"
            "  batch_size: 2\n"
            "  epochs: 1\n"
            "  loss: bce\n",
            encoding="utf-8",
        )
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("name: fake\n", encoding="utf-8")

    deltas = iter([-0.01, 0.0, 0.02])

    def fake_load_env_yaml(path):
        assert path == env_yaml
        return type("Env", (), {"name": "bb72_perf"})()

    def fake_run_round(config, env):
        delta = next(deltas)
        return RoundMetrics(
            status="ok",
            ler_plain_classical=0.12,
            ler_predecoder=0.12 - delta,
            delta_ler=delta,
            flops_per_syndrome=1000 + int(config.seed),
            n_params=2000 + int(config.seed),
            train_wallclock_s=1.5,
            eval_wallclock_s=0.25,
            checkpoint_path=str(Path(config.round_dir) / "checkpoint.pt"),
            training_log_path=str(Path(config.round_dir) / "train.log"),
        )

    monkeypatch.setattr(module, "load_env_yaml", fake_load_env_yaml)
    monkeypatch.setattr(module, "run_round", fake_run_round)
    monkeypatch.setattr(module, "_run_id", lambda: "test-run")

    run_dir = module.run_benchmark(
        env_yaml=env_yaml,
        config_dir=config_dir,
        output_root=tmp_path / "runs",
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert summary["run_id"] == "test-run"
    assert summary["has_positive_delta"] is True
    assert summary["best_delta_ler"] == 0.02
    assert summary["best_round"] == 3
    assert [row["best_delta_ler_so_far"] for row in summary["rounds"]] == [-0.01, 0.0, 0.02]
    assert "benchmark evidence" in report
    assert "not a VERIFIED holdout claim" in report
```

- [ ] **Step 2: Run the driver success test and verify it fails**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_benchmark.py::test_run_benchmark_writes_positive_summary_and_report -v
```

Expected: FAIL because `experiments/bb72-positive-delta/run.py` does not exist.

- [ ] **Step 3: Add `run.py` with the success path**

Create `experiments/bb72-positive-delta/run.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from autoqec.envs.schema import load_env_yaml
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RoundMetrics, RunnerConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_YAML = REPO_ROOT / "autoqec/envs/builtin/bb72_perf.yaml"
DEFAULT_CONFIG_DIR = REPO_ROOT / "experiments/bb72-positive-delta/configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs"


class BenchmarkFailure(RuntimeError):
    """Raised when the benchmark cannot support the positive-delta claim."""


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-bb72-positive-delta")


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _round_row(round_idx: int, config_path: Path, metrics: RoundMetrics) -> dict[str, Any]:
    return {
        "round": round_idx,
        "config": config_path.name,
        "status": metrics.status,
        "ler_plain_classical": metrics.ler_plain_classical,
        "ler_predecoder": metrics.ler_predecoder,
        "delta_ler": metrics.delta_ler,
        "flops_per_syndrome": metrics.flops_per_syndrome,
        "n_params": metrics.n_params,
        "train_wallclock_s": metrics.train_wallclock_s,
        "eval_wallclock_s": metrics.eval_wallclock_s,
        "checkpoint_path": metrics.checkpoint_path,
    }


def _build_summary(run_id: str, env_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_delta = float("-inf")
    best_round = None
    enriched = []
    for row in rows:
        delta = row.get("delta_ler")
        if delta is None:
            raise BenchmarkFailure(f"round {row['round']} missing delta_ler")
        if float(delta) > best_delta:
            best_delta = float(delta)
            best_round = int(row["round"])
        enriched_row = dict(row)
        enriched_row["best_delta_ler_so_far"] = best_delta
        enriched.append(enriched_row)

    first_delta = float(enriched[0]["delta_ler"]) if enriched else 0.0
    return {
        "run_id": run_id,
        "env_name": env_name,
        "claim": "benchmark evidence; not a VERIFIED holdout claim",
        "has_positive_delta": best_delta > 0,
        "best_round": best_round,
        "best_delta_ler": best_delta,
        "improvement_vs_round_1": best_delta - first_delta,
        "rounds": enriched,
    }


def _write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# BB72 Positive Delta Benchmark Report",
        "",
        "This is benchmark evidence, not a VERIFIED holdout claim.",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- env_name: `{summary['env_name']}`",
        f"- has_positive_delta: `{summary['has_positive_delta']}`",
        f"- best_round: `{summary['best_round']}`",
        f"- best_delta_ler: `{summary['best_delta_ler']}`",
        f"- improvement_vs_round_1: `{summary['improvement_vs_round_1']}`",
        "",
        "| Round | Config | Status | delta_ler | best_so_far | LER plain | LER predecoder | Params |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rounds"]:
        lines.append(
            f"| {row['round']} | `{row['config']}` | {row['status']} | "
            f"{row['delta_ler']} | {row['best_delta_ler_so_far']} | "
            f"{row['ler_plain_classical']} | {row['ler_predecoder']} | {row['n_params']} |"
        )
    run_dir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(env_yaml: Path, config_dir: Path, output_root: Path) -> Path:
    env_yaml = Path(env_yaml)
    config_dir = Path(config_dir)
    output_root = Path(output_root)
    env = load_env_yaml(env_yaml)
    run_id = _run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    config_paths = sorted(config_dir.glob("round_*.yaml"))
    if not config_paths:
        raise BenchmarkFailure(f"no round configs found in {config_dir}")

    for round_idx, config_path in enumerate(config_paths, start=1):
        round_dir = run_dir / f"round_{round_idx}"
        cfg = RunnerConfig(
            env_name=env.name,
            predecoder_config=_load_config(config_path),
            training_profile="benchmark",
            seed=round_idx,
            round_dir=str(round_dir),
            env_yaml_path=str(env_yaml),
            invocation_argv=[sys.executable, __file__],
        )
        metrics = run_round(cfg, env)
        rows.append(_round_row(round_idx, config_path, metrics))

    summary = _build_summary(run_id=run_id, env_name=env.name, rows=rows)
    run_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(run_dir, summary)
    return run_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV_YAML)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = run_benchmark(
        env_yaml=args.env_yaml,
        config_dir=args.config_dir,
        output_root=args.output_root,
    )
    print(f"BB72_POSITIVE_DELTA_RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the driver success test and verify it passes**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_benchmark.py::test_run_benchmark_writes_positive_summary_and_report -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add experiments/bb72-positive-delta/run.py tests/test_bb72_positive_delta_benchmark.py
git commit -m "feat: add bb72 positive delta benchmark driver"
```

---

### Task 3: Enforce Positive-Delta Failure Gates

**Files:**
- Modify: `experiments/bb72-positive-delta/run.py`
- Modify: `tests/test_bb72_positive_delta_benchmark.py`

- [ ] **Step 1: Add failing gate tests**

Append these tests to `tests/test_bb72_positive_delta_benchmark.py`:

```python
def test_run_benchmark_fails_when_all_deltas_are_nonpositive(monkeypatch, tmp_path: Path) -> None:
    module = _load_benchmark_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "round_1.yaml").write_text(
        "type: gnn\ntraining: {learning_rate: 0.001, batch_size: 2, epochs: 1}\n",
        encoding="utf-8",
    )
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("name: fake\n", encoding="utf-8")

    monkeypatch.setattr(module, "load_env_yaml", lambda _path: type("Env", (), {"name": "bb72_perf"})())
    monkeypatch.setattr(
        module,
        "run_round",
        lambda _cfg, _env: RoundMetrics(
            status="ok",
            ler_plain_classical=0.1,
            ler_predecoder=0.1,
            delta_ler=0.0,
            flops_per_syndrome=1000,
            n_params=2000,
        ),
    )
    monkeypatch.setattr(module, "_run_id", lambda: "all-zero")

    try:
        module.run_benchmark(env_yaml=env_yaml, config_dir=config_dir, output_root=tmp_path / "runs")
    except module.BenchmarkFailure as exc:
        assert "no positive delta_ler" in str(exc)
    else:
        raise AssertionError("expected BenchmarkFailure")


def test_run_benchmark_fails_on_non_ok_round(monkeypatch, tmp_path: Path) -> None:
    module = _load_benchmark_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "round_1.yaml").write_text(
        "type: gnn\ntraining: {learning_rate: 0.001, batch_size: 2, epochs: 1}\n",
        encoding="utf-8",
    )
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("name: fake\n", encoding="utf-8")

    monkeypatch.setattr(module, "load_env_yaml", lambda _path: type("Env", (), {"name": "bb72_perf"})())
    monkeypatch.setattr(
        module,
        "run_round",
        lambda _cfg, _env: RoundMetrics(status="compile_error", status_reason="bad config"),
    )
    monkeypatch.setattr(module, "_run_id", lambda: "compile-error")

    try:
        module.run_benchmark(env_yaml=env_yaml, config_dir=config_dir, output_root=tmp_path / "runs")
    except module.BenchmarkFailure as exc:
        assert "round 1 failed with status compile_error" in str(exc)
    else:
        raise AssertionError("expected BenchmarkFailure")
```

- [ ] **Step 2: Run the gate tests and verify they fail**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_benchmark.py -v
```

Expected: FAIL because `run_benchmark()` writes a summary even when all deltas are nonpositive and does not reject non-`ok` rounds.

- [ ] **Step 3: Implement the gates**

Modify `experiments/bb72-positive-delta/run.py`:

```python
def _validate_summary(summary: dict[str, Any]) -> None:
    if not summary["rounds"]:
        raise BenchmarkFailure("benchmark produced no rounds")
    if not summary["has_positive_delta"]:
        raise BenchmarkFailure("no positive delta_ler found across benchmark rounds")
    if summary["improvement_vs_round_1"] <= 0:
        raise BenchmarkFailure("best round did not improve over round 1")
```

Inside `run_benchmark()`, immediately after `metrics = run_round(cfg, env)` add:

```python
        if metrics.status != "ok":
            raise BenchmarkFailure(
                f"round {round_idx} failed with status {metrics.status}: {metrics.status_reason}"
            )
```

Inside `run_benchmark()`, immediately after `summary = _build_summary(...)` add:

```python
    _validate_summary(summary)
```

Modify `main()` so CLI failures are visible and nonzero:

```python
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_dir = run_benchmark(
            env_yaml=args.env_yaml,
            config_dir=args.config_dir,
            output_root=args.output_root,
        )
    except BenchmarkFailure as exc:
        print(f"BB72_POSITIVE_DELTA_ERROR={exc}", file=sys.stderr)
        return 1
    print(f"BB72_POSITIVE_DELTA_RUN_DIR={run_dir}")
    return 0
```

- [ ] **Step 4: Run the benchmark driver tests and verify they pass**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_benchmark.py -v
```

Expected: PASS for all tests in `tests/test_bb72_positive_delta_benchmark.py`.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add experiments/bb72-positive-delta/run.py tests/test_bb72_positive_delta_benchmark.py
git commit -m "test: enforce bb72 positive delta benchmark gates"
```

---

### Task 4: Add Reviewer-Facing README

**Files:**
- Create: `experiments/bb72-positive-delta/README.md`
- Modify: `tests/test_bb72_positive_delta_assets.py`

- [ ] **Step 1: Add the failing README test**

Append these tests to `tests/test_bb72_positive_delta_assets.py`:

```python
def test_positive_delta_readme_states_scope_and_reproduction_command() -> None:
    readme = (REPO_ROOT / "experiments/bb72-positive-delta/README.md").read_text(encoding="utf-8")

    assert "BB72/OSD" in readme
    assert "benchmark evidence" in readme
    assert "not a VERIFIED holdout claim" in readme
    assert "python experiments/bb72-positive-delta/run.py" in readme
    assert "surface_d5 + mwpm + soft_priors" in readme
```

- [ ] **Step 2: Run the asset tests and verify they fail**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_assets.py -v
```

Expected: FAIL because `README.md` does not exist.

- [ ] **Step 3: Add the README**

Create `experiments/bb72-positive-delta/README.md`:

```markdown
# BB72 Positive Delta Benchmark

This track provides reviewer-facing benchmark evidence that AutoQEC rounds can
produce a positive `delta_ler` when the neural predecoder output affects the
classical backend.

It intentionally does not replace Demo 1. Demo 1 proves the live LLM loop can
run end to end on `surface_d5`. The current `surface_d5 + mwpm + soft_priors`
path cannot demonstrate a real positive `delta_ler`, because MWPM decodes the
raw syndrome and ignores `soft_priors`.

## Why BB72/OSD

`bb72_perf.yaml` uses the BB72 parity-check artifact and
`classical_backend: osd`. In this path, `soft_priors` are passed into OSD as
`channel_probs`, so a trained predecoder can influence decoding.

## Run

```bash
python experiments/bb72-positive-delta/run.py
```

The script writes:

```text
runs/YYYYMMDD-HHMMSS-bb72-positive-delta/
|- round_1/
|- round_2/
|- round_3/
|- summary.json
`- report.md
```

## Success Criteria

The script exits nonzero unless:

- every round finishes with `status == "ok"`;
- at least one round has `delta_ler > 0`;
- the best round improves over round 1.

The result is benchmark evidence, not a VERIFIED holdout claim. The separate
verification workflow remains authoritative for `VERIFIED` Pareto admission.
```

- [ ] **Step 4: Run the asset tests and verify they pass**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_assets.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add experiments/bb72-positive-delta/README.md tests/test_bb72_positive_delta_assets.py
git commit -m "docs: document bb72 positive delta benchmark"
```

---

### Task 5: Run Manual Benchmark And Add Snapshot Contract

**Files:**
- Create: `experiments/bb72-positive-delta/expected_output/summary.json`
- Modify: `tests/test_bb72_positive_delta_assets.py`

- [ ] **Step 1: Run the full benchmark manually**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/python experiments/bb72-positive-delta/run.py
```

Expected: exit code 0 and stdout containing:

```text
BB72_POSITIVE_DELTA_RUN_DIR=
```

- [ ] **Step 2: Inspect the generated summary**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/python -c 'import json, pathlib; run_dir=pathlib.Path(sorted(pathlib.Path("runs").glob("*bb72-positive-delta"))[-1]); s=json.loads((run_dir/"summary.json").read_text()); assert s["has_positive_delta"] is True; assert s["best_delta_ler"] > 0; assert s["improvement_vs_round_1"] > 0; print(json.dumps({"run_dir": str(run_dir), "best_round": s["best_round"], "best_delta_ler": s["best_delta_ler"], "improvement_vs_round_1": s["improvement_vs_round_1"]}, indent=2))'
```

Expected: printed JSON includes the newest `runs/*bb72-positive-delta` path, a positive `best_delta_ler`, and a positive `improvement_vs_round_1`.

- [ ] **Step 3: Copy the successful summary into expected output**

Run:

```bash
mkdir -p experiments/bb72-positive-delta/expected_output
cp "$(find runs -maxdepth 1 -type d -name '*bb72-positive-delta' | sort | tail -1)/summary.json" experiments/bb72-positive-delta/expected_output/summary.json
```

Expected: `experiments/bb72-positive-delta/expected_output/summary.json` is an exact copy of the successful benchmark summary.

- [ ] **Step 4: Add the snapshot schema test**

Append this test to `tests/test_bb72_positive_delta_assets.py`:

```python
def test_positive_delta_expected_summary_schema() -> None:
    import json

    summary_path = REPO_ROOT / "experiments/bb72-positive-delta/expected_output/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["claim"] == "benchmark evidence; not a VERIFIED holdout claim"
    assert summary["has_positive_delta"] is True
    assert summary["best_delta_ler"] > 0
    assert summary["improvement_vs_round_1"] > 0
    assert summary["best_round"] in {2, 3}
    assert len(summary["rounds"]) == 3
    for row in summary["rounds"]:
        assert row["status"] == "ok"
        assert "delta_ler" in row
        assert "best_delta_ler_so_far" in row
        assert "ler_plain_classical" in row
        assert "ler_predecoder" in row
```

- [ ] **Step 5: Run focused tests**

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_bb72_positive_delta_assets.py tests/test_bb72_positive_delta_benchmark.py -v
```

Expected: PASS.

- [ ] **Step 6: Run lint on changed paths**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/ruff check experiments tests
```

Expected: PASS.

- [ ] **Step 7: Run the default non-integration gate**

Run:

```bash
make test PYTEST=/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest
```

Expected: PASS for the default non-integration suite.

- [ ] **Step 8: Commit Task 5**

Run:

```bash
git add experiments/bb72-positive-delta/expected_output/summary.json tests/test_bb72_positive_delta_assets.py
git commit -m "test: add bb72 positive delta benchmark snapshot"
```

Expected: commit succeeds and contains only the expected-output summary plus its schema test.

---

### Task 6: Final Verification And PR Evidence

**Files:**
- Modify: `demos/demo-1-surface-d5/README.md`
- Modify: `experiments/bb72-positive-delta/README.md`

- [ ] **Step 1: Add cross-reference from Demo 1 to the performance benchmark**

In `demos/demo-1-surface-d5/README.md`, add this paragraph under the caveats section:

```markdown
For positive `delta_ler` performance evidence, use
`experiments/bb72-positive-delta/`. Demo 1 is the live-loop stability path;
the current `surface_d5 + mwpm + soft_priors` route does not use neural
soft-prior outputs inside MWPM.
```

- [ ] **Step 2: Run focused documentation tests**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest tests/test_surface_demo_contract.py tests/test_bb72_positive_delta_assets.py -v
```

Expected: PASS.

- [ ] **Step 3: Run full verification gates**

Run:

```bash
/home/tx/QuAIR/qec-ai-decoder/.venv/bin/ruff check autoqec cli tests scripts experiments
make test PYTEST=/home/tx/QuAIR/qec-ai-decoder/.venv/bin/pytest
```

Expected: both commands PASS.

- [ ] **Step 4: Commit Task 6**

Run:

```bash
git add demos/demo-1-surface-d5/README.md experiments/bb72-positive-delta/README.md
git commit -m "docs: link demo stability and performance evidence"
```

Expected: commit succeeds with documentation-only changes.
