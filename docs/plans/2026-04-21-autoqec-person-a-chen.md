# AutoQEC — 陈嘉汉 Execution Plan (Claude Code owner)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up the `surface_d5_depol` environment end-to-end (Stim circuit → PyMatching baseline → 1M-shot benchmark), build the orchestration skeleton that dispatches Ideator/Coder/Analyst subagents, and ship `/autoqec-run` + `/add-env` skills plus Demo 1.

**Architecture:** Two vertical slices. (1) **QEC-core slice**: `autoqec/envs/` + `circuits/surface_d5.stim` + `autoqec/decoders/baselines/pymatching_wrap.py` — gives the whole team the canonical surface-code env. (2) **Delivery slice**: `autoqec/orchestration/` + `.claude/agents/autoqec-*.md` + `.claude/skills/autoqec-run/` + `.claude/skills/add-env/` — gives the user the research loop.

**Tech Stack:** Stim, sinter, PyMatching, Claude Code's `Agent` tool for inline subagent dispatch, pydantic for schema, pytest, click.

**Companion contracts (frozen Phase 0):** `docs/contracts/interfaces.md` §2.1 (EnvSpec), §2.5 (subagent messages), §2.6 (skill CLIs).

---

## Reading order

1. `docs/superpowers/specs/2026-04-20-autoqec-design.md` §4.1–§4.3, §5, §8, §10
2. `docs/superpowers/plans/2026-04-21-autoqec-master.md` (this plan's parent)
3. `knowledge/AUTORESEARCH_PATTERNS.md` — the `open-coscientist/framework.py` (~467 LOC → ~300 LOC) port recipe

---

## Logical phases (mapped to master phases)

- **Phase 0** — draft `EnvSpec` and subagent-message contracts (Tasks A0.x)
- **Phase 1** — scaffold surface_d5 env, PyMatching baseline, orchestrator skeleton, agent prompt files (Tasks A1.1–A1.6)
- **Phase 2** — integrate with C's Runner for first end-to-end round (Tasks A2.1–A2.2)
- **Phase 3** — `/autoqec-run`, `/add-env`, Demo 1, walkthrough (Tasks A3.1–A3.4)

---

## Files you own

| Path | Responsibility |
|---|---|
| `autoqec/envs/schema.py` | `EnvSpec` pydantic models (§2.1) |
| `autoqec/envs/loader.py` | Load + validate `*.yaml` → `EnvSpec` |
| `autoqec/envs/builtin/surface_d5_depol.yaml` | Canonical surface env |
| `circuits/surface_d5.stim` | Stim circuit (generated) |
| `autoqec/decoders/baselines/pymatching_wrap.py` | PyMatching baseline decoder |
| `autoqec/orchestration/loop.py` | Main research-loop driver |
| `autoqec/orchestration/memory.py` | L1 ↔ L2 ↔ L3 memory bridging |
| `autoqec/agents/dispatch.py` | Python ↔ Claude-Code `Agent` tool adapter |
| `.claude/agents/autoqec-ideator.md` | Ideator prompt |
| `.claude/agents/autoqec-coder.md` | Coder prompt |
| `.claude/agents/autoqec-analyst.md` | Analyst prompt |
| `.claude/skills/autoqec-run/SKILL.md` | `/autoqec-run` skill |
| `.claude/skills/add-env/SKILL.md` | `/add-env` skill |
| `demos/demo-1-surface-d5/` | Demo 1 artifacts |

---

## Task A0.1: Draft `EnvSpec` schema

**Files:**
- Create: `autoqec/envs/schema.py`

- [ ] **Step 1: Write schema from contract**

Copy the exact pydantic classes from `docs/contracts/interfaces.md` §2.1 into `autoqec/envs/schema.py` (identical to master §2.1). Import order: `from pydantic import BaseModel, Field`; `from typing import Literal, Optional`.

- [ ] **Step 2: Add `EnvSpec.from_yaml` helper**

Append:

```python
import yaml
from pathlib import Path

def load_env_yaml(path: str | Path) -> EnvSpec:
    with open(path) as f:
        data = yaml.safe_load(f)
    return EnvSpec(**data)
```

- [ ] **Step 3: Smoke test**

```bash
python -c "from autoqec.envs.schema import EnvSpec, load_env_yaml; print(EnvSpec.model_json_schema()['title'])"
```

Expected: prints `EnvSpec`.

- [ ] **Step 4: Commit**

```bash
git add autoqec/envs/schema.py
git commit -m "feat: add EnvSpec pydantic schema (Phase-0 contract)"
```

---

## Task A1.1: Generate surface_d5 Stim circuit + save `circuits/surface_d5.stim`

**Files:**
- Create: `circuits/surface_d5.stim` (generated output)
- Create: `scripts/generate_surface_circuit.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_surface_circuit.py
from pathlib import Path
import stim

def test_surface_d5_circuit_exists():
    p = Path("circuits/surface_d5.stim")
    assert p.exists(), "run scripts/generate_surface_circuit.py first"
    c = stim.Circuit.from_file(str(p))
    assert c.num_qubits > 0

def test_surface_d5_circuit_has_detectors():
    c = stim.Circuit.from_file("circuits/surface_d5.stim")
    assert c.num_detectors > 0
    assert c.num_observables >= 1
```

- [ ] **Step 2: Run — should fail (no circuit yet)**

```bash
pytest tests/test_surface_circuit.py -v
```

Expected: both tests FAIL with "no such file".

- [ ] **Step 3: Write the generator**

```python
# scripts/generate_surface_circuit.py
"""Generate a d=5 rotated surface code, circuit-level depolarizing noise, p=5e-3."""
import stim
from pathlib import Path

def generate(p: float = 5e-3, distance: int = 5, rounds: int = 5) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )

if __name__ == "__main__":
    out = Path("circuits/surface_d5.stim")
    out.parent.mkdir(parents=True, exist_ok=True)
    c = generate()
    c.to_file(str(out))
    print(f"Wrote {out} ({c.num_qubits} qubits, {c.num_detectors} detectors)")
```

- [ ] **Step 4: Run it + rerun tests**

```bash
python scripts/generate_surface_circuit.py
pytest tests/test_surface_circuit.py -v
```

Expected: generator prints qubit count; tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_surface_circuit.py circuits/surface_d5.stim tests/test_surface_circuit.py
git commit -m "feat: add surface_d5 stim circuit generator"
```

---

## Task A1.2: Write `surface_d5_depol.yaml`

**Files:**
- Create: `autoqec/envs/builtin/surface_d5_depol.yaml`
- Modify: `tests/test_surface_circuit.py` (add env-load test)

- [ ] **Step 1: Write the env YAML**

```yaml
# autoqec/envs/builtin/surface_d5_depol.yaml
name: surface_d5_depol
code:
  type: stim_circuit
  source: circuits/surface_d5.stim
noise:
  type: depolarizing
  p: [1.0e-3, 5.0e-3, 1.0e-2]
  seed_policy:
    train:   [1, 999]
    val:     [1000, 1999]
    holdout: [9000, 9999]
constraints:
  latency_flops_budget: 10000000
  param_budget: 200000
  target_ler: 1.0e-4
  target_p: 1.0e-3
baseline_decoders:
  - pymatching
classical_backend: mwpm
eval_protocol:
  min_shots_train:  1000000
  min_shots_val:    100000
  min_shots_verify: 200000
  bootstrap_ci: 0.95
  osd_orders_reported: [0]
  x_z_decoding: circuit
```

- [ ] **Step 2: Write the loader test**

Append to `tests/test_surface_circuit.py`:

```python
def test_surface_env_yaml_loads():
    from autoqec.envs.schema import load_env_yaml
    spec = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    assert spec.name == "surface_d5_depol"
    assert spec.classical_backend == "mwpm"
    assert spec.noise.p == [1e-3, 5e-3, 1e-2]
```

- [ ] **Step 3: Run**

```bash
pytest tests/test_surface_circuit.py::test_surface_env_yaml_loads -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add autoqec/envs/builtin/surface_d5_depol.yaml tests/test_surface_circuit.py
git commit -m "feat: add surface_d5_depol env YAML"
```

---

## Task A1.3: Wrap PyMatching baseline decoder

**Files:**
- Create: `autoqec/decoders/baselines/pymatching_wrap.py`
- Create: `tests/test_pymatching_baseline.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_pymatching_baseline.py
import stim
from pathlib import Path

def test_pymatching_on_surface_d5():
    from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline
    circuit = stim.Circuit.from_file("circuits/surface_d5.stim")
    dec = PymatchingBaseline.from_circuit(circuit)
    # generate a small sample
    sampler = circuit.compile_detector_sampler(seed=1)
    detections, observables = sampler.sample(shots=1000, separate_observables=True)
    predictions = dec.decode_batch(detections)
    assert predictions.shape == observables.shape
    # logical error rate should be well below 1 (MWPM works on d=5 at p=5e-3)
    ler = (predictions != observables).mean()
    assert 0.0 <= ler <= 0.5
```

- [ ] **Step 2: Run — should fail (class missing)**

```bash
pytest tests/test_pymatching_baseline.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Implement**

```python
# autoqec/decoders/baselines/pymatching_wrap.py
from __future__ import annotations
import numpy as np
import pymatching
import stim

class PymatchingBaseline:
    """Thin wrapper: Stim DEM → PyMatching MWPM decoder."""

    def __init__(self, matching: pymatching.Matching, n_observables: int):
        self.matching = matching
        self.n_observables = n_observables

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit) -> "PymatchingBaseline":
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        return cls(matching, circuit.num_observables)

    def decode_batch(self, detections: np.ndarray) -> np.ndarray:
        """detections: [n_shots, n_detectors] bool → predictions: [n_shots, n_obs] bool."""
        return self.matching.decode_batch(detections)
```

- [ ] **Step 4: Run test — should pass**

```bash
pytest tests/test_pymatching_baseline.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add autoqec/decoders/baselines/pymatching_wrap.py tests/test_pymatching_baseline.py
git commit -m "feat: PyMatching baseline wrapper for surface codes"
```

---

## Task A1.4: 1M-shot benchmark script (pins compute numbers)

**Files:**
- Create: `scripts/benchmark_surface_baseline.py`
- Create: `demos/demo-1-surface-d5/expected_output/baseline_benchmark.json` (committed for diff tests later)

- [ ] **Step 1: Write the benchmark**

```python
# scripts/benchmark_surface_baseline.py
"""Run PyMatching on 1M shots of surface_d5 at p=5e-3; print LER + wallclock."""
import json
import time
from pathlib import Path
import numpy as np
import stim

from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline

def main(n_shots: int = 1_000_000, seed: int = 42):
    circuit = stim.Circuit.from_file("circuits/surface_d5.stim")
    dec = PymatchingBaseline.from_circuit(circuit)

    sampler = circuit.compile_detector_sampler(seed=seed)
    t0 = time.time()
    detections, observables = sampler.sample(shots=n_shots, separate_observables=True)
    t_sample = time.time() - t0

    t0 = time.time()
    predictions = dec.decode_batch(detections)
    t_decode = time.time() - t0

    errors = (predictions != observables).any(axis=1).sum()
    ler = float(errors / n_shots)
    result = {
        "n_shots": n_shots,
        "ler": ler,
        "n_errors": int(errors),
        "t_sample_s": t_sample,
        "t_decode_s": t_decode,
        "detections_shape": list(detections.shape),
    }
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (takes ~1–3 min)**

```bash
python scripts/benchmark_surface_baseline.py | tee /tmp/surface_baseline.json
```

Expected: JSON with `ler` roughly 1e-3 – 5e-3 at p=5e-3.

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_surface_baseline.py
git commit -m "feat: 1M-shot PyMatching baseline benchmark"
```

---

## Task A1.5: Subagent prompt files

**Files:**
- Create: `.claude/agents/autoqec-ideator.md`
- Create: `.claude/agents/autoqec-coder.md`
- Create: `.claude/agents/autoqec-analyst.md`

- [ ] **Step 1: Write `autoqec-ideator.md`**

```markdown
---
name: autoqec-ideator
description: Research-round Ideator for AutoQEC. Proposes the next predecoder hypothesis based on env_spec, Pareto front, recent history, and a machine_state tool.
tools: Read, Grep, Glob
---

You are the Ideator in AutoQEC's multi-round research loop.

# Inputs (provided in your prompt)
- `env_spec`: pydantic dump of the EnvSpec
- `pareto_front`: list of up to 5 VERIFIED candidates with (delta_ler, flops, n_params)
- `last_5_hypotheses`: previous hypotheses + their outcomes (with killed_by_safety noted)
- `knowledge_excerpts`: short excerpt from knowledge/DECODER_ROADMAP.md §5 (building blocks)
- `machine_state_hint`: the result of calling machine_state(run_dir)

# Required first action
Call `machine_state(run_dir)` and read:
- `gpu.vram_free_gb`
- `history_timings.wall_clock_p95_s`
- `history_timings.params_vs_time` (scatter)
- `budget.total_wallclock_s_remaining`

Use these to estimate wall-clock for your candidate. Stay within `budget.total_wallclock_s_remaining`. There are NO hard architectural caps — you judge feasibility.

# Output format
Exactly one fenced JSON block:

```json
{
  "hypothesis": "<1 sentence what to try>",
  "expected_delta_ler": 5e-5,
  "expected_cost_s": 900,
  "rationale": "<why this over alternatives; reference prior rounds>",
  "dsl_hint": {"type": "gnn", "message_fn": "gated_mlp", "layers": 4}
}
```

# Hard rules
- Never re-propose a hypothesis already in `last_5_hypotheses` without explicit new motivation.
- If Pareto has plateaued (3 rounds without delta_ler improvement > 1e-5), switch predecoder family or output_mode.
- Respect the time budget: if expected_cost_s > budget.total_wallclock_s_remaining / 2, shrink the proposal.
```

- [ ] **Step 2: Write `autoqec-coder.md`**

```markdown
---
name: autoqec-coder
description: Emits Tier-1 (YAML) or Tier-2 (custom_fn) predecoder DSL configs from an Ideator hypothesis.
tools: Read, Write, Edit, Grep, Glob
---

You are the Coder in AutoQEC.

# Inputs
- `hypothesis` (from Ideator)
- `dsl_schema` (Appendix A of the spec)
- `tier2_validator_rules` (AST + smoke-test rules)
- `best_so_far`: top 3 VERIFIED configs from Pareto

# Behavior
1. Start with Tier 1. Fill every required DSL field.
2. If the hypothesis explicitly calls out a novel building block that Tier 1 cannot express, emit a Tier-2 `custom_fn` for the single relevant slot. Keep the rest of the config Tier 1.
3. Validate mentally: types, shapes, imports (torch + torch.nn.functional only).
4. You have NO Bash. You cannot run training.

# Output
Exactly one fenced JSON block:

```json
{
  "tier": "1",
  "dsl_config": {...},
  "rationale": "<why this shape>"
}
```
```

- [ ] **Step 3: Write `autoqec-analyst.md`**

```markdown
---
name: autoqec-analyst
description: Writes a 3-sentence round report, extracts verdict from metrics.json.
tools: Read, Grep
---

You are the Analyst in AutoQEC.

# Inputs
- Path to `round_N/metrics.json`
- Previous round's one-line summary
- Current Pareto front

# Behavior
- READ-ONLY. You modify nothing.
- Write a 3-sentence report: round outcome, key trade-off, delta from previous round.
- Classify the round as `candidate` (worth verification) or `ignore` (below baseline or crashed).

# Output
Exactly one fenced JSON block:

```json
{
  "summary_1line": "<one line summarizing this round>",
  "verdict": "candidate",
  "next_hypothesis_seed": "<suggestion for Ideator>"
}
```
```

- [ ] **Step 4: Commit**

```bash
git add .claude/agents/autoqec-ideator.md .claude/agents/autoqec-coder.md .claude/agents/autoqec-analyst.md
git commit -m "feat: subagent prompt files (ideator, coder, analyst)"
```

---

## Task A1.6: Orchestrator skeleton + memory module

**Files:**
- Create: `autoqec/orchestration/loop.py`
- Create: `autoqec/orchestration/memory.py`
- Create: `autoqec/agents/dispatch.py`
- Create: `tests/test_orchestration_stub.py`

- [ ] **Step 1: Write the memory module**

```python
# autoqec/orchestration/memory.py
"""Three-layer memory. L1 = disk, L2 = orchestrator context, L3 = per-subagent context."""
import json
from pathlib import Path
from typing import Any

class RunMemory:
    """L1/L2 bridge. L3 is assembled on the fly when dispatching."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = run_dir / "history.jsonl"
        self.log_path = run_dir / "log.md"
        self.pareto_path = run_dir / "pareto.json"
        if not self.pareto_path.exists():
            self.pareto_path.write_text("[]")

    # ─── L1 writes ───────────────────────────────────────
    def append_round(self, record: dict) -> None:
        with self.history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def append_log(self, md: str) -> None:
        with self.log_path.open("a") as f:
            f.write(md + "\n")

    def update_pareto(self, pareto: list[dict]) -> None:
        self.pareto_path.write_text(json.dumps(pareto, indent=2))

    # ─── L2 summaries (rebuild each round from L1) ───────
    def l2_snapshot(self, k_last: int = 3) -> dict:
        history = []
        if self.history_path.exists():
            with self.history_path.open() as f:
                history = [json.loads(line) for line in f if line.strip()]
        pareto = json.loads(self.pareto_path.read_text())
        return {
            "rounds_so_far": len(history),
            "pareto": pareto[:5],
            "last_rounds": history[-k_last:],
        }

    # ─── L3 (Ideator / Coder / Analyst) ──────────────────
    def l3_for_ideator(self, env_spec: dict, kb_excerpt: str, machine_state: dict) -> dict:
        snap = self.l2_snapshot()
        return {
            "env_spec": env_spec,
            "pareto_front": snap["pareto"],
            "last_5_hypotheses": [r.get("hypothesis") for r in snap["last_rounds"][-5:]],
            "knowledge_excerpts": kb_excerpt,
            "machine_state_hint": machine_state,
        }

    def l3_for_coder(self, hypothesis: dict, schema_md: str, best_so_far: list[dict]) -> dict:
        return {"hypothesis": hypothesis, "dsl_schema": schema_md, "best_so_far": best_so_far}

    def l3_for_analyst(self, round_dir: Path, prev_summary: str, pareto: list[dict]) -> dict:
        return {"metrics_path": str(round_dir / "metrics.json"),
                "previous_summary": prev_summary,
                "pareto_front": pareto}
```

- [ ] **Step 2: Write the dispatcher stub**

```python
# autoqec/agents/dispatch.py
"""Abstracts how subagents are invoked.

Inline mode (Claude Code chat): orchestrator calls Agent tool directly;
this module returns the *prompt strings* that should be fed to each agent.
Background mode (server) shells out via autoqec/llm/router.py.
"""
import json
from typing import Literal

Role = Literal["ideator", "coder", "analyst"]

def build_prompt(role: Role, context: dict) -> str:
    header = f"# {role.upper()} CONTEXT\n\n"
    body = json.dumps(context, indent=2, default=str)
    footer = "\n\nRespond with a single fenced ```json block per your agent spec."
    return header + body + footer

def parse_response(role: Role, text: str) -> dict:
    """Extract the first fenced ```json block from text."""
    import re
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON block in {role} response:\n{text[:500]}")
    return json.loads(m.group(1))
```

- [ ] **Step 3: Write the orchestrator loop**

```python
# autoqec/orchestration/loop.py
"""Research-loop driver. In inline mode the orchestrator is the main chat;
this function returns a generator of steps so the caller can inject LLM calls."""
from pathlib import Path
from typing import Iterable
from autoqec.envs.schema import EnvSpec
from autoqec.orchestration.memory import RunMemory
from autoqec.agents.dispatch import build_prompt

def run_round_plan(env_spec: EnvSpec, run_dir: Path, round_idx: int,
                   machine_state: dict, kb_excerpt: str,
                   dsl_schema_md: str) -> dict:
    """Returns the prompts for the 3 subagents + the config paths.
    Caller (inline Agent tool or background subprocess) fills in responses."""
    mem = RunMemory(run_dir)
    snap = mem.l2_snapshot()

    ideator_ctx = mem.l3_for_ideator(env_spec.model_dump(), kb_excerpt, machine_state)
    return {
        "round_idx": round_idx,
        "round_dir": str(run_dir / f"round_{round_idx}"),
        "ideator_prompt": build_prompt("ideator", ideator_ctx),
        # coder + analyst prompts assembled after each prior step completes
    }
```

- [ ] **Step 4: Write stub tests**

```python
# tests/test_orchestration_stub.py
from pathlib import Path

def test_run_memory_records_rounds(tmp_path):
    from autoqec.orchestration.memory import RunMemory
    mem = RunMemory(tmp_path / "run1")
    mem.append_round({"round": 1, "hypothesis": "test"})
    snap = mem.l2_snapshot()
    assert snap["rounds_so_far"] == 1

def test_dispatch_prompt_roundtrip():
    from autoqec.agents.dispatch import build_prompt, parse_response
    p = build_prompt("ideator", {"env_spec": {"name": "x"}})
    assert "IDEATOR" in p
    fake = 'some text\n```json\n{"hypothesis": "h", "expected_delta_ler": 1e-4, "expected_cost_s": 10, "rationale": "r"}\n```\ntail'
    parsed = parse_response("ideator", fake)
    assert parsed["hypothesis"] == "h"
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_orchestration_stub.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add autoqec/orchestration/ autoqec/agents/dispatch.py tests/test_orchestration_stub.py
git commit -m "feat: orchestration skeleton + 3-layer memory + dispatcher stub"
```

---

## Task A2.1: End-to-end handshake with C (stub-config round)

**Gate:** Do not start this task until C's Task C2.2 (Runner accepting `RunnerConfig`) is green on main.

**Files:**
- Create: `runs/handshake/env.yaml` (copy of surface_d5_depol)
- Create: `runs/handshake/stub_config.yaml` (Tier-1 minimal GNN)
- Create: `scripts/e2e_handshake.py`

- [ ] **Step 1: Write the handshake script**

```python
# scripts/e2e_handshake.py
"""Phase-2 handshake: invoke Runner directly with a hand-written Tier-1 config,
bypassing LLM subagents. Output should land in runs/handshake/round_0/."""
from pathlib import Path
import yaml

from autoqec.envs.schema import load_env_yaml
from autoqec.runner.schema import RunnerConfig
from autoqec.runner.runner import run_round   # from C

def main():
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn", "output_mode": "soft_priors",
            "gnn": {"layers": 2, "hidden_dim": 16, "message_fn": "mlp",
                    "aggregation": "sum", "normalization": "layer",
                    "residual": False, "edge_features": ["syndrome_bit"]},
            "head": "linear",
            "training": {"learning_rate": 1e-3, "batch_size": 64,
                         "epochs": 1, "loss": "bce", "profile": "dev"},
        },
        training_profile="dev",
        seed=0,
        round_dir=str(Path("runs/handshake/round_0").absolute()),
    )
    metrics = run_round(cfg, env)
    print(metrics.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run (with C present)**

```bash
python scripts/e2e_handshake.py
```

Expected: `metrics.status == "ok"`; `metrics.ler_plain_classical` finite; `metrics.delta_ler` numeric (can be negative or ~0 for a tiny untrained GNN).

- [ ] **Step 3: Commit**

```bash
git add scripts/e2e_handshake.py
git commit -m "feat: Phase-2 end-to-end handshake script"
```

---

## Task A2.2: Wire LLM subagents through inline `Agent` tool

**Files:**
- Modify: `autoqec/orchestration/loop.py`
- Create: `autoqec/tools/machine_state.py`
- Create: `scripts/run_single_round.py`

- [ ] **Step 1: Write `machine_state.py`**

```python
# autoqec/tools/machine_state.py
import json, time
from pathlib import Path
import torch

def machine_state(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    hist = []
    p = run_dir / "history.jsonl"
    if p.exists():
        with p.open() as f:
            hist = [json.loads(l) for l in f if l.strip()]
    timings = [r.get("train_wallclock_s", 0) + r.get("eval_wallclock_s", 0) for r in hist]
    killed = sum(1 for r in hist if r.get("status") == "killed_by_safety")
    gpu = {}
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "vram_free_gb": (torch.cuda.mem_get_info()[0]) / 1e9,
        }
    return {
        "gpu": gpu,
        "history_timings": {
            "rounds_so_far": len(hist),
            "wall_clock_mean_s": sum(timings)/max(len(timings), 1),
            "wall_clock_p95_s": sorted(timings)[int(0.95 * len(timings))] if timings else 0,
            "params_vs_time": [(r.get("n_params", 0), t) for r, t in zip(hist, timings)],
            "killed_by_safety_count": killed,
        },
        "budget": {"total_wallclock_s_spent": sum(timings),
                   "total_wallclock_s_remaining": None},  # caller fills
    }
```

- [ ] **Step 2: Write single-round driver**

```python
# scripts/run_single_round.py
"""Drive one full research round using claude -p subprocess or inline Agent tool.
In inline mode, this script returns prompts for the user to paste; in subprocess
mode it shells out. Keep it simple — a full loop is built later."""
import json, sys
from pathlib import Path
from autoqec.envs.schema import load_env_yaml
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.loop import run_round_plan
from autoqec.tools.machine_state import machine_state

def main(env_yaml: str, run_dir: str, round_idx: int):
    env = load_env_yaml(env_yaml)
    rd = Path(run_dir)
    ms = machine_state(rd)
    kb = (Path("knowledge/DECODER_ROADMAP.md").read_text()[:3000]
          if Path("knowledge/DECODER_ROADMAP.md").exists() else "")
    dsl_md = Path("docs/superpowers/specs/2026-04-20-autoqec-design.md").read_text()
    plan = run_round_plan(env, rd, round_idx, ms, kb, dsl_md)
    print(json.dumps(plan, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
```

- [ ] **Step 3: Commit**

```bash
git add autoqec/tools/machine_state.py scripts/run_single_round.py
git commit -m "feat: machine_state tool + single-round driver"
```

---

## Task A3.1: `/autoqec-run` skill

**Files:**
- Create: `.claude/skills/autoqec-run/SKILL.md`

- [ ] **Step 1: Write the skill**

```markdown
---
name: autoqec-run
description: Run the AutoQEC research loop on a given env YAML. Orchestrates ideator/coder/analyst subagents, invokes the Runner, writes history.jsonl and pareto.json. Use when the user asks "run AutoQEC on <env>" or similar.
---

# /autoqec-run

## When to use
- User says "run AutoQEC", "start a research loop", or provides an env YAML path.
- User asks for a decoder discovery run on a specific code × noise × constraints triple.

## Required inputs
- `env_yaml` (path to EnvSpec YAML, e.g. `autoqec/envs/builtin/surface_d5_depol.yaml`)
- `rounds` (integer, default 10)
- `profile` (`dev` or `prod`, default `dev`)

## Behaviour
1. Read and validate the env YAML via `autoqec.envs.schema.load_env_yaml`.
2. Create `runs/<timestamp>/` and initialize `log.md`, `history.jsonl`, `pareto.json`.
3. For each round 1..N:
   a. Dispatch `autoqec-ideator` subagent with L3 context (env + pareto + history + machine_state + knowledge excerpts).
   b. Dispatch `autoqec-coder` subagent with the Ideator's hypothesis + DSL schema.
   c. Invoke `python -m autoqec run-round` with the Coder's YAML; wait for `metrics.json`.
   d. Dispatch `autoqec-analyst` subagent with the metrics.
   e. If Analyst verdict is `candidate`, invoke `/verify-decoder` on the round dir.
   f. Append a 1-line summary to log.md; update pareto.json.
4. After loop, print final Pareto front and path to `runs/<id>/log.md`.

## Tool-use rules
- The orchestrator (you) may call `Agent`, `Bash`, `Read`, `Write`.
- Ideator subagent: Read, Grep, Glob (plus machine_state via Agent prompt).
- Coder subagent: Read, Write, Edit.
- Analyst subagent: Read, Grep.

## Failure handling
- If a round's `metrics.status == "killed_by_safety"`, still record it to history and continue.
- If 3 consecutive rounds fail, halt and output a diagnostic notice.

## Underlying CLI
The skill is a thin wrapper over `python -m cli.autoqec run <env> --rounds <N> --profile <p>`.
For cost control, prefer `--profile dev` for the first 3 rounds of any new env.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/autoqec-run/SKILL.md
git commit -m "feat: /autoqec-run skill"
```

---

## Task A3.2: `/add-env` skill

**Files:**
- Create: `.claude/skills/add-env/SKILL.md`
- Modify: `cli/autoqec.py` — implement `add-env` subcommand (currently a stub)

- [ ] **Step 1: Write the skill**

```markdown
---
name: add-env
description: Interactively create a new AutoQEC env YAML by asking the user 4–6 focused questions (code type, noise, constraints, baseline). Validates and writes to envs/<name>.yaml.
---

# /add-env

## When to use
- User says "add a new env", "create an environment for code X", or provides a Stim/DEM file.

## Dialog checklist (in order)
1. **Env name** (slug, letters/digits/underscores).
2. **Code source**: path to `.stim` file, or parity-check matrix (`.alist`/`.npz`), or known code family + distance.
   - If user gives a code family (`surface d=5`, `bb 72-12-6`), offer to generate the circuit via `scripts/generate_*_circuit.py`.
3. **Noise model**: `depolarizing` default; ask for `p` sweep (3 values typical).
4. **Classical backend**: `mwpm` for surface, `osd` for qLDPC (infer from code type + double-check).
5. **Baseline decoders**: pymatching for surface; bposd (+ relay_bp if available) for qLDPC.
6. **Constraints**: `target_ler`, `target_p`, `latency_flops_budget`, `param_budget` (offer defaults).

## Output
Write `envs/<name>.yaml` following `EnvSpec` schema.
Validate via `python -m cli.autoqec validate-env envs/<name>.yaml` (adds a check command).

## Failure handling
- If Stim circuit fails to parse: offer to regenerate or ask user to upload.
- If parity-check matrix has unexpected shape: echo shape + ask for confirmation.
```

- [ ] **Step 2: Implement `add-env` CLI logic**

Replace the stub in `cli/autoqec.py`:

```python
@main.command(name="add-env")
@click.option("--out", required=True)
@click.option("--name", prompt="Env name (slug)", type=str)
@click.option("--code-source", prompt="Code source path (.stim / .alist)", type=str)
@click.option("--noise-p", prompt="Noise p values, comma-separated", default="1e-3,5e-3,1e-2")
@click.option("--backend", prompt="Classical backend", type=click.Choice(["mwpm", "osd"]), default="mwpm")
def add_env(out, name, code_source, noise_p, backend):
    import yaml, re
    data = {
        "name": name,
        "code": {"type": "stim_circuit", "source": code_source},
        "noise": {"type": "depolarizing", "p": [float(x) for x in noise_p.split(",")]},
        "constraints": {},
        "baseline_decoders": ["pymatching" if backend == "mwpm" else "bposd"],
        "classical_backend": backend,
    }
    from autoqec.envs.schema import EnvSpec
    EnvSpec(**data)  # raises on invalid
    with open(out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    click.echo(f"Wrote {out}")
```

- [ ] **Step 3: Smoke test**

```bash
echo -e "my_surface\ncircuits/surface_d5.stim\n1e-3,5e-3\nmwpm" | python -m cli.autoqec add-env --out /tmp/e.yaml
cat /tmp/e.yaml
```

Expected: valid YAML with `classical_backend: mwpm`.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/add-env/SKILL.md cli/autoqec.py
git commit -m "feat: /add-env skill + CLI implementation"
```

---

## Task A3.3: Demo 1 — surface_d5 full research run

**Files:**
- Create: `demos/demo-1-surface-d5/README.md`
- Create: `demos/demo-1-surface-d5/run.sh`
- Create: `demos/demo-1-surface-d5/walkthrough.md`

- [ ] **Step 1: Write `run.sh`**

```bash
#!/usr/bin/env bash
# Demo 1: full research loop on surface_d5_depol, 10 rounds prod profile.
set -euo pipefail
ROUNDS=${ROUNDS:-10}
PROFILE=${PROFILE:-prod}
python -m cli.autoqec run \
  autoqec/envs/builtin/surface_d5_depol.yaml \
  --rounds "$ROUNDS" --profile "$PROFILE"
echo ""
echo "Pareto front:"
cat runs/$(ls -t runs | head -1)/pareto.json
```

- [ ] **Step 2: Write `README.md`**

```markdown
# Demo 1: surface_d5 research loop

## Goal
Show AutoQEC discovering a GNN predecoder for distance-5 surface code under depolarizing noise. Produces Pareto ≥3 VERIFIED candidates; at least one should match PyMatching LER within statistical tolerance.

## Run
```bash
bash demos/demo-1-surface-d5/run.sh
```

## Expected outputs
- `runs/<id>/log.md` — research narrative
- `runs/<id>/pareto.json` — 3+ VERIFIED candidates
- `runs/<id>/round_*/` — per-round configs, metrics, checkpoints

## Runtime
~3.3 hours on RTX 4090 in prod profile; ~15 min in dev profile.

## Acceptance criteria
- Pareto contains ≥3 entries with `verdict=VERIFIED`.
- At least one candidate has `delta_ler ≥ 0` (matches or beats PyMatching within CI).
- `log.md` has round-by-round narrative with non-trivial rationales.
```

- [ ] **Step 3: Write `walkthrough.md`**

```markdown
# Demo 1 walkthrough

## What happens in each round
1. Ideator reads current Pareto + knowledge excerpts → proposes "try GNN with message_fn=gated_mlp, 3 layers".
2. Coder emits Tier-1 DSL YAML.
3. Runner compiles, trains (3–20 min), evaluates on val seeds, writes metrics.json.
4. Analyst reads metrics, writes 3-sentence summary.
5. If candidate: `/verify-decoder` runs independent_eval on holdout; appends verdict.

## Where to look
- `log.md`: chronological story.
- `pareto.json`: final accepted candidates with (delta_ler, flops, n_params).
- `round_<N>/verification_report.md`: detailed audit per candidate.

## Known limitations
- This demo uses uniform priors for the PyMatching baseline (not circuit-level priors).
- Prod profile is 10 rounds; longer runs typically hit diminishing returns after ~20 rounds.
```

- [ ] **Step 4: Commit**

```bash
chmod +x demos/demo-1-surface-d5/run.sh
git add demos/demo-1-surface-d5/
git commit -m "feat: Demo 1 surface_d5 research loop"
```

---

## Task A3.4: Execute Demo 1 smoke run + fix integration issues

- [ ] **Step 1: Run dev-profile smoke test**

```bash
ROUNDS=3 PROFILE=dev bash demos/demo-1-surface-d5/run.sh
```

Expected: 3 rounds complete, `pareto.json` has ≥1 entry.

- [ ] **Step 2: Run prod (overnight if needed)**

```bash
bash demos/demo-1-surface-d5/run.sh
```

- [ ] **Step 3: Capture expected-output snapshot**

```bash
RUN_DIR=$(ls -t runs | head -1)
cp runs/$RUN_DIR/pareto.json demos/demo-1-surface-d5/expected_output/pareto_sample.json
cp runs/$RUN_DIR/log.md demos/demo-1-surface-d5/expected_output/log_sample.md
git add demos/demo-1-surface-d5/expected_output/
git commit -m "docs: snapshot Demo 1 expected output"
```

---

## Self-review checklist

- [x] **Spec coverage**: G1 (harness end-to-end) → A2.2 + A3.3; G2 surface env → A1.1–A1.4; G3 skills → A3.1–A3.2; G4 demos → A3.3; G6 backend routing → A1.5 agent files + A2.2 dispatch; G7 machine_state → A2.2.
- [x] **No placeholders**: every file has concrete code/content.
- [x] **Type consistency**: `EnvSpec` use is identical across A1.2, A2.1, A3.2. `RunnerConfig` matches C's contract.
- [x] **Handoff points**: A2.1 gated on C's Runner; A3.3 gated on C's Runner + B's `/verify-decoder`.
