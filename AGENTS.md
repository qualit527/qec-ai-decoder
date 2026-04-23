# AGENTS.md

Developer guidance for AI coding agents (Claude Code, Codex, etc.) when working on this repository.
For project overview, team structure, and demo walkthroughs, see `README.md`.

## Build & Test

```bash
pip install -e '.[dev]'                     # install with dev deps
python -m pytest tests/ -m "not integration" -v   # unit tests (CI uses this)
python -m pytest tests/ -m "integration" -v --run-integration  # integration tests (manual entry)
python -m pytest tests/test_bootstrap.py -v        # single test file
python -m pytest tests/test_bootstrap.py::test_bootstrap_ci_basic -v  # single test
ruff check autoqec cli tests scripts              # lint (CI uses this)
ruff check --fix autoqec cli tests scripts        # lint with auto-fix
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml --rounds 1 --profile dev --no-llm  # smoke test
```

## Make Targets

```makefile
make install        # pip install -e '.[dev]'
make test           # pytest tests/ -m "not integration" -v
make test-integration # pytest tests/ -m "integration" -v --run-integration
make lint           # ruff check autoqec cli tests scripts
make run            # full research loop with LLM agents (needs API keys)
make run-nollm      # N-round random-template smoke loop (no LLM needed)
make demo-2         # run bb72 qLDPC demo
make run-all-claude # run with Claude models for all agents
make run-cheap      # run with cheaper haiku model for ideator
```

Integration execution guidance lives in `docs/test-plan.md`. The default
repo gate stays `make lint` + `make test`; use `make test-integration`
only when you explicitly want the end-to-end layer.

## Architecture

### Decoder Pipeline

```
syndrome + code DEM (stim circuit or parity-check matrix)
        │
        ↓
┌─────────────────────────────────────────┐
│ AI Predecoder (agent searches here)     │
│   type: gnn | neural_bp                 │
│   output_mode: hard_flip | soft_priors  │
│   defined via PredecoderDSL YAML config │
└─────────────────────────────────────────┘
        │
        ↓
┌─────────────────────────────────────────┐
│ Classical Backend (fixed per env)       │
│   surface codes → MWPM (PyMatching)     │
│   qLDPC codes   → BP+OSD (ldpc)        │
└─────────────────────────────────────────┘
        │
        ↓
 logical correction
```

### Research Loop

`/autoqec-run` or `make run` drives: **Ideator → Coder → Runner → Analyst → record** per round. The orchestrator dispatches subagents via `autoqec.agents.dispatch` and records results via `autoqec.orchestration.memory.RunMemory`.

### CLI Commands

| Command | Purpose |
|---------|---------|
| `run-round` | Execute a single training+eval round from a DSL config YAML |
| `run` | Multi-round loop (LLM mode or `--no-llm` random templates) |
| `verify` | Independent holdout verification of a checkpoint |
| `add-env` | Interactive EnvSpec YAML composer |
| `review-log` | Summarize a run's history.jsonl (rounds, kills, Pareto count) |
| `diagnose` | Inspect a round_dir for config/metrics/log presence |

### Packages

| Package | Purpose |
|---------|---------|
| `autoqec.envs` | `EnvSpec` + `load_env_yaml` — quantum code + noise + constraints + eval protocol. Built-in envs in `envs/builtin/*.yaml`. |
| `autoqec.envs.schema` | Pydantic models: `EnvSpec`, `CodeSpec`, `NoiseSpec`, `SeedPolicy`, `ConstraintsSpec`, `EvalProtocol` |
| `autoqec.decoders` | DSL schema, compiler, neural modules, and classical backend adapters |
| `autoqec.decoders.dsl_schema` | `PredecoderDSL` — strict pydantic model for YAML configs (`type`, `output_mode`, `gnn`/`neural_bp` blocks, `training`) |
| `autoqec.decoders.dsl_compiler` | `compile_predecoder(config, n_var, n_check)` → `BipartiteGNN` or `NeuralBP` module |
| `autoqec.decoders.modules` | `PredecoderBase` (interface), `gnn.py` (BipartiteGNN), `neural_bp.py` (NeuralBP), `mlp.py` |
| `autoqec.decoders.backend_adapter` | `decode_with_predecoder()` — routes predecoder output through MWPM or OSD |
| `autoqec.decoders.baselines` | `PymatchingBaseline` (MWPM), `BpOsdBaseline` (BP+OSD) — classical-only baselines |
| `autoqec.decoders.custom_fn_rules` | AST + smoke validation for Tier-2 `custom_fn` escape hatch |
| `autoqec.runner` | Training + evaluation engine. `run_round()` compiles, trains, evaluates, and saves checkpoint. |
| `autoqec.runner.data` | `load_code_artifacts()`, `sample_syndromes()` — Stim or parity-check data loading |
| `autoqec.runner.safety` | `RunnerSafety` — wall-clock cutoff, VRAM pre-check, NaN rate guard, forbidden import list |
| `autoqec.runner.flops` | FLOPs estimation via `fvcore.nn.FlopCountAnalysis` with graceful fallback |
| `autoqec.eval` | Independent verification module — **ISOLATED from runner** (must not import `autoqec.runner`) |
| `autoqec.eval.independent_eval` | `independent_verify()` — 3-guard holdout verifier (seed isolation, bootstrap CI, ablation sanity) |
| `autoqec.eval.bootstrap` | `bootstrap_ci_mean()` — non-parametric bootstrap for LER confidence intervals |
| `autoqec.eval.schema` | `VerifyReport` — verdict (VERIFIED/SUSPICIOUS/FAILED), holdout LER, CI, ablation result |
| `autoqec.cheaters` | `MemorizerPredecoder` — hand-crafted cheating decoder for reward-hacking detection tests |
| `autoqec.pareto` | Pareto front maintenance across (Δ_LER, FLOPs, n_params) |
| `autoqec.orchestration` | Multi-agent loop driver and 3-layer memory system |
| `autoqec.orchestration.memory` | `RunMemory` — L1 (disk: history.jsonl, log.md, pareto.json), L2 (summary snapshot), L3 (per-subagent context) |
| `autoqec.orchestration.loop` | `run_round_plan()`, `build_coder_prompt()`, `build_analyst_prompt()` — prompt assembly |
| `autoqec.agents` | Subagent prompt builder, response parser, and pydantic schemas |
| `autoqec.agents.dispatch` | `build_prompt(role, ctx)` / `parse_response(role, text)` — shared wire format for inline and background modes |
| `autoqec.agents.schemas` | `IdeatorResponse`, `CoderResponse`, `AnalystResponse` — strict response validation |
| `autoqec.tools` | `machine_state()` — GPU probe + timing history for the Ideator's L3 context |

### Key Design Decisions

- **Eval isolation**: `autoqec/eval/independent_eval.py` must NOT import from `autoqec.runner.*`. Data-loading helpers are copy-pasted as private functions (`_CodeArtifacts`, `_load_code_artifacts`, `_sample_holdout`). This ensures the verifier is decoupled from the training loop. A test in `tests/test_isolation_rule.py` enforces this with a regex check.
- **Seed policy**: Every `EnvSpec` defines disjoint `train`/`val`/`holdout` seed ranges (default: train 1-999, val 1000-1999, holdout 9000-9999). Holdout seeds are never seen during training. The verifier checks for seed leakage.
- **Two-tier DSL**: Tier-1 uses structured YAML fields (`gnn`, `neural_bp` blocks with validated options). Tier-2 allows a `custom_fn` Python code block with AST validation + smoke test.
- **Runner checkpoint format**: `{"class_name", "state_dict", "output_mode", "dsl_config"}` — the verifier rebuilds the model via `compile_predecoder(dsl_config)` then loads `state_dict`.
- **3-guard verification**: (1) seed-leakage check ensures holdout seeds don't overlap train/val, (2) bootstrap CI ensures Δ_LER is statistically significant, (3) ablation sanity shuffles model params and confirms performance degrades.
- **Memorizer ablation**: `MemorizerPredecoder` stores knowledge in `self.table` (not weights). The verifier detects this type and calls `ablate()` to clear the table instead of shuffling params.
- **Safety guardrails**: `RunnerSafety` enforces wall-clock cutoff (900s), VRAM pre-check, max NaN rate (1%), and forbidden imports. A round can return `status="killed_by_safety"`.
- **Profile modes**: `dev` profile caps shots (256 train / 64 val) and epochs (1) for fast iteration. `prod` uses higher caps (2048 / 256, 3 epochs).
- **Deterministic sampling**: Seeds are deterministically selected from seed ranges via `_select_seeds()`. Same seed range + n_shots always produces the same data.
- **Classical backend selection**: Determined by `env_spec.classical_backend` — `"mwpm"` for surface codes (PyMatching), `"osd"` for qLDPC codes (BP+OSD from `ldpc` package).

## Environments

Environments are defined as YAML files in `autoqec/envs/builtin/` with this structure:

```yaml
name: surface_d5_depol
code:
  type: stim_circuit          # or parity_check_matrix
  source: circuits/surface_d5.stim
noise:
  type: depolarizing
  p: [1e-3, 5e-3, 1e-2]
  seed_policy:
    train: [1, 999]
    val: [1000, 1999]
    holdout: [9000, 9999]
constraints:
  latency_flops_budget: 10000000
  param_budget: 200000
  target_ler: 1e-4
  target_p: 1e-3
baseline_decoders: [pymatching]
classical_backend: mwpm        # or osd
eval_protocol:
  min_shots_train: 1000000
  min_shots_val: 100000
  min_shots_verify: 200000
  bootstrap_ci: 0.95
```

Built-in environments:
- `surface_d5_depol.yaml` — d=5 rotated surface code, depolarizing noise (MWPM backend)
- `bb72_depol.yaml` — [[72,12,6]] bivariate bicycle qLDPC code (OSD backend)

Add new environments with: `python -m cli.autoqec add-env --out my_env.yaml`

## Predecoder DSL

Predecoder configs are YAML files validated against `PredecoderDSL` (see `autoqec/decoders/dsl_schema.py`).

```yaml
type: gnn
output_mode: soft_priors
gnn:
  layers: 3
  hidden_dim: 64
  message_fn: mlp
  aggregation: sum
  normalization: layer
  residual: true
training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 10
  loss: bce
  profile: dev
```

Example configs live in `autoqec/example_db/` (e.g., `gnn_small.yaml`, `gnn_gated.yaml`, `neural_bp_min.yaml`).

## Skills

| Skill | Purpose |
|-------|---------|
| `/autoqec-run` | Run the full Ideator → Coder → Runner → Analyst research loop. Recipe in `.claude/skills/autoqec-run/SKILL.md`. |
| `/verify-decoder` | Independent holdout verification of a Pareto candidate checkpoint. `.claude/skills/verify-decoder/SKILL.md`. |
| `/review-log` | Read a run's history.jsonl, summarize rounds, flag stuck hypotheses. `.claude/skills/review-log/SKILL.md`. |
| `/diagnose-failure` | Root-cause a broken or stalled round, recommend a fix. `.claude/skills/diagnose-failure/SKILL.md`. |

## Subagents

Three specialized subagents drive the research loop. Their specs are in `.claude/agents/`:

| Agent | Output schema | Purpose |
|-------|--------------|---------|
| `autoqec-ideator` | `IdeatorResponse` | Proposes next hypothesis given env, Pareto front, and history |
| `autoqec-coder` | `CoderResponse` | Translates hypothesis into Tier-1 YAML or Tier-2 custom_fn DSL config |
| `autoqec-analyst` | `AnalystResponse` | Analyzes round metrics, classifies as `candidate` or `ignore`, seeds next hypothesis |

All subagent responses are validated against pydantic schemas in `autoqec.agents.schemas`. Responses must contain exactly one fenced ` ```json ` block.

## File Layout

```
autoqec/                          # Main package
  agents/                         # Subagent dispatch, schemas
  cheaters/                       # MemorizerPredecoder (reward-hacking test)
  decoders/                       # DSL schema, compiler, modules, baselines
    baselines/                    # PymatchingBaseline, BpOsdBaseline
    modules/                      # BipartiteGNN, NeuralBP, PredecoderBase
  envs/                           # EnvSpec + built-in env YAMLs
    builtin/                      # surface_d5_depol.yaml, bb72_depol.yaml
  eval/                           # Independent verification (ISOLATED from runner)
  example_db/                     # Predecoder template YAMLs
  orchestration/                  # Loop driver + 3-layer memory
  pareto/                         # Pareto front maintenance
  runner/                         # Training + eval engine + safety
  tools/                          # machine_state probe
cli/autoqec.py                    # CLI entry point
circuits/                         # .stim and .npy code files
demos/                            # Reproducible demo scenarios
  demo-1-surface-d5/              # Surface code end-to-end
  demo-2-bb72/                    # qLDPC benchmarking
  demo-4-reward-hacking/          # Memorizer detection demo
  demo-5-failure-recovery/        # Diagnose-failure demo
docs/
  contracts/interfaces.md         # Phase-0 interface contracts (3-owner sign-off)
  plans/                          # Per-owner implementation plans
  specs/                          # Per-owner task briefs
  reward_hacking_taxonomy.md      # Reward-hacking attack taxonomy
knowledge/                        # Research knowledge base (bibliography, synthesis docs)
scripts/                          # Utility scripts (benchmark, scout, run_single_round)
tests/                            # Test suite (see markers below)
```

## Test Markers

- `@pytest.mark.integration` — end-to-end tests excluded from CI (`-m "not integration"`)
- No marker — unit tests, run in CI

Key test files by ownership area:
- Verification: `test_bootstrap.py`, `test_independent_eval.py`, `test_verify_integration.py`, `test_isolation_rule.py`
- Reward hacking: `test_reward_hacking.py`
- Baselines: `test_bposd_baseline.py`, `test_pymatching_baseline.py`
- Pareto: `test_pareto.py`
- DSL: `test_dsl_compiler.py`, `test_dsl_schema.py`, `test_custom_fn_validator.py`
- Runner: `test_runner_safety.py`, `test_runner_smoke.py`
- Modules: `test_gnn_module.py`, `test_neural_bp_module.py`

## Commit Convention

Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR:
1. Python 3.12 setup
2. `pip install -e '.[dev]'`
3. `make lint` → `ruff check autoqec cli tests scripts`
4. `make test` → `pytest tests/ -m "not integration" -v`

## Dependencies

- `stim` — quantum circuit simulation and detector sampling
- `pymatching` — Minimum Weight Perfect Matching decoder (surface codes)
- `ldpc` — Belief Propagation + OSD decoder (qLDPC codes)
- `torch` — neural network training and inference
- `pydantic` — schema validation for EnvSpec, RoundMetrics, VerifyReport, subagent responses
- `click` — CLI framework
- `fvcore` — FLOPs estimation
- `numpy` — numerical operations
- `PyYAML` — YAML config loading

## Team & Ownership

| Owner | Model binding | Primary responsibility |
|-------|--------------|----------------------|
| Chen Jiahan (team leader) | Claude Code | Orchestration + surface_d5 env + `/autoqec-run` + `/add-env` + Demo 1 |
| Xie Jingu | GLM | `independent_eval` + bb72 qLDPC + 3 audit skills + Demo 4 & 5 |
| Lin Tengxiang | Codex | DSL + Runner + predecoder templates + Makefile + Demo 2 |

Phase-0 contracts: `docs/contracts/interfaces.md` — edits require 3-of-3 owner sign-off.
