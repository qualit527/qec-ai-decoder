# QEC AI-Enhanced Decoder — **AutoQEC**

[![CI](https://github.com/qualit527/qec-ai-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/qualit527/qec-ai-decoder/actions/workflows/ci.yml)

AutoQEC is an LLM-agent-driven auto-research harness for discovering **neural predecoders** for quantum error-correcting codes. Given an environment triple `(code_spec, noise_model, constraints)`, the system runs 10–20 rounds of *hypothesis → DSL config → training → evaluation → analysis* and emits verified predecoder checkpoints on the accuracy–latency–parameters Pareto front.

- **Spec**: [`docs/superpowers/specs/2026-04-20-autoqec-design.md`](docs/superpowers/specs/2026-04-20-autoqec-design.md) (v2.2)
- **Master plan**: [`docs/superpowers/plans/2026-04-21-autoqec-master.md`](docs/superpowers/plans/2026-04-21-autoqec-master.md)
- **Per-owner plans**: [`docs/superpowers/plans/`](docs/superpowers/plans/)
- **Knowledge base**: `knowledge/` — 81-paper index + 3 synthesis documents (roadmap, strategic assessment, autoresearch patterns)

## Architecture at a glance

```
syndrome + code DEM
        │
        ↓
┌─────────────────────────────────────────┐
│ AI Predecoder (agent searches here)     │
│   type:        gnn | neural_bp          │
│   output_mode: hard_flip | soft_priors  │
└─────────────────────────────────────────┘
        │
        ↓
┌─────────────────────────────────────────┐
│ Classical Backend (fixed per env)       │
│   surface codes → MWPM (PyMatching)     │
│   qLDPC         → OSD                   │
└─────────────────────────────────────────┘
        │
        ↓
 logical correction
```

The classical backend guarantees structural validity; the predecoder contributes `Δ_LER = LER(plain_classical) − LER(predecoder + classical)` — a single clean number per round.

## Team & ownership

| Owner | Model binding | Primary responsibility | Plan file |
|---|---|---|---|
| **陈嘉汉 (team leader / repo maintainer)** | Claude Code | Orchestration + `surface_d5` env bring-up + `/autoqec-run` + `/add-env` + Demo 1 | [`person-a-chen.md`](docs/superpowers/plans/2026-04-21-autoqec-person-a-chen.md) |
| **谢金谷** | GLM | `independent_eval` + `bb72` qLDPC benchmarking + 3 audit/triage skills + Demo 4 & 5 | [`person-b-xie.md`](docs/superpowers/plans/2026-04-21-autoqec-person-b-xie.md) |
| **林腾祥** | Codex | DSL + Runner + predecoder templates + Makefile + Demo 2 | [`person-c-lin.md`](docs/superpowers/plans/2026-04-21-autoqec-person-c-lin.md) |

Phase-0 contract file (once created): `docs/contracts/interfaces.md` — edits require 3-of-3 owner sign-off.

## Planned deliverables

### 5 Features (core capabilities of the harness)

| # | Feature | Owner | Status |
|---|---|---|---|
| **F1** | End-to-end research loop over any `(code, noise, constraints)` triple | 陈嘉汉 | planned |
| **F2** | Tier-1 canonical DSL + Tier-2 `custom_fn` escape hatch with AST+smoke validation | 林腾祥 | planned |
| **F3** | Independent verification module with 3 fair-baseline guards (seed isolation, bootstrap CI, ablation sanity) | 谢金谷 | planned |
| **F4** | Multi-agent orchestration (Ideator / Coder / Analyst) with tool whitelisting + 3-layer memory + `machine_state` tool | 陈嘉汉 | planned |
| **F5** | Pareto-front maintenance across (Δ_LER, FLOPs, n_params) with verify-admitted candidates | 谢金谷 | planned |

### 5 Demos (each produces a reproducible artifact)

| # | Demo | Proves | Owner | Priority |
|---|---|---|---|---|
| **D1** | `surface_d5` full research run | End-to-end harness works | 陈嘉汉 | **P0** |
| **D2** | `bb72` qLDPC research run | Genericity across codes / backends | 林腾祥 | P1 |
| **D3** | `/add-env` onboarding | Non-coder can add environments | 陈嘉汉 | P2 |
| **D4** | Reward-hacking detection | Memorizer cheater gets `FAILED` verdict | 谢金谷 | **P0** |
| **D5** | Failure recovery | `/diagnose-failure` identifies broken-config root cause | 谢金谷 | P2 |

### 5 Skills (LLM-reasoning user surfaces, exposed as `/<name>`)

| # | Skill | Purpose | Owner |
|---|---|---|---|
| **S1** | `/autoqec-run` | Run the full research loop on an env YAML | 陈嘉汉 |
| **S2** | `/add-env` | Interactively create a new env YAML | 陈嘉汉 |
| **S3** | `/verify-decoder` | Audit a Pareto candidate against holdout seeds | 谢金谷 |
| **S4** | `/review-log` | Read an entire research notebook, flag stuck hypotheses / overfitting | 谢金谷 |
| **S5** | `/diagnose-failure` | Root-cause a broken or stalled round, recommend a fix | 谢金谷 |

## Day-1 first PR — substantive vertical slices

Every member submits **one meaningful PR** via their bound agent (Claude Code / GLM / Codex) on Day 1. Team leader **陈嘉汉** verifies and merges. Each PR delivers a usable system increment (not just a schema), so by end of Day 1 the team has **baseline + verifier + GNN factory** — the three pillars.

| Owner | Task | Deliverables | PR size | Plan reference |
|---|---|---|---|---|
| **陈嘉汉** (leader) | **Scaffold + `surface_d5` baseline end-to-end**. Everything needed to say "here's the classical LER we beat": package tree, `pyproject.toml`, CLI stubs, Stim circuit generator, env YAML, PyMatching baseline wrapper, tests. | `autoqec/` package dirs, `pyproject.toml`, `cli/autoqec.py`, `autoqec/envs/schema.py` (EnvSpec), `scripts/generate_surface_circuit.py`, `circuits/surface_d5.stim`, `autoqec/envs/builtin/surface_d5_depol.yaml`, `autoqec/decoders/baselines/pymatching_wrap.py`, `tests/test_surface_circuit.py`, `tests/test_pymatching_baseline.py` | ~400 LOC | [master M0.3](docs/superpowers/plans/2026-04-21-autoqec-master.md#task-m03-create-skeleton-autoqec-package) + [A1.1–A1.3](docs/superpowers/plans/2026-04-21-autoqec-person-a-chen.md#task-a11-generate-surface_d5-stim-circuit--save-circuitssurface_d5stim) |
| **谢金谷** | **`independent_eval` with 3 fair-baseline guards + reward-hacking probe**. The publishability gate: seed isolation, bootstrap 95% CI, ablation sanity. Plus a hand-crafted Memorizer cheater whose verdict test confirms the guards fire. | `autoqec/eval/schema.py` (VerifyReport), `autoqec/eval/bootstrap.py`, `autoqec/eval/independent_eval.py`, `autoqec/cheaters/memorize.py`, `tests/test_bootstrap.py`, `tests/test_independent_eval.py`, `tests/test_isolation_rule.py` (CI lint blocking `from autoqec.runner` imports), `tests/test_reward_hacking.py` | ~300 LOC | [person-b B0.1–B1.3, B1.6](docs/superpowers/plans/2026-04-21-autoqec-person-b-xie.md#task-b01-draft-verifyreport-schema) |
| **林腾祥** | **DSL (schema + compiler) + BipartiteGNN module + 3 GNN seed templates**. The neural-predecoder core: feed YAML → get trainable `nn.Module`. Completes the `compile_predecoder(cfg, n_var, n_check)` path end-to-end. | `autoqec/runner/schema.py` (RunnerConfig + RoundMetrics), `autoqec/decoders/modules/base.py` (PredecoderBase), `autoqec/decoders/dsl_schema.py`, `autoqec/decoders/modules/gnn.py` (BipartiteGNN), `autoqec/decoders/dsl_compiler.py`, `autoqec/example_db/gnn_{small,medium,gated}.yaml`, `tests/test_dsl_schema.py`, `tests/test_gnn_module.py`, `tests/test_dsl_compiler.py`, `tests/test_seed_templates.py` | ~500 LOC | [person-c C0.1–C0.2, C1.1–C1.2, C1.4, C1.6 (GNN only)](docs/superpowers/plans/2026-04-21-autoqec-person-c-lin.md#task-c01-draft-runnerconfig--roundmetrics) |

**Acceptance for each PR** (leader merges when all satisfied):
1. Pydantic contract fields match `docs/superpowers/plans/2026-04-21-autoqec-master.md` §2 verbatim.
2. `pytest tests/ -m "not integration" -v` passes on the PR branch.
3. New LOC carries working tests, not placeholders.
4. Commit message follows `<type>: <description>` format.

**Merge order** (because 谢金谷 and 林腾祥 import from 陈嘉汉's `EnvSpec`):
1. **陈嘉汉's PR merges first** (scaffold + `EnvSpec` + surface baseline).
2. **谢金谷 + 林腾祥 rebase onto main, open PRs in parallel.** Their work doesn't cross — merge in either order.

**Why these tasks and not bigger/smaller**:
- Smaller (schema-only): leaves Day 1 afternoon empty; three tiny PRs that don't stress the agent-driven workflow.
- Bigger (full Day-1 plan): risks one person blocking the merge queue; leader has to review ~1500 LOC in one sitting.
- **These three deliverables** each exercise one vertical of the system (baseline / verifier / GNN factory) and unblock Phase-2 integration without becoming a review bottleneck.

Defered to Day-1 afternoon or Day 2: Neural-BP module (C1.3), 1M-shot benchmark (A1.4), bb72 env (B1.4–B1.5), orchestration skeleton (A1.5–A1.6), Runner + RunnerSafety (C1.7).

## Quick start (after Phase-0 contracts merge)

```bash
# Install
pip install -e '.[dev]'

# Unit tests
pytest tests/ -m "not integration" -v

# Run a no-LLM smoke loop (random seed templates, 3 rounds dev profile)
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml \
  --rounds 3 --profile dev --no-llm

# Verify a round
python -m cli.autoqec verify runs/<id>/round_1 \
  --env autoqec/envs/builtin/surface_d5_depol.yaml
```

## Logical phases (from master plan)

| Phase | Gate | Typical duration |
|---|---|---|
| **Phase 0** — Contract freeze | `docs/contracts/interfaces.md` committed; 6 contracts signed off by all 3 owners | ~0.5 day |
| **Phase 1** — Parallel scaffolds | Each owner's unit tests green in isolation | ~1 day |
| **Phase 2** — End-to-end integration | One full research round completes: config → Runner → metrics → verify → report | ~0.5–1 day |
| **Phase 3** — Demos, skills, polish | Demo 1 + Demo 4 ship VERIFIED; `/autoqec-run` + `/verify-decoder` callable | ~1 day |

## Repo layout (planned)

```
qec-ai-decoder/
├── autoqec/                        # Python package
│   ├── envs/                       # EnvSpec + builtin envs
│   ├── runner/                     # Non-LLM train + eval + safety + FLOPs
│   ├── eval/                       # independent_eval (ISOLATED from runner)
│   ├── decoders/                   # DSL schema, compiler, GNN/Neural-BP modules
│   ├── orchestration/              # Research-loop driver + 3-layer memory
│   ├── agents/                     # Subagent dispatcher
│   ├── llm/                        # claude-cli / codex-cli router
│   ├── tools/                      # machine_state, etc.
│   ├── pareto/                     # Front maintenance
│   └── example_db/                 # Tier-1 seed templates
├── cli/autoqec.py                  # click CLI entry
├── .claude/
│   ├── agents/                     # Subagent prompt files
│   └── skills/                     # 5 user-facing skills
├── demos/                          # 5 demos with run.sh + walkthrough.md
├── envs/                           # User-authored env YAMLs
├── circuits/                       # *.stim files
├── runs/                           # .gitignore'd; per-run outputs
├── knowledge/                      # Literature, roadmap, strategic docs
├── docs/superpowers/specs/         # Design specs
├── docs/superpowers/plans/         # Implementation plans
├── docs/contracts/                 # Phase-0 interface contracts
├── tests/
├── Makefile
└── pyproject.toml
```

## License

TBD — decision scheduled for Phase-3 (public release prep).

## Novelty positioning

> AutoQEC is the first LLM-agent-driven discovery engine for quantum error-correction decoders. Given any `(code, noise, constraint)` triple, it systematically produces Pareto fronts of reproducibility-verified neural predecoders — turning hand-craft decoder design into a scalable, auditable research workflow.

Target venue: **Quantum** (primary) · IEEE QCE (secondary). See `knowledge/STRATEGIC_ASSESSMENT.md` for the full defense.
