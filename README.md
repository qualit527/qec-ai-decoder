<p align="center">
  <img src="docs/branding/autoqec-ai-logo.png" alt="AutoQEC AI logo" width="720">
</p>

# QEC AI-Enhanced Decoder — **AutoQEC**

[![CI](https://github.com/qualit527/qec-ai-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/qualit527/qec-ai-decoder/actions/workflows/ci.yml)
[![testcov](https://codecov.io/github/qualit527/qec-ai-decoder/graph/badge.svg)](https://codecov.io/github/qualit527/qec-ai-decoder)
[![docs API](https://img.shields.io/badge/docs-API-blue)](https://qualit527.github.io/qec-ai-decoder/)

AutoQEC is an LLM-agent-driven auto-research harness for discovering **neural predecoders** for quantum error-correcting codes. Given an environment triple `(code_spec, noise_model, constraints)`, the system runs 10–20 rounds of *hypothesis → DSL config → training → evaluation → analysis* and emits verified predecoder checkpoints on the accuracy–latency–parameters Pareto front.

- **Spec**: [`docs/superpowers/specs/2026-04-20-autoqec-design.md`](docs/superpowers/specs/2026-04-20-autoqec-design.md) (v2.2)
- **API documentation**: [`docs/api-documentation.md`](docs/api-documentation.md)
- **Master plan**: [`docs/superpowers/plans/2026-04-21-autoqec-master.md`](docs/superpowers/plans/2026-04-21-autoqec-master.md)
- **Per-owner plans**: [`docs/superpowers/plans/`](docs/superpowers/plans/)
- **Test plan**: [`docs/test-plan.md`](docs/test-plan.md)
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

Per-round isolation and the Pareto-as-branches model are specified in
[`spec §15`](docs/superpowers/specs/2026-04-20-autoqec-design.md#15-worktree-based-experiment-model).
Each round runs on its own `exp/<run_id>/<NN>-<slug>` git branch inside a
`.worktrees/` checkout, Pareto members are the complete non-dominated
set of VERIFIED branches, and compose rounds test `git merge parent-A
parent-B` as a first-class scientific probe. Startup reconciliation
(§15.10) keeps `history.jsonl` and the live branch set in sync.

## Team & ownership

| Owner | Model binding | Primary responsibility | Plan file |
|---|---|---|---|
| **Chen Jiahan (team leader / repo maintainer)** | Claude Code | Orchestration + `surface_d5` env bring-up + `/autoqec-run` + `/add-env` + Demo 1 | [`person-a-chen.md`](docs/superpowers/plans/2026-04-21-autoqec-person-a-chen.md) |
| **Xie Jingu** | GLM | `independent_eval` + `bb72` qLDPC benchmarking + 3 audit/triage skills + Demo 4 & 5 | [`person-b-xie.md`](docs/superpowers/plans/2026-04-21-autoqec-person-b-xie.md) |
| **Lin Tengxiang** | Codex | DSL + Runner + predecoder templates + Makefile + Demo 2 | [`person-c-lin.md`](docs/superpowers/plans/2026-04-21-autoqec-person-c-lin.md) |

Phase-0 contract file (once created): `docs/contracts/interfaces.md` — edits require 3-of-3 owner sign-off.

## Planned deliverables

### 6 Features (core capabilities of the harness)

| # | Feature | Owner | Status |
|---|---|---|---|
| **F1** | End-to-end research loop over any `(code, noise, constraints)` triple | Chen Jiahan | planned |
| **F2** | Tier-1 canonical DSL + Tier-2 `custom_fn` escape hatch with AST+smoke validation | Lin Tengxiang | planned |
| **F3** | Independent verification module with 3 fair-baseline guards (seed isolation, bootstrap CI, ablation sanity) | Xie Jingu | planned |
| **F4** | Multi-agent orchestration (Ideator / Coder / Analyst) with tool whitelisting + 3-layer memory + `machine_state` tool | Chen Jiahan | planned |
| **F5** | Pareto-front maintenance across (Δ_LER, FLOPs, n_params) with verify-admitted candidates | Xie Jingu | planned |
| **F6** | Worktree-based experiment model (branches-as-Pareto; compose rounds; startup reconciliation) | Full team | **implemented** |

### 5 Demos (each produces a reproducible artifact)

| # | Demo | Proves | Owner | Priority |
|---|---|---|---|---|
| **D1** | `surface_d5` full research run | End-to-end harness works | Chen Jiahan | **P0** |
| **D2** | `bb72` qLDPC research run | Genericity across codes / backends | Lin Tengxiang | P1 |
| **D3** | `/add-env` onboarding | Non-coder can add environments | Chen Jiahan | P2 |
| **D4** | Reward-hacking detection | Memorizer cheater gets `FAILED` verdict | Xie Jingu | **P0** |
| **D5** | Failure recovery | `/diagnose-failure` identifies broken-config root cause | Xie Jingu | P2 |

### 5 Skills (LLM-reasoning user surfaces, exposed as `/<name>`)

| # | Skill | Purpose | Owner |
|---|---|---|---|
| **S1** | `/autoqec-run` | Run the full research loop on an env YAML | Chen Jiahan |
| **S2** | `/add-env` | Interactively create a new env YAML | Chen Jiahan |
| **S3** | `/verify-decoder` | Audit a Pareto candidate against holdout seeds | Xie Jingu |
| **S4** | `/review-log` | Read an entire research notebook, flag stuck hypotheses / overfitting | Xie Jingu |
| **S5** | `/diagnose-failure` | Root-cause a broken or stalled round, recommend a fix | Xie Jingu |

## Demo 1 — `surface_d5_depol` end-to-end

A reproducible one-round demo of the full research loop: **AI hypothesis →
DSL config → real predecoder training → analyst report** on the d=5
rotated surface code under circuit-level depolarising noise (`p = 5e-3`).
Baseline reference: 1M-shot PyMatching gives `LER = 0.01394` (seed 42,
committed in `demos/demo-1-surface-d5/expected_output/baseline_benchmark.json`).

### Install

```bash
pip install -e '.[dev]'            # torch, stim, pymatching, pydantic, click, …
pytest tests/ -m "not integration" # unit suite: should be all green
make test-integration              # manual integration suite; see docs/test-plan.md
```

### Run — two paths

**Path A: LLM loop** (inside Claude Code, the full experience)

```
/autoqec-run autoqec/envs/builtin/surface_d5_depol.yaml --rounds 1 --profile dev
```

Claude Code follows [`.claude/skills/autoqec-run/SKILL.md`](.claude/skills/autoqec-run/SKILL.md):
the Ideator proposes a hypothesis, the Coder emits a schema-legal
`PredecoderDSL` config, the Runner trains + evaluates, the Analyst
writes a 3-sentence verdict. Every subagent response is validated
against the pydantic schemas in `autoqec/agents/schemas.py` before it
touches `history.jsonl`.

**Path B: no-LLM baseline** (pure CLI, no Claude Code needed)

```bash
python scripts/run_quick.py                       # cross-platform
# or, on macOS/Linux with bash:
bash demos/demo-1-surface-d5/run_quick.sh
```

Both call `python -m cli.autoqec run <env> --rounds 3 --profile dev --no-llm`.
Useful as a smoke test for the training path without an LLM in the loop.

### Where results land

The two run paths produce **different** run-root layouts. Both write the
same per-round contents under `round_<N>/`.

**Path A (`/autoqec-run` LLM loop)** — full orchestration-side bookkeeping:

```
runs/<YYYYMMDD-HHMMSS>/
├── log.md               # narrative, one line per round (Analyst verdict)
├── history.jsonl        # one enriched record per round (hypothesis + DSL +
│                        # RoundMetrics + verdict + summary_1line)
├── pareto.json          # top-5 candidates by −Δ LER (verdict="candidate" only)
└── round_<N>/           # see below
```

**Path B (`run_quick.sh` no-LLM)** — bare Runner output, no Analyst/Pareto:

```
runs/<YYYYMMDD-HHMMSS>/
├── history.jsonl        # one RoundMetrics record per round (no hypothesis,
│                        # no verdict — no LLM ran)
├── history.json         # aggregate of above
└── round_<N>/           # see below
```

**Both paths — per-round contents**:

```
round_<N>/
├── config.yaml          # the DSL config the Runner trained (Coder's, or
│                        # random template in Path B)
├── train.log            # per-step training loss
├── checkpoint.pt        # trained weights + dsl_config
└── metrics.json         # RoundMetrics (§2.2): status, Δ LER, FLOPs, n_params, …
```

### How to read the output

- **Δ LER** = `ler_plain_classical − ler_predecoder`. Positive means the
  predecoder beat plain PyMatching on that round's val shots. Dev profile
  uses 64 val shots so Δ LER CI is wide — expect `Δ ≈ 0` with occasional
  lucky rounds. Prod profile numbers are the publishable ones.
- **Candidate vs ignore**: Analyst flags `verdict = "candidate"` when Δ
  LER is positive within CI. `pareto.json` only admits candidates;
  ignored rounds still land in `history.jsonl` for debugging.
- **Compare to the baseline**: match `ler_predecoder` against the 1M-shot
  reference (0.01394). If your dev-profile `ler_plain_classical` sits
  around 0.015–0.020 that's the normal 64-shot noise.

### Sample output

Two reference runs are committed under
[`demos/demo-1-surface-d5/expected_output/`](demos/demo-1-surface-d5/expected_output/):

- `sample_run/` — 3-round `--no-llm` smoke (random templates, CPU torch).
- `llm_loop_round1/` — a single live `/autoqec-run` round: the Ideator
  proposed "tiny FFN seed"; the Coder mapped it onto a 1-layer GNN (FFN
  is not a valid `PredecoderDSL.type`); the Runner trained a
  68 865-parameter model in 11 s; the Analyst returned `verdict="ignore"`
  (Δ LER = 0 on 64 val shots, as expected for dev profile).

### Caveats

- **No holdout verification.** `verdict = "candidate"` is an Analyst
  judgement, not a VERIFIED holdout evaluation — the `/verify-decoder`
  skill (Xie Jingu's subtree) is not yet on main.
- **Dev profile is a smoke profile, not a publishable one.** It caps
  training at 256 shots × 1 epoch. For real Δ LER numbers use
  `--profile prod` + multiple seeds; expect 10–20 min per round on GPU.

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
