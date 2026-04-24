<p align="center">
  <img src="docs/branding/autoqec-ai-logo.png" alt="AutoQEC AI logo" width="720">
</p>

# QEC AI-Enhanced Decoder — **AutoQEC**

[![CI](https://github.com/qualit527/qec-ai-decoder/actions/workflows/ci.yml/badge.svg)](https://github.com/qualit527/qec-ai-decoder/actions/workflows/ci.yml)
[![testcov](https://codecov.io/github/qualit527/qec-ai-decoder/graph/badge.svg)](https://codecov.io/github/qualit527/qec-ai-decoder)
[![docs API](https://img.shields.io/badge/docs-API-blue)](https://qualit527.github.io/qec-ai-decoder/)

AutoQEC is an LLM-agent-driven auto-research harness for discovering **neural predecoders** for quantum error-correcting codes. Given an environment triple `(code_spec, noise_model, constraints)`, the system runs 10–20 rounds of *hypothesis → DSL config → training → evaluation → analysis* and emits verified predecoder checkpoints on the accuracy–latency–parameters Pareto front.

---

## ⚡ Review in one prompt

Clone the repo, open a Claude Code / Codex CLI in the project root, and paste the block below verbatim. The agent will install deps, run the four headline demos, and hand back a pass/fail table. Total wall-clock on CPU-only hardware: **~10–15 min** (skip step 0's unit suite if you're in a hurry — ~3 min off).

````text
Run the 4 demos below in order and report results. All commands run from the
repo root. First create and activate .venv (Windows: `.venv\Scripts\activate`,
Unix: `source .venv/bin/activate`), then `pip install -e '.[dev]'`. Do NOT
enter the live-LLM path (nested CLI sessions are unstable). For each step
report: cmd, wall-clock, key artifact paths, pass/fail against the stated
criteria. At the end give one markdown summary table.

0. Preflight (optional but strongly recommended):
   - `ruff check autoqec cli tests scripts` must print "All checks passed"
   - `pytest tests/ -m "not integration" -q` must be all green (~4 min, 290 tests)
   - `pytest tests/test_reward_hacking_traps.py -m integration --run-integration -v`
     must be 3/3 green (trap_A/B/C verifier guards)

1. Demo 1 — surface-code end-to-end loop (~2.5 min):
   `bash demos/demo-1-surface-d5/run_quick.sh`
   Pass criteria: in the newest `runs/<ts>/`, all of `history.jsonl`,
   `candidate_pareto.json`, `fork_graph.json`,
   `round_1/{config.yaml,train.log,checkpoint.pt,metrics.json}` exist;
   `metrics.json.status == "ok"`.

2. Demo 2 — qLDPC cross-code-family (~3 min, routes through OSD not MWPM):
   `MODE=fast bash demos/demo-2-bb72/run.sh`
   Pass criteria: in the new `runs/<ts>/round_1/metrics.json`, `status == "ok"`;
   the env config uses `classical_backend: osd` (prove the backend actually
   switched away from the surface-code MWPM path).

3. Demo 4 — reward-hacking rejection (~1 min, memorizer must be rejected):
   `bash demos/demo-4-reward-hacking/run.sh`
   Pass criteria: `runs/demo-4/round_0/verification_report.json.verdict` is in
   {SUSPICIOUS, FAILED} — never VERIFIED; script exits 0.

4. Demo 5 — failure root-cause diagnosis (~5 s):
   `bash demos/demo-5-failure-recovery/run.sh`
   Pass criteria: stdout JSON contains `"status": "compile_error"` and a
   `status_reason` field.

Summary table columns: demo | cmd | wall-clock | key artifact | pass/fail.
Rules:
- If any step fails, keep going to the next step; record the fail in the
  table rather than aborting.
- On failure, paste the last 30 lines of that step's stdout/stderr — do not
  silently swallow errors.
- Do not modify source files; only read, run, and report.
- If every demo passes, end with the line "harness end-to-end healthy"; if any
  failed, end with "needs triage: <which step>".
````

**What the four demos prove, in plain language:**

| Demo | Answer it gives the reviewer |
|---|---|
| 1 — `surface_d5` no-LLM smoke | The full round pipeline (DSL → train → eval → Pareto → `fork_graph.json`) actually writes every artifact, with `Δ_LER` reported on the surface-code path. |
| 2 — `bb72` qLDPC | The same harness swaps MWPM → OSD cleanly; `(code, noise, constraints)` is really the only input knob. |
| 4 — Reward-hacking detection | A hand-built memorizer cheater gets **rejected** (`SUSPICIOUS` or `FAILED`), never admitted to Pareto — the independent verifier actually guards the front. |
| 5 — Failure recovery | `cli.autoqec diagnose` takes a broken round dir and emits a machine-readable root cause (`compile_error` + reason), feeding the `/diagnose-failure` skill. |

> The live-LLM research loop (`/autoqec-run`) is intentionally out of the one-prompt flow — nesting Claude-Code-inside-Claude-Code is unstable. See [`docs/verification/human-verification-report-2026-04-24.md`](docs/verification/human-verification-report-2026-04-24.md) for the last end-to-end retest and [`.claude/skills/autoqec-run/SKILL.md`](.claude/skills/autoqec-run/SKILL.md) for the manual path.

---

- **Spec**: [`docs/superpowers/specs/2026-04-20-autoqec-design.md`](docs/superpowers/specs/2026-04-20-autoqec-design.md) (v2.2)
- **API documentation**: [`docs/api-documentation.md`](docs/api-documentation.md)
- **Master plan**: [`docs/superpowers/plans/2026-04-21-autoqec-master.md`](docs/superpowers/plans/2026-04-21-autoqec-master.md)
- **Per-owner plans**: [`docs/superpowers/plans/`](docs/superpowers/plans/)
- **Test plan**: [`docs/verification/human-verification-test-plan.md`](docs/verification/human-verification-test-plan.md)
- **Developer test targets**: [`docs/test-plan.md`](docs/test-plan.md) — `make lint`, `make test`, and `make test-integration` gate the full unit + GPU/integration suites (the one-prompt flow above only exercises `make test` + traps).
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

## Deliverables

### 6 Features (core capabilities of the harness)

All six are present on `main`; the one-prompt flow above exercises F1–F6 end-to-end.

| # | Feature | Status | Evidence |
|---|---|---|---|
| **F1** | End-to-end research loop over any `(code, noise, constraints)` triple | **implemented** | `cli/autoqec.py run` + `autoqec/orchestration/llm_loop.py`; Demo 1 / Demo 2 drive the no-LLM path |
| **F2** | Tier-1 canonical DSL + Tier-2 `custom_fn` escape hatch with AST+smoke validation | **implemented** | `autoqec/decoders/{dsl_schema,dsl_compiler,custom_fn_validator,custom_fn_rules}.py`; 18 hostile cases in `tests/test_custom_fn_validator.py` |
| **F3** | Independent verification module with 3 fair-baseline guards (seed isolation, paired bootstrap CI, ablation sanity) | **implemented** | `autoqec/eval/independent_eval.py` + `eval/bootstrap.py`; trap_A/B/C guards proven on main (Demo 4 + `tests/test_reward_hacking_traps.py`) |
| **F4** | Multi-agent orchestration (Ideator / Coder / Analyst) with tool whitelisting + 3-layer memory + `machine_state` tool | **implemented** | `autoqec/agents/dispatch.py` + `.claude/agents/autoqec-{ideator,coder,analyst}.md`; `autoqec/orchestration/memory.py`; `autoqec/tools/machine_state.py`; `autoqec/runner/safety.py` |
| **F5** | Pareto-front maintenance across (Δ_LER, FLOPs, n_params) with verify-admitted candidates | **implemented** | `autoqec/pareto/front.py` + `orchestration/round_recorder.py`; atomic `pareto.json` swap covered by `tests/test_pareto_atomic*.py` |
| **F6** | Worktree-based experiment model (branches-as-Pareto; compose rounds; startup reconciliation; `fork_graph.json`) | **implemented** | `autoqec/orchestration/{worktree,subprocess_runner,reconcile,fork_graph}.py`; persistence proven by `tests/test_fork_graph_persist.py` |

### 5 Demos (each produces a reproducible artifact)

Four ship as runnable demos. The `/add-env` demo (D3) stays planned — the CLI subcommand exists (`python -m cli.autoqec add-env …`) but no packaged demo directory yet.

| # | Demo | Proves | Priority | Status |
|---|---|---|---|---|
| **D1** | `surface_d5` full research run — `demos/demo-1-surface-d5/run_quick.sh` | End-to-end harness works | **P0** | **implemented** |
| **D2** | `bb72` qLDPC research run — `demos/demo-2-bb72/run.sh` (fast/dev/prod modes) | Genericity across codes / classical backends (MWPM → OSD) | P1 | **implemented** |
| **D3** | `/add-env` onboarding | Non-coder can add environments | P2 | planned (CLI ready, demo dir pending) |
| **D4** | Reward-hacking detection — `demos/demo-4-reward-hacking/run.sh` | Memorizer cheater gets `SUSPICIOUS` / `FAILED` verdict | **P0** | **implemented** |
| **D5** | Failure recovery — `demos/demo-5-failure-recovery/run.sh` | `cli.autoqec diagnose` identifies `compile_error` root cause | P2 | **implemented** |

### Skills (LLM-reasoning user surfaces, exposed as `/<name>`)

All five skills under `.claude/skills/` are discoverable from Claude Code. `/add-env` remained a CLI-only subcommand; `/read-zulip` was added to recover off-repo hackathon context.

| Skill | Purpose | Status |
|---|---|---|
| `/autoqec-run` | Run the full research loop on an env YAML | **implemented** |
| `/verify-decoder` | Audit a Pareto candidate against holdout seeds (wraps `cli.autoqec verify`) | **implemented** |
| `/review-log` | Read an entire `runs/<id>/log.md`, flag stuck hypotheses / overfitting | **implemented** |
| `/diagnose-failure` | Root-cause a broken or stalled round, recommend a fix (wraps `cli.autoqec diagnose`) | **implemented** |
| `/read-zulip` | Pull Zulip stream/topic history for off-repo project context | **implemented** |
| `/add-env` | Interactively create a new env YAML | planned (CLI only for now: `python -m cli.autoqec add-env`) |

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

MIT — see [LICENSE](LICENSE). Copyright © 2026 AutoQEC Contributors.
