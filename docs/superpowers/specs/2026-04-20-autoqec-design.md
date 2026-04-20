# AutoQEC — Design Specification

**Date**: 2026-04-20
**Status**: Draft, pending user review
**Authors**: Team (陈嘉汉、谢金谷、林腾祥) + brainstorming with Claude
**Project**: QEC AI-enhanced decoder (AutoQEC)
**Repo**: `qec-ai-decoder`

---

## 1. Overview

AutoQEC is a generic LLM-agent-driven auto-research harness for discovering neural decoders for quantum error-correcting codes. Given any tuple

```
(code_spec, noise_model, constraints)
```

the system autonomously runs multi-round research loops: *hypothesis → code → experiment → analysis → iterate*. It outputs a set of checkpoints on the accuracy–latency–parameter Pareto front plus a reproducible research notebook in markdown.

The harness is built as a clean abstraction (user replaces the `(code, noise, constraints)` triple and runs), with two reference environments shipped as MVP evidence (surface code + a small qLDPC), and a stretch environment (non-Pauli noise) planned for post-MVP. It is the first AI-Scientist-class automated research loop targeting the QEC decoder design space — an open niche confirmed by literature survey.

## 2. Goals & Non-Goals

### Goals

- **G1.** Generic `(code, noise, constraints)` → Pareto-front decoder checkpoints, without per-env harness code changes.
- **G2.** A multi-agent DAG (Ideator → Coder → Runner → Analyst → Reviewer) that runs multiple "research rounds" autonomously, producing a readable research notebook.
- **G3.** Two reference environments for MVP evidence: surface code (canonical baseline) and BB qLDPC (where GNN/Neural-BP has room to improve over BP+OSD).
- **G4.** Reproducible Pareto placement via independent decoder verification on held-out seeds.
- **G5.** Dual LLM backend (Claude CLI, Codex CLI) selectable per-agent so Max-plan users run without API keys; cost ablation is a single Makefile switch.
- **G6.** Inline mode (runs directly in a Claude Code chat via subagents) and background mode (detached Python + LangGraph) share the same prompt definitions and runner code.
- **G7.** Contributor entry path for QEC domain experts who are not Python developers (via `/add-env` skill), mirroring the problem-reductions community model.

### Non-Goals (MVP)

- **N1.** Full Pareto coverage across all code families. MVP ships two envs; user can add more via `/add-env`.
- **N2.** Automatic paper generation (Sakana v1/v2's writer stack). We produce a structured markdown lab notebook and leave paper drafting to humans.
- **N3.** FPGA / ASIC deployment. Latency is measured as FLOPs (floating-point operations, hardware-agnostic Pareto axis via `fvcore`/`ptflops`) plus wall-clock on the training host (reference only). See §4.8.
- **N4.** Real-chip small-sample fine-tuning. Listed in the original PDF but out of MVP scope — synthetic data only for the first release.
- **N5.** Web dashboard. Markdown notebook + PNG Pareto plots only.

## 3. User-Facing API

### 3.1 Environment YAML (primary entrypoint)

```yaml
# envs/bb72_depol.yaml
name: bb72_depol
code:
  type: stim_circuit          # or: parity_check_matrix | tanner_graph
  source: circuits/bb72.stim
noise:
  type: depolarizing          # or: biased | leakage | custom_dem
  p: [0.001, 0.005, 0.01]     # multi-p sweep
  seed_policy:
    train: 1..999
    val:   1000..1999
    holdout: 9000..9999       # never seen by Coder/Ideator/Runner
constraints:
  latency_flops: 1e7
  param_budget: 1e6
  target_ler: 1e-4
baseline_decoders: [pymatching, bposd]
```

The three groups (`code`, `noise`, `constraints`) are **decoupled** — users can freely recombine.

### 3.2 CLI

```bash
autoqec run envs/bb72_depol.yaml --rounds 50 --budget-usd 20
autoqec resume runs/2026-04-21-bb72/
autoqec pareto runs/*/ --output pareto.html
autoqec verify runs/2026-04-21-bb72/round_42/   # independent verification
autoqec topology-check runs/*/                   # gap analysis across runs
```

### 3.3 Claude Code Skill (Inline)

```
/autoqec-run envs/bb72_depol.yaml --rounds 5
```

Runs in the current chat as an orchestrator that dispatches subagents (see §4.3).

### 3.4 Background / long run

```
/autoqec-run envs/bb72_depol.yaml --rounds 50 --background
```

Spawns detached Python process using `nohup`; writes to `runs/<name>-<date>/`. User can `tail` the log or resume the chat later and read `runs/*/log.md`.

## 4. Architecture

### 4.1 Four layers

```
Layer 4 — Entrypoints
  ├─ .claude/skills/autoqec-run/SKILL.md      (inline orchestrator)
  ├─ cli/autoqec.py                            (CLI, backend subprocess launcher)
  └─ Makefile targets                          (run-forever, run-cheap, run-with-codex)

Layer 3 — Subagent definitions  (SINGLE SOURCE OF TRUTH)
  .claude/agents/
    ├─ autoqec-ideator.md    (tools: Read, Grep, Glob)
    ├─ autoqec-coder.md      (tools: Read, Write, Edit, Grep, Glob — NO Bash)
    ├─ autoqec-analyst.md    (tools: Read, Grep)
    └─ autoqec-reviewer.md   (tools: Read, Grep, Glob)

Layer 2 — Python core
  autoqec/
    ├─ orchestration/     LangGraph DAG (background mode only)
    ├─ agents/            Thin Python wrappers that invoke subagents via Agent SDK or subprocess
    ├─ runner/            Non-LLM: train, eval, checkpoint, FLOPs
    ├─ eval/              LER, confidence intervals
    │   └─ independent_eval.py   (ISOLATED — must not import runner/)
    ├─ decoders/          DSL schema + PyTorch templates + baseline wrappers
    ├─ envs/              EnvSpec loader, CodeSpec / NoiseSpec / Constraints
    ├─ example_db/        Canonical builders (envs, decoders) — shared by CLI/tests/docs
    ├─ schemas/           Reference material cited by subagents: dsl_schema.yaml,
    │                     env_schema.yaml, decoder_template_index.md
    ├─ llm/               backend router (claude-cli / codex-cli; api post-MVP)
    ├─ logging/           lab-notebook markdown generator
    └─ pareto/            front maintenance, topology analysis

Layer 1 — External dependencies
  stim, sinter, pymatching, ldpc, stimbposd, qLDPC, torch, langgraph, click
```

### 4.2 Subagent definitions (Layer 3)

Each `.claude/agents/<name>.md` file is the **single source of truth** for that role's behavior. Both modes consume the same file:

- **Inline mode**: main orchestrator uses the `Agent` tool with `subagent_type="autoqec-ideator"` — Claude Code runtime loads the `.md` file.
- **Background mode**: Python reads the `.md` file and passes its content via `claude -p --append-system-prompt` or `codex exec -c system=<content>`.

Tool whitelisting is **architectural reward-hacking defense**:
- Coder has **no Bash** → physically cannot execute training or eval scripts; can only produce code artifacts.
- Ideator has no Write → cannot pollute the working tree with fake reports.
- Analyst is read-only → cannot edit runner code or checkpoints.

### 4.3 Agent DAG & state

State type (shared between inline and background):

```python
class AutoQECState(TypedDict):
    env_spec: EnvSpec
    history: list[RoundResult]            # accumulated
    pareto_front: list[Checkpoint]        # only verified entries
    current_round: int
    current_hypothesis: Optional[Hypothesis]
    retry_count: int
    costs: CostLedger                     # per-agent token / USD tracking
```

Flow (one round):

```
Ideator (subagent) ── proposes hypothesis ──┐
                                            ↓
                                       Coder (subagent)
                                            │ produces DSL or PyTorch
                                            ↓
                   (LINEAR RETRY for MVP;   Runner (pure Python)
                    AIDE solution tree post-MVP)   trains + evals
                                            ↓
                                       Analyst (subagent)
                                            │ writes round report
                                            ↓
                               score ≥ Pareto candidate threshold?
                                   ↙                          ↘
                              yes                              no
                                ↓                               ↓
                          verify-decoder                 append log,
                         (independent eval)              next round
                           ↙        ↘
                      VERIFIED   FAILED/SUSPICIOUS
                          ↓            ↓
                   admit to Pareto   flag + human review
                                            ↓
                          (every K rounds) Reviewer (subagent)
                             meta-reflection on trajectory
```

### 4.4 Inline vs Background mode

| Aspect | Inline | Background |
|---|---|---|
| Who runs the DAG? | Main Claude Code chat (orchestrator) dispatches subagents via `Agent` tool | Python process runs LangGraph DAG |
| Where does LLM work happen? | Subagent context windows (isolated from main chat) | `claude -p` or `codex exec` subprocesses |
| Typical round count | 5–50 (subagents keep main context small) | 50–500 |
| Survives user closing client? | No | Yes (`nohup` detach) |
| Checkpointing | File system only (manual resume) | LangGraph SQLite checkpointer (auto resume) |

Both share:
- `.claude/agents/*.md` (subagent prompts)
- `autoqec/agents/prompts/` (any include fragments)
- `autoqec/runner/`, `autoqec/eval/`, `autoqec/envs/`, `autoqec/decoders/`

### 4.5 Runner (non-LLM)

`autoqec/runner/train_and_eval.py` is a deterministic Python script:

1. Load env_spec + decoder artifact (DSL or PyTorch)
2. Compile decoder (DSL → `nn.Module` via `autoqec/decoders/dsl_compiler.py`)
3. Stim + sinter generate syndrome batches using **train seeds only**
4. Train `N` epochs with early stopping on **val seeds**
5. Final eval on **val seeds**; emit metrics JSON
6. Save checkpoint to `runs/<run_id>/round_<k>/`

Runner **never** calls an LLM. Coder has no Bash, so the only path to invoke Runner is from the orchestrator (inline) or LangGraph node (background).

### 4.6 Independent `verify-decoder`

Isolated module `autoqec/eval/independent_eval.py`:

- Does not import `autoqec.runner.*` (CI-enforced)
- Loads checkpoint directly via PyTorch + env_spec via YAML
- Re-runs ≥5000 shots on **holdout seeds** (Coder/Ideator/Runner never saw these)
- Independent LER computation, independent FLOPs counter
- Ablation sanity checks:
  - Random-shuffle weights → LER should collapse to baseline
  - All-zero syndrome → output should be "logical OK"
  - Single-bit syndrome flip → output should change (unless bit is a known stabilizer symmetry)
- Statistical test: compare independent LER against Runner-reported val LER
  - |Δ| < 1σ → `VERIFIED`
  - 1–3σ → `SUSPICIOUS` (overfit suspected)
  - > 3σ → `FAILED` (reward hacking suspected)
- Output `verification_report.md` (ephemeral; Analyst reads to decide Pareto admit)

Also exposed as a standalone skill `/verify-decoder` for manual audit.

### 4.7 Backend router (`autoqec/llm/router.py`)

```python
def call_llm(messages, model, backend) -> str:
    if backend == "claude-cli":
        return _claude_cli(messages, model)
    elif backend == "codex-cli":
        return _codex_cli(messages, model)
    elif backend == "api":        # post-MVP
        return _api(messages, model)
```

MVP supports `claude-cli` and `codex-cli` (no API key required — uses the user's Claude or Codex subscription). Per-agent backend is selected via env var `AUTOQEC_<ROLE>_BACKEND` (set by Makefile or CLI flags). Mixing is allowed: e.g., Ideator on `claude-cli` + Claude Opus, Coder on `codex-cli` + GPT-5.4.

Inline mode does not use this router for agent calls — it dispatches subagents via the `Agent` tool directly inside the current Claude Code session. The router is used only by background mode (Python DAG) and by Runner-internal sub-calls (if any).

### 4.8 Pareto & metrics

- **Accuracy**: LER at fixed `p` for MVP. Threshold-scan `p_th` post-MVP.
- **Latency**: **FLOPs** (hardware-agnostic, primary axis) + wall-clock (host-dependent, reference column). Measured via `fvcore` / `ptflops`.
- **Params**: `sum(p.numel() for p in model.parameters())`.
- **Training cost**: recorded in notebook but **not a Pareto axis** (to avoid "train less to look cheaper" reward hacking).

Pareto maintenance lives in `autoqec/pareto/front.py`; admission requires `/verify-decoder` = `VERIFIED`.

## 5. Environments

### 5.1 MVP (two reference envs, shipped in `envs/builtin/`)

1. **surface_d5_depol.yaml** — Rotated surface code d=5, circuit-level depolarizing, sweep `p ∈ {1e-3, 5e-3, 1e-2}`. Baseline: PyMatching (MWPM). Story: "harness matches MWPM, explores low-latency variants."
2. **bb72_depol.yaml** — BB code [[72,12,6]], phenomenological depolarizing. Baseline: ldpc BP+OSD. Story: "agent discovers GNN/Neural-BP variants on the Pareto front."

### 5.2 Stretch (post-MVP)

3. **chip_leakage.yaml** — custom DEM from calibration data (or synthetic biased + leakage). Showcases non-Pauli noise — the original PDF's key claim that classical decoders can't handle.

### 5.3 User extension

Users add envs via:
- YAML + `autoqec run envs/my_env.yaml` (primary path)
- `.py` escape hatch for custom channels / code classes (post-MVP)
- `/add-env` skill for non-Python users (interactive)

## 6. Decoder Artifacts

### 6.1 B-DSL (structured configuration)

`autoqec/decoders/dsl_schema.yaml` defines the template grammar:

```yaml
decoder:
  type: tanner_gnn | neural_bp | linear_bp_unrolled | ...
  layers: int
  hidden_dim: int
  message_fn: mlp_gated | attention | gru | ...
  aggregation: sum | mean | attention | ...
  residual: bool
  normalization: none | layer | batch
  bp_iterations: int           # only for neural_bp
  ...
```

`dsl_compiler.py` maps config → `nn.Module`. Coder agent writes YAML fragments matching this schema.

### 6.2 C-PyTorch (free-form source)

Coder writes a full `.py` module implementing:

```python
class Decoder(nn.Module):
    def __init__(self, env_spec: EnvSpec): ...
    def forward(self, syndromes: Tensor) -> Tensor: ...
    # Must return shape (batch, n_logical_obs)
```

Validated by a smoke test in the Coder→Runner handoff (construct model, forward a dummy batch, check shapes). Failures trigger linear retry (MVP) or debug subtree (post-MVP AIDE pattern).

## 7. Tree-Search Upgrade Path

MVP starts linear on both inner and outer loops; upgrade paths are additive:

- **Layer B upgrade** (Coder↔Runner → AIDE solution tree): per hypothesis, spawn ≥2 drafts in parallel; failing drafts branch into debug subtrees; successful drafts branch into refine subtrees. Best-first expansion.
- **Layer C upgrade** (Ideator → v2-style hypothesis tree): multi-hypothesis parallel expansion from current best Pareto node; score-based pruning.

In Inline mode, "parallel" is natural — one message with multiple `Agent` tool calls runs them concurrently.

## 8. Reward-Hacking & Safety Defenses

| Layer | Defense |
|---|---|
| Tool whitelists | Coder has no Bash; Ideator no Write; Analyst read-only |
| Seed isolation | Train / val / holdout sets; Coder/Ideator/Runner never see holdout |
| Runner determinism | Non-LLM, deterministic, all seeds explicit |
| Independent verification | `/verify-decoder` uses an isolated eval module (CI-enforced no-import-runner) |
| Ablation sanity | Shuffled weights must collapse to baseline; zero syndrome → clean output |
| Per-agent cost cap | `per-round` / `per-env` / `per-day` (defaults $0.50 / $20 / $50 for API mode; token quotas for CLI mode) |
| Subprocess timeouts | Each `claude -p` / `codex exec` / training job has a hard wall-clock limit |

## 9. Developer Skills (`.claude/skills/`)

Mirrors the problem-reductions pattern:

| Skill | Purpose |
|---|---|
| `/autoqec-run` | User entry point (inline or detached) |
| `/add-env` | Interactive: turn domain expert's chip data / description into an env YAML, open PR |
| `/add-decoder-template` | Interactive: contribute a new DSL template or PyTorch skeleton to seed Coder |
| `/verify-decoder` | Standalone audit of a Pareto candidate (used inline + as independent tool) |
| `/review-log` | Read a run's notebook; grade narrative quality, flag overfitting/reward-hacking signs |
| `/topology-check` | Emit gap/coverage analysis across accumulated runs |

## 10. Contributor Model

Inspired by problem-reductions:

- Contribute 5 verified envs → paper authorship
- Contribute 3 DSL templates that reach Pareto front on some env → authorship
- Issues tagged `[env]` / `[template]` / `[rule]` feed the `make run-forever` queue (optional GitHub Projects integration post-MVP)

## 11. Repository Structure

```
qec-ai-decoder/
├── autoqec/                    # Python package
│   ├── __init__.py
│   ├── orchestration/
│   ├── agents/                 # Python wrappers around Agent SDK / subprocess calls
│   ├── schemas/                # dsl_schema.yaml, env_schema.yaml, decoder_template_index.md
│   ├── runner/
│   ├── eval/
│   │   ├── __init__.py
│   │   └── independent_eval.py # ISOLATED
│   ├── decoders/
│   ├── envs/
│   │   ├── loader.py
│   │   └── builtin/            # surface_d5, bb72, (chip_leakage post-MVP)
│   ├── example_db/             # canonical builders (SSOT)
│   ├── llm/
│   │   └── router.py           # backend: claude-cli / codex-cli / (api post-MVP)
│   ├── logging/
│   └── pareto/
├── cli/
│   └── autoqec.py              # click-based CLI
├── .claude/
│   ├── agents/                 # SSOT subagent definitions
│   │   ├── autoqec-ideator.md
│   │   ├── autoqec-coder.md
│   │   ├── autoqec-analyst.md
│   │   └── autoqec-reviewer.md
│   └── skills/
│       ├── autoqec-run/SKILL.md
│       ├── add-env/SKILL.md
│       ├── add-decoder-template/SKILL.md
│       ├── verify-decoder/SKILL.md
│       ├── review-log/SKILL.md
│       └── topology-check/SKILL.md
├── envs/                       # symlink or mirror of autoqec/envs/builtin/
├── runs/                       # .gitignore'd; per-run output dirs
├── circuits/                   # .stim files for reference envs
├── tests/
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-20-autoqec-design.md   # this file
├── Makefile                    # run, run-forever, run-cheap, run-with-codex, ...
├── pyproject.toml
├── README.md                   # three user-persona entry sections
└── .gitignore
```

## 12. Makefile

Orchestrates per-agent backend and model selection:

```makefile
# Defaults — MVP
IDEATOR_BACKEND   ?= claude-cli
IDEATOR_MODEL     ?= claude-opus-4-7
CODER_BACKEND     ?= claude-cli
CODER_MODEL       ?= claude-sonnet-4-6
ANALYST_BACKEND   ?= claude-cli
ANALYST_MODEL     ?= claude-haiku-4-5
REVIEWER_BACKEND  ?= claude-cli
REVIEWER_MODEL    ?= claude-opus-4-7

run:              python -m autoqec run $(ENV) --rounds $(ROUNDS)
run-cheap:        $(MAKE) run IDEATOR_MODEL=claude-haiku-4-5
run-with-codex:   $(MAKE) run CODER_BACKEND=codex-cli CODER_MODEL=gpt-5.4-codex
run-forever:      while true; do NEXT=$$(autoqec queue pop); \
                    [ -z "$$NEXT" ] && sleep 60 && continue; \
                    $(MAKE) run ENV=$$NEXT; \
                  done
verify:           python -m autoqec verify $(RUN_DIR)
test:             pytest tests/ -v
```

## 13. Testing Strategy

- **Unit tests** (CPU-only, <5s each): DSL compilation, EnvSpec loading, Pareto maintenance, Runner on toy decoder + tiny env.
- **Integration tests** (marked `@pytest.mark.integration`): end-to-end 2-round run with Haiku on `surface_d5` minimal config.
- **Reward-hacking regression tests**: construct a synthetic "cheating" decoder (returns memorized answers for val seeds), confirm `/verify-decoder` returns `FAILED` on holdout.
- **Prompt regression tests**: for each `.claude/agents/*.md`, mock LLM responses and check the Python wrapper parses + forwards correctly.

## 14. MVP Acceptance Criteria

A successful MVP demo must show:

1. `autoqec run envs/surface_d5_depol.yaml --rounds 20 --backend claude-cli` produces:
   - A `runs/<id>/log.md` notebook with 20 rounds of (hypothesis / code / metrics / analyst summary)
   - A Pareto PNG with ≥3 VERIFIED candidates
   - At least one candidate matching PyMatching LER within 1σ
2. Same for `envs/bb72_depol.yaml` — at least one VERIFIED candidate strictly Pareto-dominates BP+OSD on (FLOPs, LER).
3. `/add-env` skill successfully walks a non-expert through adding a third env YAML.
4. `/verify-decoder` correctly flags a hand-constructed cheating decoder as `FAILED`.
5. Running the exact same env config with `IDEATOR_MODEL=claude-haiku-4-5` (cheap ablation) on 10 rounds completes and produces a degraded but valid Pareto — evidence that backend routing works.

## 15. Open Risks

| Risk | Mitigation |
|---|---|
| Coder's PyTorch free-form path has high failure rate (≥40% per Sakana v1) | MVP defaults to DSL path; PyTorch escape hatch is opt-in per hypothesis; linear retry × 3 then skip |
| Inline mode context window overflow on long runs | Subagents isolate per-role context; main orchestrator summary stays <2k tokens per round |
| BB qLDPC baseline integration (ldpc/stimbposd) has rough edges | `autoqec/decoders/baselines/` wraps each with a uniform interface, pinned to specific package versions |
| `claude -p` / `codex exec` subprocess flakiness over long runs | Per-call timeouts + retry-with-backoff; Checkpoint after every round to allow resume |
| Reward hacking we didn't anticipate | `/verify-decoder` + tool whitelists + CI ban on cross-module imports; manual inspection of top-Pareto decoders before any public claim |

## 16. Deferred to Post-MVP

- API backend mode
- AIDE solution-tree inside Coder↔Runner loop
- v2-style hypothesis-tree on Ideator
- Real-chip small-sample fine-tuning
- Third env (`chip_leakage`) with non-Pauli noise
- GitHub Projects integration for `run-forever`
- Auto-paper draft generation
- Web dashboard
- Python-code env escape hatch

---

## Appendix A — Survey summary (informs this design)

- **AI Scientist v2** (Sakana, 2025): agentic tree search + Experiment Manager — inspires our hypothesis-tree upgrade path.
- **AIDE** (Weco): best-first solution tree — inspires our Coder↔Runner inner loop upgrade.
- **Agent Laboratory** (MIT): role-based multi-agent — closest prior art to our DAG structure.
- **Karpathy `autoresearch`** (2026): minimal ~630 LOC loop — inspires our DSL-only MVP baseline.
- **MLR-Bench**: reports ~80% of coding agents fabricate results — motivates independent `/verify-decoder`.
- **Sakana DGM / SIGIR review**: reward-hacking case studies — motivates tool whitelists + isolated eval module.
- **problem-reductions** (CodingThrust, Rust library, 2026): skill-as-pipeline-stage pattern, `verify-reduction` subroutine, single-source-of-truth example DB, Makefile with run-forever + dual-LLM backend, contributor authorship model — all adopted.
- **QEC-decoder auto-research gap**: landscape search found no AI-Scientist-class loop targeting QEC decoders; hand-designed neural decoders (AlphaQubit, GNNs, Neural-BP) exist but none are auto-discovered. AutoQEC claims novelty on closing this loop.

## Appendix B — Glossary

- **DEM**: Detector Error Model (Stim's native fault description)
- **LER**: Logical Error Rate
- **MWPM**: Minimum-Weight Perfect Matching (PyMatching)
- **BP+OSD**: Belief Propagation with Ordered Statistics Decoding
- **Tanner graph**: bipartite graph of checks and variables defining a code's parity checks
- **qLDPC**: quantum Low-Density Parity-Check code (e.g., BB code [[72,12,6]])
- **FLOPs**: Floating-point operations; used here as hardware-agnostic latency proxy
- **Pareto front**: set of non-dominated points in (accuracy, latency, params) space
