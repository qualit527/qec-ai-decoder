# AutoQEC — Design Specification

**Version**: v2.3
**Date (v1)**: 2026-04-20
**Date (v2)**: 2026-04-21
**Date (v2.2)**: 2026-04-21
**Date (v2.3)**: 2026-04-22
**Status**: Draft, ready for team review
**Authors**: Team (Chen Jiahan、Xie Jingu、Lin Tengxiang) + brainstorming with Claude
**Project**: QEC AI-enhanced decoder (AutoQEC)
**Repo**: `qec-ai-decoder`
**Companion docs**: `knowledge/DECODER_ROADMAP.md` · `knowledge/STRATEGIC_ASSESSMENT.md` · `knowledge/AUTORESEARCH_PATTERNS.md`

---

## Revision history

- **v1 (2026-04-20)**: initial brainstorming spec. Assumed open-ended timeline; full multi-agent DAG with Reviewer; two reference envs as MVP; dual DSL + PyTorch paths; tree-search upgrade path. See git history.
- **v2 (2026-04-21 morning)**: rescoped for a **1-week timeline**. Narrowed search space to **AI predecoder + classical backend** (GNN or Neural-BP → MWPM or OSD). Dropped Reviewer agent, PyTorch C-path, tree-search upgrades, background-mode checkpointing, contributor infrastructure. Added: **Tier-1 rich DSL + Tier-2 custom-fn escape hatch**, `machine_state` tool for self-aware compute budgeting, three-layer memory architecture, 5 skills and 5 demos (per project deliverable requirement), codex-cli as primary production backend. Novelty framing updated per `STRATEGIC_ASSESSMENT.md` to Framing B (generic Pareto discovery, not SOTA-beating).
- **v2.1 (2026-04-21 afternoon)**: timeline compressed from 5 days to **3 days** (project reality). Core infrastructure (`DSL`, `independent_eval`, `Pareto`) moved to Days 1-2; skills consolidated to Day 3 thin wrappers over already-working CLIs. Demos and skills tagged with P0 / P1 / P2 priorities; 5-skill and 5-demo deliverables are preserved but P2 items are minimum-viable.
- **v2.2 (2026-04-21 evening)**: added a recommended **3-person ownership split** across **Claude Code, Codex, GLM**. Each contributor now owns one QEC-core workstream plus one delivery-facing workstream, with explicit Day 1-3 responsibilities, shared checkpoints, and demo ownership.
- **v2.3 (2026-04-22)**: added **§15 Worktree-based experiment model**. Each research round gets its own git branch and optional `.worktrees/<id>/` checkout (ported from `problem-reductions/scripts/pipeline_worktree.py`, MIT). The fork graph (not a linear commit history on `main`) is the research trajectory; Pareto members are commit SHAs (with branch names as human-readable aliases); failed hypotheses live on as named branches for negative-result value. Rationale: performance gains in neural-predecoder research are **not additive**, so auto-merging VERIFIED branches into `main` creates drifting baselines and hides mechanism-level conflicts. "Compose rounds" let the Ideator explicitly request `git merge branch-A branch-B`, testing compositional claims as first-class scientific questions — with a mandatory paired-holdout comparison protocol (§15.6.4) so claims survive training variance. Runner gains an optional `code_cwd` field; when set, the Runner is invoked as a subprocess so it picks up worktree-local edits to `autoqec/decoders/modules/*.py`. Schema changes are **not uniformly additive**: `memory.py`'s `l3_for_ideator` payload shape changes, `round_recorder.py`'s Pareto algorithm changes from top-5 sort to non-dominated filter, `agents/schemas.py` must declare new fields under `extra="forbid"`. §15.9 splits these into "hard API changes" vs "pure additive" with an explicit 8-step landing order. `commit_sha` (not branch name) is the canonical provenance key. Startup reconciliation (§15.10) repairs git-branch / `history.jsonl` drift from crashed runs.

---

## 1. Overview

AutoQEC is a generic, LLM-agent-driven auto-research harness for discovering **neural predecoders** for quantum error-correcting codes. The user provides an environment triple

```
(code_spec, noise_model, constraints)
```

and the system autonomously runs 10-20 research rounds of *hypothesis → DSL config → training → evaluation → analysis*. It outputs verified checkpoints on the accuracy–latency–parameters Pareto front plus a reproducible markdown research notebook.

The discovered artifact is always a **predecoder**: a small neural network that either (a) cleans the syndrome (`hard_flip` mode) or (b) emits per-fault priors (`soft_priors` mode). The cleaned syndrome or priors feed a fixed classical backend — MWPM (PyMatching) for surface codes, OSD for qLDPC — which guarantees structural correctness of the logical correction. This "predecoder + classical" decomposition makes the system deploy-safe (worst case: predecoder adds no value; it cannot produce incorrect outputs) and sharpens the measurement (Δ_LER versus the bare classical backend is a single clean number per round).

## 2. Novelty positioning

> **AutoQEC is the first LLM-agent-driven discovery engine for quantum error-correction decoders.**
> Given any `(code, noise, constraint)` triple, it systematically produces Pareto fronts of reproducibility-verified neural predecoders — turning hand-craft decoder design into a scalable, auditable research workflow.

Defensible claims the MVP evidence supports:
1. **First auto-research system targeting QEC decoder design.** Confirmed by literature survey (81 papers, Apr 2026). The only prior NAS-for-neural-QEC work is Overwater 2022 (FFN-only, non-agent, single-env).
2. **Generic `(code, noise, constraint) → Pareto` reformulation.** Traditional decoder papers report one architecture × one code × one point. AutoQEC outputs the Pareto curve for any triple using the same harness.
3. **Reproducibility-by-design.** Reward-hacking defenses (isolated eval module, tool-whitelisted subagents, holdout-seed isolation, bootstrap-CI, ablation sanity checks) baked into the harness at launch rather than retrofitted.

Target venue: **Quantum** (primary), IEEE QCE (secondary). Skip Nature QI / PRX Quantum — they expect SOTA-beating numbers we do not claim.

## 3. Goals & Non-Goals

### Goals (1-week MVP)

- **G1.** Working end-to-end harness: user invokes a skill / runs CLI → agent loop completes ≥10 rounds → outputs verified Pareto candidates.
- **G2.** Two reference environments as evidence: `surface_d5_depol` (PyMatching baseline) and `bb72_depol` (BP+OSD + Relay-BP baselines). The same harness code runs both.
- **G3.** 5 user-facing skills covering the user journey (see §8).
- **G4.** 5 demos, each producing a reproducible artifact, together exercising all skills (see §9).
- **G5.** Independent decoder verification catches a hand-crafted cheating predecoder.
- **G6.** Codex-cli primary backend (server deployment friendly); claude-cli works for inline development and demos.
- **G7.** Agent is *self-aware* about compute: no hard DSL limits, Ideator queries `machine_state` tool and reads historical round timings.

### Non-Goals (MVP — explicitly deferred to post-MVP)

- **N1.** Full Pareto coverage across all code families.
- **N2.** Automatic paper generation (Sakana writer stack).
- **N3.** FPGA / ASIC deployment. Latency reported as FLOPs (hardware-agnostic).
- **N4.** Real-chip small-sample fine-tuning.
- **N5.** Web dashboard.
- **N6.** Reviewer subagent (meta-reflection over history).
- **N7.** AIDE solution-tree or AI-Scientist-v2 tree-search upgrades.
- **N8.** PyTorch free-form C-path (only DSL Tier 1 + Tier 2 escape hatch).
- **N9.** LangGraph SQLite checkpoint-resume. Crashes restart from disk state.
- **N10.** Contributor authorship program / community infra.
- **N11.** Third env with non-Pauli noise (leakage, coherent, biased).
- **N12.** API backend for LLM calls (claude-cli + codex-cli suffice for all targets).
- **N13.** RAG over paper corpus via paper-qa (Ideator reads curated excerpts from DECODER_ROADMAP.md).

## 4. Architecture

### 4.1 Two-layer decoder stack

```
syndrome + code DEM
        │
        ↓
┌─────────────────────────────────────────┐
│ AI Predecoder  (the agent searches here)│
│  type:        gnn | neural_bp           │
│  output_mode: hard_flip | soft_priors   │
└─────────────────────────────────────────┘
        │   (cleaned syndrome, OR per-fault priors)
        ↓
┌─────────────────────────────────────────┐
│ Classical Backend  (fixed per env)      │
│   surface codes → MWPM (PyMatching)     │
│   qLDPC         → OSD                   │
└─────────────────────────────────────────┘
        │
        ↓
logical correction
```

The classical backend guarantees structural validity; the predecoder's contribution is quantified as `Δ_LER = LER(plain_classical) − LER(predecoder + classical)`.

### 4.2 Agent DAG (3 subagents)

```
          ┌─────────────────┐
          │ Main orchestrator│  (Claude Code inline chat OR
          │                  │   server-side Python process)
          └────────┬─────────┘
                   │ per round
     ┌─────────────┼─────────────┐
     ↓             ↓             ↓
┌─────────┐  ┌─────────┐  ┌─────────┐
│Ideator  │  │ Coder   │  │ Analyst │
│R,G,Gl   │  │R,W,E,G  │  │  R,G    │
│+machine │  │         │  │         │
│_state   │  │         │  │         │
└────┬────┘  └────┬────┘  └────▲────┘
     │            │            │
     └─────┬──────┘            │
           ↓                   │
     ┌─────────────────┐       │
     │     Runner      │───────┘
     │  (pure Python,  │
     │   non-LLM)      │
     └─────────────────┘
```

**Subagent roles and tool whitelists** (tool whitelist is a *physical* reward-hacking defense):

| Role | Tools | Job |
|---|---|---|
| **Ideator** | Read, Grep, Glob, `machine_state` | Reads env_spec + history + KB excerpts; emits hypothesis + expected Δ |
| **Coder** | Read, Write, Edit, Grep, Glob | Reads hypothesis + DSL schema; emits DSL config YAML (Tier 1) or `custom_fn` snippet (Tier 2). **No Bash** — cannot run training. |
| **Analyst** | Read, Grep | Reads runner's metrics.json; writes round report. **Read-only** — cannot change code or checkpoints. |
| **Runner** (non-LLM) | All file tools + Bash | Compiles DSL, runs training, evaluates on val seeds, writes metrics.json + checkpoint.pt. Invoked only by orchestrator/main process. |

Reviewer is deliberately excluded from MVP; the main orchestrator itself handles stop-conditions.

### 4.3 Three-layer memory architecture

```
┌────────────────────────────────────────────────────────────┐
│ Layer 1 — Disk (source of truth, persistent)               │
│   runs/<id>/                                               │
│     ├── log.md         append-only narrative               │
│     ├── history.jsonl  per-round record (hypothesis,config,│
│     │                  metrics,timing,vram,verdict)        │
│     ├── pareto.json    current verified Pareto front       │
│     └── round_<N>/                                         │
│         ├── config.yaml                                    │
│         ├── train.log                                      │
│         ├── metrics.json                                   │
│         └── checkpoint.pt                                  │
└────────────────────────────────────────────────────────────┘
            ↓ summarize-on-write (1 line per round)
┌────────────────────────────────────────────────────────────┐
│ Layer 2 — Main orchestrator context (<2k tokens)           │
│   env_spec summary + current Pareto ≤5 + last-3-round      │
│   one-liners + budget consumed. Rebuilt each round from L1.│
└────────────────────────────────────────────────────────────┘
            ↓ dispatch per-round
┌────────────────────────────────────────────────────────────┐
│ Layer 3 — Per-subagent context (3-5k tokens)               │
│   Ideator: env + Pareto + last 5 hypothesis + KB excerpts  │
│   Coder:   hypothesis + DSL schema + 3 best-so-far configs │
│   Analyst: round metrics + previous-round delta            │
└────────────────────────────────────────────────────────────┘
```

Core rule: **summarize on write, not on read**. Each round's entry is compressed to a 1-line summary when persisted; subsequent rounds read summaries, not raw.

Crash recovery: disk is canonical. If the main chat closes or the process dies, the next invocation reads L1 to rebuild L2 and continue.

### 4.4 Backend routing

```python
# autoqec/llm/router.py
def call_llm(messages, model, backend) -> str:
    if backend == "codex-cli":   # MVP primary (server deploy friendly)
        return _codex_cli(messages, model)    # subprocess `codex exec ...`
    elif backend == "claude-cli": # MVP secondary (inline dev / Claude Code demo)
        return _claude_cli(messages, model)   # subprocess `claude -p ...`
    elif backend == "api":        # post-MVP
        raise NotImplementedError
```

Inline mode (Claude Code main chat) uses the `Agent` tool directly instead of the router — no subprocess. Router is used only by background mode (Python process) and by any sub-tool that makes its own LLM calls.

Per-agent backend overrides via env vars so a single Makefile invocation can mix backends:

```bash
# Makefile
AUTOQEC_IDEATOR_BACKEND ?= codex-cli
AUTOQEC_IDEATOR_MODEL   ?= gpt-5.4
AUTOQEC_CODER_BACKEND   ?= codex-cli
AUTOQEC_CODER_MODEL     ?= gpt-5.4-codex
AUTOQEC_ANALYST_BACKEND ?= claude-cli  # cheap role, can use Haiku
AUTOQEC_ANALYST_MODEL   ?= claude-haiku-4-5
```

### 4.5 Runner and `RunnerSafety`

The runner (`autoqec/runner/`) is pure Python, deterministic, seed-pinned. It does not call any LLM. Its interface:

```python
def run_round(config: dict, env_spec: EnvSpec, round_dir: Path) -> RoundMetrics
```

Safety sentinels are **runtime guards, not design constraints** — the DSL itself imposes no hard `max_params` / `max_layers`:

```python
class RunnerSafety:
    WALL_CLOCK_HARD_CUTOFF_S = 2700     # 45 min — kill if exceeded
    VRAM_PRE_CHECK            = True    # estimate VRAM, abort before OOM
    MAX_NAN_RATE              = 0.01    # kill training if NaN rate >1%
    FORBIDDEN_IMPORTS         = ["os.system", "subprocess", "sys.exit"]  # for Tier-2 custom_fn
```

If a round hits a sentinel, it is logged as `status: "killed_by_safety"` with the reason; Analyst writes it into history. Ideator sees this in the next round via `machine_state.history_timings` and naturally adjusts.

### 4.6 Independent verify-decoder

Every Pareto candidate must be verified before admission. The verification module is **physically isolated**:

```python
# autoqec/eval/independent_eval.py
"""
ISOLATED — must not import autoqec.runner.*
CI check enforces this.
"""

def independent_verify(checkpoint: Path, env_spec: EnvSpec,
                        holdout_seeds: list[int]) -> VerifyReport
```

MVP enforces **3 of 6** fair-baseline guards from [@bbfair2026] (remaining 3 deferred to v2.1):

1. **Seed isolation**: holdout seeds `9000-9999` are never seen by train / val / Coder / Ideator.
2. **Bootstrap 95% CI**: 200K holdout shots, bootstrap resample 1000× for LER confidence interval.
3. **Ablation sanity**: randomly shuffle predecoder weights → LER must collapse toward baseline (evidence the model actually learned something).

Deferred to post-MVP:
4. Two-p slope check (generalization across noise rates).
5. DEM-from-train-only verification.
6. Extrapolation-cycle check.

Output of verify: `{VERIFIED | SUSPICIOUS | FAILED}` plus a `verification_report.md`. Only `VERIFIED` candidates enter the final Pareto. `SUSPICIOUS` are flagged for human review; `FAILED` are discarded with the report archived.

## 5. Environments

### 5.1 EnvSpec schema

```yaml
# envs/<name>.yaml
name: <str>
code:
  type: stim_circuit | parity_check_matrix | tanner_graph
  source: <path>
noise:
  type: depolarizing | biased | leakage | custom_dem
  p: [<list of physical error rates to sweep>]
  seed_policy:
    train:   [1, 999]
    val:     [1000, 1999]
    holdout: [9000, 9999]   # Coder/Ideator/Runner must not access
constraints:
  latency_flops_budget: <int>       # soft guidance, not hard
  param_budget: <int>                # soft guidance
  target_ler: <float>
  target_p: <float>
baseline_decoders:
  - pymatching          # always for surface
  - bposd               # ldpc v2
  - relay_bp            # ldpc v2 (if available)
classical_backend: mwpm | osd
eval_protocol:
  min_shots_train: 1_000_000
  min_shots_val: 100_000
  min_shots_verify: 200_000
  bootstrap_ci: 0.95
  osd_orders_reported: [0, 10]     # both reported per fair-baseline
  x_z_decoding: circuit            # or: x_only
```

Three groups (`code`, `noise`, `constraints`) are decoupled — users recombine freely.

### 5.2 MVP envs

**`envs/surface_d5_depol.yaml`** (required):
- Rotated surface code, distance d=5, circuit-level depolarizing noise
- p sweep: `[1e-3, 5e-3, 1e-2]`
- Baselines: PyMatching (MWPM) with uniform priors
- Classical backend for predecoder: MWPM
- Stim circuit obtained via `stim.Circuit.generated(...)` — standard library helper

**`envs/bb72_depol.yaml`** (stretch, Day 4+):
- Bivariate Bicycle code [[72, 12, 6]], phenomenological depolarizing
- p sweep: `[1e-3, 3e-3, 5e-3]`
- Baselines: plain BP+OSD (order 0 and 10), Relay-BP
- Classical backend for predecoder: OSD
- Stim circuit path TBD during Day-1 environment bring-up — candidates: (a) `qLDPC` Python package, (b) `stimbposd` examples, (c) Bravyi et al 2024 github. If none usable, hand-construct ~200 LOC (documented in DECODER_ROADMAP.md §3).

### 5.3 User extensions

- Primary path: write YAML → `autoqec run envs/my_env.yaml`
- Non-Python users: `/add-env` skill walks them through it interactively
- Python escape hatch for custom noise channels: post-MVP

## 6. Predecoder DSL (Tier 1 + Tier 2)

### 6.1 Tier 1 — canonical DSL

Tier 1 enumerates published building blocks from `DECODER_ROADMAP.md §5`. The agent's primary action space is organized by family:

```yaml
predecoder:
  type: gnn | neural_bp
  output_mode: hard_flip | soft_priors

  # ─── GNN family ────────────────────────────────
  layers: <int>                          # continuous, agent-chosen
  hidden_dim: <int>                      # continuous
  message_fn: mlp | gated_mlp | attention | gru_cell | edge_attention | geometric_attention | residual_mlp | normalized_mlp
  aggregation: sum | mean | max | attention_pool | set_transformer | gated_sum
  normalization: none | layer | batch | edge_norm | graph_norm
  residual: <bool>
  edge_features: [syndrome_bit, round_idx, stabilizer_type, distance, prior_weight]   # multi-select

  # ─── Neural-BP family ───────────────────────────
  bp_iterations: <int>                   # continuous
  weight_sharing: none | per_layer | per_check
  damping: fixed | learnable_scalar | learnable_per_iter
  attention_aug: <bool>
  attention_heads: <int>                 # if attention_aug=true

  # ─── Training block (both families) ─────────────
  head: linear | mlp_small
  training:
    learning_rate: <float>
    batch_size: <int>
    epochs: <int>
    loss: bce | focal | weighted_bce
    profile: dev | prod
```

**Discrete choices** are enumerable by the Ideator; **continuous fields** are agent-proposed based on `machine_state` and history. `profile: dev` uses 100K shots × 3 epochs (~3 min/round) for rapid iteration; `profile: prod` uses 1M shots × 10 epochs (~15-25 min/round) for serious runs.

Discrete combinations are substantial (GNN alone: 8 × 6 × 5 × 2 × 2^5 ≈ 15,000 canonical forms) — agent exhaustion is not a concern in 10-20 rounds.

### 6.2 Tier 2 — `custom_fn` escape hatch

When Ideator believes the solution lies outside Tier 1, Coder may propose a custom function for any of `message_fn`, `aggregation`, `head`:

```yaml
predecoder:
  type: gnn
  output_mode: soft_priors
  message_fn:
    type: custom
    code: |
      def message(x_src, x_dst, e_ij, params):
          gate = torch.sigmoid(params["W_gate"](e_ij))
          return gate * params["W_src"](x_src) + (1 - gate) * params["W_dst"](x_dst)
    params_declared:
      W_gate: Linear(edge_dim, 1)
      W_src: Linear(hidden, hidden)
      W_dst: Linear(hidden, hidden)
```

**Validation pipeline** before compilation:
1. **AST parse** — must be a single function definition.
2. **Signature check** — inputs / outputs match the expected contract for that slot.
3. **Import whitelist** — only `torch`, `torch.nn.functional`; no `os`, `subprocess`, `sys.exit`.
4. **Smoke test** — compile with declared params, run forward on a dummy batch, check output shape + no NaN.

If any step fails, `custom_fn` is rejected and the round falls back to a Tier-1 default with an error logged to `history.jsonl`. Expected Tier-2 validation failure rate: <15% (vs 42% for v1's full PyTorch free-form C-path).

### 6.3 No hard DSL limits — the agent is self-aware about compute

Historically, DSLs encode `max_params`, `max_layers`, etc. AutoQEC does not. Instead:

- **DSL accepts any value** in continuous fields.
- **Runner enforces runtime safety** (§4.5): wall-clock cutoff, VRAM pre-check, NaN-rate monitor. Catastrophe → round logged as `killed_by_safety`, continues.
- **`machine_state` tool** exposed to Ideator subagent:

```python
# autoqec/tools/machine_state.py
def machine_state(run_dir: Path) -> dict:
    return {
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": ...,
            "vram_free_gb": ...,
        },
        "history_timings": {
            "rounds_so_far": ...,
            "wall_clock_mean_s": ...,
            "wall_clock_p95_s": ...,
            "params_vs_time": [(params, wall_s), ...],     # scatter of observations
            "killed_by_safety_count": ...,
        },
        "budget": {
            "total_wallclock_s_spent": ...,
            "total_wallclock_s_remaining": ...,
        },
    }
```

- **Ideator prompt** explicitly instructs: *"Before proposing, call `machine_state`. Use historical `params_vs_time` to estimate wall-clock for your candidate. Stay within `budget.total_wallclock_s_remaining`. No hard caps exist — you decide what's feasible."*

This frames compute as a **first-class variable the agent optimizes**, matching real researcher behavior. Plateau in LER → agent redirects to smaller / faster architectures or switches family. Failures (timeouts, OOM) become data that Ideator uses next round.

## 7. Evaluation & Pareto

### 7.1 Metrics

- **Accuracy**: `Δ_LER = LER(plain_classical) − LER(predecoder + classical)`. Measured with bootstrap 95% CI over 200K holdout shots.
- **Latency proxy**: FLOPs per syndrome evaluation (via `fvcore` / `ptflops`). Hardware-agnostic, reproducible. Wall-clock reported as a secondary column (host-dependent).
- **Params**: `sum(p.numel() for p in model.parameters())`.
- **Cost**: LLM tokens + GPU wall-clock per round. Logged to `history.jsonl`, **not** a Pareto axis (would create a reward-hacking incentive to short-train).

### 7.2 Fair-baseline compliance

The `/verify-decoder` skill enforces 3 of 6 guards for MVP (see §4.6). Post-MVP will add the remaining 3.

### 7.3 Per-env baselines

| env | classical baseline for Δ_LER | stretch baselines reported |
|---|---|---|
| `surface_d5_depol` | plain PyMatching (uniform priors) | — (MWPM is the standard) |
| `bb72_depol` | plain BP + OSD (order 0) | BP + OSD (order 10), Relay-BP |

bb72's "NN + OSD beats BP + OSD" claim is the key defensible win — the NN replaces the BP step with something better at producing OSD priors.

## 8. 5 User-Facing Skills

All 5 are `.claude/skills/<name>/SKILL.md` files invokable as `/<name>`. Each requires genuine LLM reasoning (not wrappable as a pure CLI). Skills invoke underlying CLI commands for mechanical work.

### 8.1 `/autoqec-run`

Main entry. Orchestrates the agent loop.

- **Input**: env YAML path, round budget, LLM-budget cap.
- **Behavior**: reads env, loops rounds. Each round: (a) call Ideator subagent, (b) call Coder subagent, (c) invoke Runner CLI, (d) call Analyst subagent, (e) if candidate, invoke `/verify-decoder`, (f) append to log.md + history.jsonl + pareto.json.
- **LLM reasoning**: when to stop, how to route failures, how much to summarize into L2 context.
- **Underlying CLI**: `autoqec run <env.yaml>`.

### 8.2 `/add-env`

Interactive env creation for non-Python users.

- **Input**: user's free-form description, possibly paths to `.stim` or DEM files.
- **Behavior**: Q&A dialog. Parses natural language. Inspects provided files. Fills YAML schema with sane defaults. Validates. Writes `envs/<name>.yaml` and opens it for review.
- **LLM reasoning**: disambiguation (e.g., "surface code" → rotated or unrotated? circuit-level or phenom?), reading `.stim` files to extract code params, inferring plausible noise rate ranges.

### 8.3 `/verify-decoder`

Independent Pareto candidate audit.

- **Input**: path to `runs/<id>/round_N/`.
- **Behavior**: invokes `autoqec verify` CLI (which calls `independent_eval.py`). Reads the mechanical `VERIFIED/SUSPICIOUS/FAILED` verdict + raw metrics. LLM **interprets** borderline cases: reads `training.log`, the DSL config, the history of previous rounds; writes a diagnostic paragraph explaining root cause (overfitting / underfitting / reward-hacking pattern) and recommends next action.
- **LLM reasoning**: interpretation of statistics in context, detective work across multiple files.
- **Underlying CLI**: `autoqec verify <run_dir>`.

### 8.4 `/review-log`

Research-notebook review.

- **Input**: path to `runs/<id>/log.md` (or a set of them).
- **Behavior**: reads entire log.md (may be 20-50 rounds). Judges narrative coherence, flags overfitting signs, notes hypotheses that repeated without recognition, evaluates research value. Writes a review markdown file.
- **LLM reasoning**: reading free-form narrative, detecting subtle patterns (e.g., "rounds 7-10 all propose tiny variants — agent stuck in local basin").

### 8.5 `/diagnose-failure`

On-demand or auto-triggered diagnosis.

- **Input**: path to a failed or stalled `runs/<id>/`.
- **Behavior**: reads recent history, error logs, state. LLM reasons about root cause (DSL hyperparameter bad / adapter bug / learning rate / NaN pattern / VRAM overflow / env config bad). Recommends a fix; optionally emits a patched config for the user to re-run.
- **LLM reasoning**: debugging, cross-reference to known failure modes.
- **No autonomous execution** of the fix — always returns recommendation for human to apply.

## 9. 5 Demos

Each demo is a `demos/demo-N-<slug>/` directory containing `README.md`, `run.sh`, `expected_output/`, and a `walkthrough.md` narrative.

| # | Demo | Skills exercised | Priority | Est. runtime | Success criterion |
|---|---|---|---|---|---|
| **1** | surface_d5 full run | `/autoqec-run`, `/verify-decoder` | **P0** | ~3.3 h | Pareto has ≥3 VERIFIED candidates; at least one matches PyMatching LER within statistical tolerance. |
| **4** | Reward-hacking detection | `/verify-decoder`, `/review-log` | **P0** | ~20 min | Hand-crafted cheating predecoder (memorized train/val syndromes) receives `FAILED` verdict with correct diagnosis on holdout seeds. |
| **2** | bb72 full run | `/autoqec-run`, `/verify-decoder` | **P1** | ~3.3 h | Pareto has ≥3 VERIFIED candidates; at least one *Δ-dominates* plain BP+OSD (order 0) on at least one (accuracy, FLOPs) slice. If time-constrained on Day 3: acceptable fallback is 3 rounds dev-profile with clear qualitative Pareto. |
| **3** | `/add-env` onboarding | `/add-env`, `/autoqec-run` (5 rounds dev profile) | **P2** | ~30 min | Non-coder teammate describes env → skill produces valid YAML → at least 1 round of meaningful output. |
| **5** | Failure recovery | `/diagnose-failure`, `/review-log` | **P2** | ~20 min | Deliberately malformed config (e.g. `hidden_dim: -1`, or NaN-prone hyperparameters) → skill identifies root cause and proposes a fix. |

Demos 1 and 4 are **must-ship** (they prove the core claim). Demo 2 is the **genericity proof**; a simplified fallback is acceptable under Day-3 pressure. Demos 3 and 5 are nice-to-have polish.

## 10. Repository Structure

```
qec-ai-decoder/
├── autoqec/                        # Python package (7 core submodules)
│   ├── __init__.py
│   ├── orchestration/              # research-loop driver + memory + worktree (§15)
│   │   ├── loop.py                 # Ideator/Coder/Analyst prompt assembly
│   │   ├── memory.py               # L1/L2/L3 state — now returns fork_graph
│   │   ├── round_recorder.py       # history.jsonl + pareto.json + log.md
│   │   ├── worktree.py             # §15: ported from problem-reductions/pipeline_worktree.py
│   │   ├── fork_graph.py           # §15.4: tree serialization from git + files
│   │   └── subprocess_runner.py    # §15.8: code_cwd-aware Runner dispatch
│   ├── agents/                     # Python wrappers that dispatch to subagents
│   ├── runner/                     # Non-LLM: train+eval+FLOPs (port aide/interpreter.py)
│   │   └── safety.py               # RunnerSafety sentinels
│   ├── eval/
│   │   └── independent_eval.py     # ISOLATED — CI enforces no-runner-import
│   ├── decoders/
│   │   ├── dsl_schema.py           # pydantic model for Tier-1 DSL
│   │   ├── dsl_compiler.py         # YAML → nn.Module
│   │   ├── custom_fn_validator.py  # Tier-2 AST+smoke checker
│   │   └── baselines/              # pymatching / ldpc / relay_bp wrappers
│   ├── envs/
│   │   ├── loader.py
│   │   └── builtin/
│   │       ├── surface_d5_depol.yaml
│   │       └── bb72_depol.yaml
│   ├── llm/
│   │   └── router.py               # codex-cli / claude-cli / (api stub)
│   ├── tools/
│   │   └── machine_state.py        # tool callable by Ideator subagent
│   ├── logging/                    # markdown lab-notebook generator
│   ├── pareto/                     # front maintenance + topology analysis
│   └── example_db/                 # canonical Tier-1 DSL templates
├── cli/
│   └── autoqec.py                  # click CLI
├── .claude/
│   ├── agents/
│   │   ├── autoqec-ideator.md
│   │   ├── autoqec-coder.md
│   │   └── autoqec-analyst.md
│   └── skills/
│       ├── autoqec-run/SKILL.md
│       ├── add-env/SKILL.md
│       ├── verify-decoder/SKILL.md
│       ├── review-log/SKILL.md
│       └── diagnose-failure/SKILL.md
├── demos/
│   ├── demo-1-surface-d5/
│   ├── demo-2-bb72/
│   ├── demo-3-add-env/
│   ├── demo-4-reward-hacking/
│   └── demo-5-failure-recovery/
├── envs/                           # mirror of autoqec/envs/builtin/
├── circuits/                       # *.stim files
├── runs/                           # .gitignore'd; per-run training output (shared by all branches)
├── .worktrees/                     # .gitignore'd; transient git worktree checkouts per round (§15)
├── knowledge/                      # committed: INDEX, bib, roadmaps; gitignored: PDFs, markdown, refs
├── docs/superpowers/specs/         # this file lives here
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

## 11. Makefile

```makefile
# Backend + model per agent — codex-cli primary
AUTOQEC_IDEATOR_BACKEND ?= codex-cli
AUTOQEC_IDEATOR_MODEL   ?= gpt-5.4
AUTOQEC_CODER_BACKEND   ?= codex-cli
AUTOQEC_CODER_MODEL     ?= gpt-5.4-codex
AUTOQEC_ANALYST_BACKEND ?= claude-cli
AUTOQEC_ANALYST_MODEL   ?= claude-haiku-4-5

COMMON_ENV = \
  AUTOQEC_IDEATOR_BACKEND=$(AUTOQEC_IDEATOR_BACKEND) \
  AUTOQEC_IDEATOR_MODEL=$(AUTOQEC_IDEATOR_MODEL) \
  AUTOQEC_CODER_BACKEND=$(AUTOQEC_CODER_BACKEND) \
  AUTOQEC_CODER_MODEL=$(AUTOQEC_CODER_MODEL) \
  AUTOQEC_ANALYST_BACKEND=$(AUTOQEC_ANALYST_BACKEND) \
  AUTOQEC_ANALYST_MODEL=$(AUTOQEC_ANALYST_MODEL)

.PHONY: run verify pareto test demo-1 demo-2 demo-3 demo-4 demo-5

run:        ; $(COMMON_ENV) python -m autoqec run $(ENV) --rounds $(ROUNDS)
verify:     ; python -m autoqec verify $(RUN_DIR)
pareto:     ; python -m autoqec pareto compare $(RUN_DIRS)
test:       ; pytest tests/ -v

# Demo-specific targets — composable
demo-1: ; bash demos/demo-1-surface-d5/run.sh
demo-2: ; bash demos/demo-2-bb72/run.sh
demo-3: ; bash demos/demo-3-add-env/run.sh
demo-4: ; bash demos/demo-4-reward-hacking/run.sh
demo-5: ; bash demos/demo-5-failure-recovery/run.sh

# Cost ablations — one-command switches
run-all-claude:  $(MAKE) run AUTOQEC_IDEATOR_BACKEND=claude-cli AUTOQEC_CODER_BACKEND=claude-cli
run-cheap:       $(MAKE) run AUTOQEC_IDEATOR_MODEL=claude-haiku-4-5
```

## 12. Three-day delivery timeline

Budget: ~7 net work-days (9 gross team-days − ~25% coordination). Core research artifacts are front-loaded; skills are Day-3 thin wrappers over working CLIs.

### 12.1 Three-person ownership split (recommended)

Principle: **no one is assigned only "glue work."** Each person owns one QEC-core artifact (`code/noise/baseline/predecoder/verify`) and one delivery-facing artifact (`orchestration/skills/demo/docs`) so all three have visible technical ownership and research participation.

Suggested fixed pairing:
- **Chen Jiahan → Claude Code**
- **Xie Jingu → GLM**
- **Lin Tengxiang → Codex**

| Person | Model | Primary ownership | QEC-core responsibility | Delivery-facing responsibility | Demo ownership |
|---|---|---|---|---|---|
| **Chen Jiahan** | **Claude Code** | Orchestration + `surface_d5` environment bring-up | Build and validate `surface_d5_depol`: Stim circuit generation, sinter data path, syndrome extraction, PyMatching baseline, 1M-shot benchmark, and surface-code `Δ_LER` sanity checks | Own orchestrator skeleton, subagent prompt wiring, `/autoqec-run`, `/add-env`, Demo 1 walkthrough | **Demo 1** primary, Demo 3 secondary |
| **Xie Jingu** | **GLM** | Verification + `bb72` / qLDPC evaluation | Own `independent_eval.py`, holdout-seed isolation, bootstrap-CI, ablation sanity check, `bb72_depol` env, BP+OSD / Relay-BP baselines, and qLDPC-side result interpretation | Own `/verify-decoder`, `/review-log`, `/diagnose-failure`, reward-hacking case construction, and final result tables / diagnostic text | **Demo 4** primary, Demo 5 primary, Demo 2 secondary |
| **Lin Tengxiang** | **Codex** | Predecoder stack + Runner | Own `dsl_schema.py`, `dsl_compiler.py`, GNN / Neural-BP templates, Runner training loop, FLOPs counting, and the interface from `hard_flip` / `soft_priors` outputs into MWPM / OSD. This is the main implementation path for the neural QEC predecoder itself | Own Runner integration, config compilation, Makefile / CLI polish, and end-to-end stability for Demo 1 and Demo 2 runs | **Demo 2** primary, Demo 1 secondary |

中文通俗解释：
- **Chen Jiahan（Claude Code）**：负责把整个实验框架和 `surface_d5` 这条 QEC 基础链路搭起来，包括电路生成、syndrome 数据流程、PyMatching 基线，以及把多代理流程串起来。可以把他理解成“总装工程师”，先把实验场地和主流程跑通。
- **Xie Jingu（GLM）**：负责验证和 `bb72` / qLDPC 这条线，包括 `independent_eval.py`、holdout seed、bootstrap CI、ablation sanity check，以及 BP+OSD / Relay-BP 基线。可以把他理解成“质量裁判 + 第二赛道负责人”，负责证明结果不是巧合，也不是作弊。
- **Lin Tengxiang（Codex）**：负责 AI predecoder 本体，也就是项目最核心的神经网络解码器部分，包括 DSL、GNN / Neural-BP 模板、训练 Runner，以及把模型输出接到 MWPM / OSD 这些经典 QEC 后端。可以把他理解成“模型主力开发”，负责把预解码器真正做出来并跑起来。

This split ensures all three directly touch QEC content:
- **Claude Code owner** handles the **surface-code baseline and syndrome pipeline**, not just prompts or docs.
- **Codex owner** handles the **neural predecoder design / training / backend coupling**, which is the core QEC novelty path.
- **GLM owner** handles **fair verification and qLDPC benchmarking**, which is necessary to make the QEC claim publishable rather than anecdotal.

### 12.2 Day-by-day breakdown by person

| Day | Chen Jiahan / Claude Code | Xie Jingu / GLM | Lin Tengxiang / Codex |
|---|---|---|---|
| **Day 1** | Bring up `surface_d5` circuit + PyMatching baseline + 1M-shot benchmark; draft `.claude/agents/autoqec-*.md`; define env/orchestrator input schema | Scout / wire `bb72` source path; wrap BP+OSD baseline; define verify metrics schema and holdout protocol; draft reward-hacking test case | Implement **DSL schema + compiler**; add 3 GNN + 3 Neural-BP seed templates; define Runner config contract and predecoder I/O contract |
| **Day 2** | Integrate orchestrator with Ideator/Coder/Analyst calls; complete one full dev-profile round on `surface_d5`; own integration debugging on the orchestration side | Implement **`independent_eval.py`** and `autoqec verify`; add bootstrap-CI + ablation sanity; wire Pareto update logic and qLDPC eval if time permits | Implement Runner train/eval/FLOPs + `RunnerSafety`; connect predecoder output to MWPM / OSD; fix compile/runtime failures from first end-to-end round |
| **Day 3** | Run **Demo 1** full prod on `surface_d5`; package `/autoqec-run` and `/add-env`; produce walkthrough and lab-notebook narrative | Run **Demo 4** cheating-predecoder audit; finalize `/verify-decoder`, `/review-log`, `/diagnose-failure`; summarize result tables and failure analysis for final PR | Support Demo 1 stability; attempt **Demo 2** `bb72` run; finalize Runner / DSL / CLI polish and fallback configs for dev-profile reruns |

### 12.3 Shared checkpoints and backup coverage

- **Day-1 EOD interface sync (mandatory, 30 min)**: Claude owner brings `env_spec` / orchestration contract; Codex owner brings Runner / DSL contract; GLM owner brings `metrics.json` / verify contract. Freeze these before Day 2 integration.
- **Cross-review rule**: each owner must review one other owner's QEC artifact, not just their prompts or docs.
- **Experiment participation rule**: each owner must personally run or inspect at least one meaningful QEC experiment:
  - Claude owner: one `surface_d5` baseline-vs-predecoder comparison.
  - Codex owner: one predecoder training run with real `Δ_LER` output.
  - GLM owner: one holdout verification or `bb72` baseline comparison.
- **Backup coverage**:
  - Claude owner is backup reader for `independent_eval.py` and demo scripts.
  - Codex owner is backup reader for orchestrator-to-Runner handoff.
  - GLM owner is backup reader for env YAMLs and baseline wrappers.

### Execution principle

Days 1-2 prioritize the novelty-carrying infrastructure (`DSL`, `independent_eval`, `Pareto`, orchestration, Runner). Day 3 packages the working system into user-facing skills, demos, walkthroughs, and final polish.

| Day | Primary outcomes |
|---|---|
| **Day 1** | Port `open-coscientist/framework.py` skeleton; draft 3 subagent `.md` stubs; set up Agent-tool dispatch scaffolding; bring up Stim + sinter and `surface_d5` circuit; add PyMatching baseline wrapper; run **1M-shot train benchmark** to pin compute numbers; implement **DSL schema (pydantic) + `dsl_compiler.py`**; add 3 GNN + 3 Neural-BP seed templates for `example_db/`; add `machine_state` tool |
| **Day 2** | **🔴 Main loop integration** so Runner ↔ Orchestrator ↔ Subagents complete 1 round end-to-end; implement Runner (train + eval + FLOPs) + `RunnerSafety` sentinels; wire `bb72` env if time; implement **`independent_eval.py`** with 3 MVP guards + `autoqec verify` CLI; add **Pareto maintenance + visualization** + `autoqec pareto` CLI |
| **Day 3** | Run Demo 1 (`surface_d5` full prod); attempt Demo 2 (`bb72` full run, P1 stretch); finalize baseline scripts; ship **5 SKILL.md thin wrappers** (P0: `/autoqec-run`, `/verify-decoder`; P1: `/add-env`; P2: `/review-log`, `/diagnose-failure`); finish Demo 4 cheating-predecoder, walkthroughs, and final PR |

### Skill priority (Day 3)

| P | Skills | Target quality |
|---|---|---|
| **P0** (demo-critical) | `/autoqec-run`, `/verify-decoder` | Full working wrappers (~100 LOC SKILL.md each) |
| **P1** (high-value) | `/add-env` | Functional Q&A flow, produces valid env YAML |
| **P2** (minimum viable) | `/review-log`, `/diagnose-failure` | SKILL.md with polished prompts; skill runnable but exercised lightly in Demo 5 only if time permits |

All 5 SKILL.md files exist at end of Day 3 (deliverable fulfilled); depth varies by priority.

### Demo priority (deliverable: 5 demos)

| # | Demo | P | 3-day minimum |
|---|---|---|---|
| 1 | surface_d5 full run | **P0** | 10 rounds prod, Pareto ≥3 verified |
| 4 | Reward-hacking detection | **P0** | Hand-crafted cheating predecoder → FAILED verdict |
| 2 | bb72 full run | **P1 stretch** | 10 rounds if time; else 3-round dev-profile version |
| 3 | `/add-env` onboarding | **P2** | 5-min interactive walkthrough recording |
| 5 | `/diagnose-failure` | **P2** | Broken config → skill emits fix recommendation |

**Worst case (3 days tight)**: Demos 1 + 4 + 3 solid; Demos 2 + 5 have skeleton walkthroughs. Still 5 demos deliverable.
**Best case**: all 5 demos solid.

### Key risk: Day-2 integration misalignment

Day 2 morning the Orchestrator must already call subagents and Runner. If the Runner interface does not match the orchestrator invocation, Day 2 afternoon turns into integration debugging. Mitigation: **Day-1 EOD 30-minute interface sync** plus a shared contract file for the orchestration/Runner boundary.

### Compute wallclock budget

- Dev profile for all iteration work: ~3 min / round × many attempts (fits interactively)
- Prod profile for demos: 10 rounds × ~20 min = ~3.3h per env
- 1 env prod + 1 env dev-scale + 1-2h verify → ~5h total, comfortable on overnight 4090
- **Full 5-demo stretch**: ~7h GPU time; feasible as an overnight or late-Day-3 run

### What is cut from the 5-day plan

| Dropped | Where it moves |
|---|---|
| Day 5 polish buffer | absorbed into Day 3 if time; else post-MVP |
| Full 6 fair-baseline guards (was already 3 in v2) | no change |
| Post-MVP skills (`add-decoder-template`, etc.) | never planned |
| Reviewer agent full wiring | never planned (dropped in v2 rescope) |
| 50-round "stress run" | deferred to post-hackathon |

## 13. Risks & Mitigations

| Risk | P | Mitigation |
|---|---|---|
| Day-2 integration misalignment | **High** | Explicit Day-1 EOD interface-sync meeting; orchestration and Runner contracts written to a shared file |
| Relay-BP not in `ldpc` package | Med | Fall back to reporting only BP+OSD (order 0+10); acknowledge in paper as a limitation |
| bb72 Stim circuit unavailable | Low | Three candidate sources (`qLDPC`, `stimbposd`, Bravyi github); last resort = 200-LOC hand-build (STRATEGIC §2 noted this is tractable) |
| Coder Tier-2 custom_fn failure rate too high | Med | Start with Tier-1 only; only enable Tier-2 Thu-Fri if plateau observed |
| Agent plateau at baseline with no gains | Med | Framing B defense: "autonomous convergence to SOTA boundary" is itself publishable. Worst case: fallback thesis in STRATEGIC §6 |
| One contributor unavailable | Med | Cross-training: core files should each have at least one backup reader/maintainer before Wed |
| Demo fails live | Low | Pre-record demo videos Thu night as insurance |
| Worktree disk pressure (§15) | Low | `MAX_ACTIVE_WORKTREES=3` + `WORKTREE_DISK_BUDGET_GB=5` checks; `git worktree prune` on orchestrator startup reaps crashed-run leftovers |
| Subprocess Runner launch overhead (§15.8) | Low | One subprocess per round (not per batch); adds ~1–2 s to 3–20 min training, negligible |
| Merge-conflict rate too high in compose rounds (§15.6) | Med | Ideator prompt penalises parent pairs already marked `FAILED_compose`; compose rounds bounded to ≤20% of total rounds per run |
| `pareto.json` and git branches drift apart | Med | `autoqec/orchestration/fork_graph.py` reconciles at round start; mismatch raises and pauses the run for human review |

## 14. Deferred to post-MVP (labeled, not forgotten)

All items in §3 N1-N13 are explicitly deferred. Tracked separately in GitHub Issues post-PR.

Priority ordering for v2.1 (month 1 after hackathon):
1. Third env with non-Pauli noise (leakage / biased)
2. Remaining 3 fair-baseline guards
3. AIDE solution-tree for Coder↔Runner inner loop
4. Reviewer subagent with meta-reflection

Priority ordering for v3 (month 3-6):
5. AI-Scientist-v2-style hypothesis tree search
6. Real-chip small-sample fine-tuning
7. paper-qa RAG over curated QEC corpus
8. Contributor authorship program (à la problem-reductions)

---

## 15. Worktree-based experiment model

This section supersedes the informal "each round writes into `runs/<id>/round_N/`" convention implied by §4.3 and §7. The two are compatible — `runs/` still holds training artefacts; worktrees add a parallel **code-level** record of each round as a named git branch.

### 15.1 Motivation

Research on neural predecoders is **not monotone-compositional**. Round 5's gated-MLP message function may give `Δ_LER = +4e-4`; round 12's attention aggregation may give `Δ_LER = +3e-4`; their merge may collapse to zero (or negative) because they address the same failure mode, or the two learned gates fight each other at inference. Auto-merging verified rounds into `main` therefore introduces a **drifting baseline** (rounds after the merge compare against main + A, not against the true baseline) and **hides mechanism-level conflicts** behind a single cumulative commit history.

Solution: **branches-as-Pareto**. `main` stays at the infrastructure baseline. Each round produces a named branch from an explicit `fork_from` parent. The Pareto front is a set of branch names, not a merged state. Composition is tested on-demand through explicit **compose rounds** that run `git merge parent-A parent-B` in a fresh worktree and measure the result.

The branch mechanics are ported near-verbatim from `problem-reductions/scripts/pipeline_worktree.py` (MIT, 520 LOC). That repo is the only autoresearch reference project that uses literal `git worktree`; AIDE / AI-Scientist v1-v2 / ml-master / dolphin / openevolve all use filesystem-directory sandboxes because their units-of-work are ML runs, not code commits. Our unit-of-work **is** a code commit (Coder edits `autoqec/decoders/modules/*.py` under Tier 2), which makes `git worktree` the natural mechanism.

### 15.2 Data model

```
main                                              # infrastructure baseline; never auto-updated by the research loop
│
├── exp/<run_id>/01-gnn-small                     (fork_from=baseline,      Δ_vs_baseline=+1e-5,  VERIFIED, on-Pareto)
├── exp/<run_id>/02-gated-mlp                     (fork_from=baseline,      Δ=+4e-4,              VERIFIED, on-Pareto)
├── exp/<run_id>/03-deep-gnn                      (fork_from=baseline,      Δ=-2e-4,              FAILED,   killed_by_safety)
├── exp/<run_id>/04-neural-bp                     (fork_from=baseline,      Δ=+3e-4,              VERIFIED, on-Pareto)
├── exp/<run_id>/07-stacked-attn                  (fork_from=02-gated-mlp,  Δ_vs_parent=+1e-4,    VERIFIED, on-Pareto)
└── exp/<run_id>/12-compose-02-04                 (fork_from=[02, 04],      status=compose_conflict)
```

Invariants:
- `main` advances only via traditional infrastructure PRs (Runner, DSL, env schema, contracts). The research loop **never** writes to `main`.
- A round's `delta_ler` has two distinct reportings. `delta_vs_parent` is measured under Runner's training-seed / val-shot regime and is used as a **search-guidance signal** for the Ideator. **Pareto admission and compose-round judgments use `delta_vs_baseline` computed by `/verify-decoder` on the canonical holdout bundle** (seeds `9000-9999`, `min_shots_verify`). Mixing the two is a category error; see §15.6 for the paired comparison protocol.
- FAILED branches are retained as named branches (commits are cheap) but their `.worktrees/` checkouts are removed after the round completes. Negative results remain queryable via `history.jsonl` + branch inspection.
- `pareto.json` is the **complete non-dominated archive** over `(delta_vs_baseline [higher better], flops_per_syndrome [lower], n_params [lower])`. No size cap — every VERIFIED branch that is not strictly dominated by another VERIFIED branch is a member. The top-5 "preview" (for Ideator L2 context and `/review-log` headlines) lives in `pareto_preview.json` as a derived view. `round_recorder.py`'s current top-5 sorted list is replaced by non-dominated filtering (see §15.9.1).
- **Identifier invariants split by row type.** Every `history.jsonl` / `pareto.json` row carries `round_attempt_id` (UUID minted at Ideator emit-time; always present, immutable). Rows backed by a committed worktree additionally carry `commit_sha` (canonical provenance key; immutable once set) and `branch` (human-readable alias; mutable). Non-commit terminal outcomes — `compose_conflict` — carry `round_attempt_id` only and set `commit_sha=null, branch=null`. A checkpoint is replayable iff `(commit_sha, env_yaml_sha256, dsl_config_sha256, requirements_fingerprint)` is intact (§15.5). A failed attempt is re-identifiable via `round_attempt_id` alone.
- `git branch --list 'exp/<run_id>/*'` is a view derived from `history.jsonl`; `fork_graph.py` reconciles the two at orchestrator startup and pauses the run if they disagree (branch without history row, or history row without branch).

### 15.3 Round lifecycle

Orchestrator runs from `main`'s working directory (read-only during a research run). Per round:

```
1. L2 snapshot       ← read history.jsonl + pareto.json + git branch --list 'exp/<run_id>/*'
                       and assemble fork_graph (§15.4)
2. Ideator subagent  → emits {hypothesis, fork_from, expected_delta, expected_cost_s, rationale}
                        where fork_from is one of:
                          "baseline"                  (new line of attack)
                          "exp/<run_id>/<N>-<slug>"   (stack on a prior VERIFIED branch)
                          ["exp/.../02-...", "exp/.../04-..."]  (compose round — triggers git merge;
                                                                 canonicalized as sorted list, §15.6.1)
3. worktree create   → autoqec.orchestration.worktree.create_round_worktree(run_id, round_idx, slug, fork_from)
                       creates .worktrees/exp-<run_id>-<N>-<slug>/ + branch exp/<run_id>/<N>-<slug>
                       for a compose round: create_compose_worktree runs `git merge parents[1:]`
                       after checking out parents[0]; on conflict → goto step 10a (compose_conflict path)
4. Coder subagent    → cwd = worktree_dir, writes DSL yaml and (for Tier 2) edits
                        autoqec/decoders/modules/*.py within the worktree
5. Commit code       → git -C <worktree> add && git commit -m "exp(<run_id>/<N>): <hypothesis>"
                       captures commit_sha (immutable provenance key)
6. Runner subprocess → subprocess_runner.run_round_in_subprocess(RunnerConfig(code_cwd=worktree_dir, ...))
                       writes runs/<id>/round_N/{metrics.json, checkpoint.pt, train.log}
                       writes worktree/round_N_pointer.json with commit_sha + env_yaml_sha256 +
                         dsl_config_sha256 + requirements_fingerprint + metrics summary + absolute artifact paths
                       commits the pointer file in the worktree (bumps commit_sha)
7. Analyst subagent  → reads runs/<id>/round_N/metrics.json → verdict candidate | ignore
8. /verify-decoder   → if verdict=candidate, runs independent_eval on holdout seeds 9000-9999
                       with env_spec.eval_protocol.min_shots_verify → VerifyReport
                       {verdict ∈ VERIFIED/SUSPICIOUS/FAILED, delta_vs_baseline_holdout,
                        commit_sha (required), branch}
9. Pareto update     → if VERIFIED and the candidate is non-dominated by any existing Pareto member,
                       append to pareto.json; evict any existing members now dominated by the new one.
                       Then regenerate pareto_preview.json (top-5 by -delta_vs_baseline) for L2 context.
10. Bookkeeping      → round_recorder writes history.jsonl row (always, regardless of step-3 outcome)
                       + log.md line. Every row carries `round_attempt_id` (UUID minted at Ideator
                       emit-time, before any git op). Rows backed by a real commit additionally
                       carry `commit_sha` and `branch`. Rows that ended before commit
                       (e.g. compose_conflict) carry `commit_sha=null, branch=null` but still have
                       `round_attempt_id` as the canonical row key.
10a. Compose_conflict → if step 3 hit a merge conflict: write history.jsonl with
                        status=compose_conflict, fork_from_canonical (sorted parents), conflicting_files,
                        round_attempt_id (required), commit_sha=null, branch=null.
                        Delete the synthetic branch (`git branch -D <branch>`) so it does NOT silently
                        point at parents[0]. The compose failure lives in history.jsonl only;
                        fork_graph.py surfaces it so the Ideator does not re-propose the same set.
11. Worktree cleanup → git worktree remove --force <worktree>
                       (for VERIFIED / FAILED / SUSPICIOUS: the branch persists and carries the commit;
                        for compose_conflict: the branch is already deleted in step 10a)
12. Next round       → loop back to step 1 with the new branch (if any) and the new history row
                       now visible in fork_graph
```

The orchestrator's cwd stays on `main` throughout. All writes land either in `.worktrees/` (source code, one commit per round) or `runs/<id>/round_N/` (training artefacts, shared across branches).

### 15.4 Fork-graph serialization (Ideator L3 context)

The Ideator sees the **full** fork graph — all rounds in the current run, including FAILED and dominated branches — because avoiding re-proposing an already-failed combination requires access to negative history.

```json
{
  "run_id": "20260422-140000",
  "fork_graph": {
    "nodes": [
      {"branch": "baseline",
       "delta_vs_baseline": 0.0, "ler": 5.0e-3, "flops": 0, "params": 0,
       "status": "baseline"},
      {"branch": "exp/20260422-140000/02-gated-mlp",
       "parent": "baseline", "delta_vs_parent": 4.0e-4, "delta_vs_baseline": 4.0e-4,
       "ler": 4.6e-3, "flops": 180000, "params": 42000,
       "status": "VERIFIED", "on_pareto": true,
       "hypothesis_1line": "gated MLP message, hidden=32"},
      {"branch": "exp/20260422-140000/03-deep-gnn",
       "parent": "baseline", "delta_vs_parent": -2.0e-4,
       "status": "FAILED", "failure_reason": "killed_by_safety: wall_clock_cutoff",
       "hypothesis_1line": "8-layer GNN with batchnorm"},
      {"branch": "exp/20260422-140000/12-compose-02-04",
       "parents": ["exp/20260422-140000/02-gated-mlp",
                    "exp/20260422-140000/04-neural-bp"],
       "status": "FAILED_compose",
       "failure_reason": "git merge conflict in autoqec/decoders/modules/gnn.py"}
    ],
    "pareto_front": [
      "exp/20260422-140000/02-gated-mlp",
      "exp/20260422-140000/04-neural-bp",
      "exp/20260422-140000/07-stacked-attn"
    ]
  }
}
```

Size estimate: 10–20 rounds × ~200 tokens/node ≈ 3–4k tokens; fits inside Ideator's 5k L3 budget without compression.

### 15.5 Worktree code scope

| Path | Location | Rationale |
|---|---|---|
| `autoqec/decoders/modules/*.py` | **in worktree** | Coder's Tier-2 editing target |
| `autoqec/decoders/dsl_compiler.py`, `custom_fn_validator.py` | **in worktree** | rare but possible Tier-2 touchpoints |
| `autoqec/example_db/*.yaml` | **in worktree** | new Tier-1 seeds are branch-local |
| `autoqec/runner/*`, `autoqec/orchestration/*`, `autoqec/eval/*`, `autoqec/envs/*`, `cli/`, `Makefile`, `pyproject.toml` | **main only** | infrastructure; updated via traditional PRs |
| `circuits/*.stim`, `autoqec/envs/builtin/*.yaml` | **main only** | environment definitions; experiments must not rewrite the env under test |
| `runs/<id>/round_N/checkpoint.pt / metrics.json / train.log` | **outside the worktree** (shared path) | binary/log artefacts stay external; every branch references them by absolute path via the pointer file below |
| `docs/`, `knowledge/` | **main only** | documentation is infrastructure |
| `.claude/agents/`, `.claude/skills/` | **main only** | agent definitions don't change per experiment |

`round_N_pointer.json` (the single file the worktree commits after Runner finishes). Provenance fields are **all required**; consumers reject pointer files missing any of them:

```json
{
  "run_id": "20260422-140000",
  "round_idx": 5,
  "branch": "exp/20260422-140000/05-gated-mlp",
  "commit_sha": "abc1234567890...",
  "fork_from": "baseline",
  "fork_from_canonical": "baseline",

  "provenance": {
    "env_yaml_sha256": "a1b2c3...",
    "dsl_config_sha256": "d4e5f6...",
    "requirements_fingerprint": "py3.10-torch2.5.0-stim1.13-pymatching2.2-ldpc2.1",
    "repo_root_resolved": "/abs/path/to/qec-ai-decoder"
  },

  "metrics_summary": {
    "delta_vs_parent": 4.0e-4,
    "flops_per_syndrome": 180000,
    "n_params": 42000,
    "status": "ok"
  },

  "artifact_paths": {
    "checkpoint": "/abs/path/runs/20260422-140000/round_05/checkpoint.pt",
    "metrics":    "/abs/path/runs/20260422-140000/round_05/metrics.json",
    "train_log":  "/abs/path/runs/20260422-140000/round_05/train.log"
  }
}
```

Artifact paths are **absolute** to survive relocation of the orchestrator's cwd; `repo_root_resolved` gives a relative-path reconstruction basis if the run is moved to a new machine. `requirements_fingerprint` is a short string derived from the frozen dependency set (`pip freeze | sha256 | head`); post-MVP will upgrade to a full lockfile digest.

`.gitignore` adds `.worktrees/`.

### 15.6 Compose rounds

Ideator requests a compose round by emitting `fork_from: ["exp/.../A", "exp/.../B", ...]` (a list of two or more parent branches).

#### 15.6.1 Fork-from canonicalization

`fork_from` is treated as an **unordered set** of parents for dedup purposes — `[A, B]` and `[B, A]` are the same experiment. However, `git merge` is order-sensitive (the first parent's tree becomes the merge base). Resolution:

- `fork_from_canonical` is the alphabetically-sorted join of branch names (e.g. `"02-gated-mlp|04-neural-bp"`). This is the field the Ideator-response parser writes into `history.jsonl` and the dedup key `fork_graph.py` uses when checking "already proposed".
- `fork_from_ordered` preserves the Ideator's emitted order and drives the `git worktree add <wt> -b <branch> parents[0]` + `git merge parents[1:]` sequence, because re-running with a different base parent can trivially succeed where the original failed (different conflict markers). Both fields are persisted.
- The Ideator is told in its prompt: "propose each parent set at most once; the canonicalized set is what the system deduplicates against."

#### 15.6.2 Compose round flavors (distinct status values)

A compose round has two flavors, which are **not interchangeable scientifically**:

- **`compose_pure`**: Coder makes zero edits; the branch's tree is exactly the `git merge` result. Tests the hypothesis "do A and B's effects compose?"
- **`compose_with_edit`**: Coder edits the merged tree (e.g. to fix API breakage introduced by merging two files with overlapping interfaces). Tests the hypothesis "with a minimal manual reconciliation, do A and B's effects compose?"

The Ideator's `fork_from` response must include `compose_mode: "pure" | "with_edit"`. `RoundMetrics.status` ∈ {`ok`, `compose_conflict`, `compile_error`, `train_error`, `killed_by_safety`}; the flavor is recorded in a separate `compose_mode` field and carried into Pareto / fork graph. Mixing pure and with_edit results in composition claims is explicitly disallowed.

#### 15.6.3 Conflict path and durable record

On `git merge` conflict (step 3 of §15.3 lifecycle):

1. `git merge --abort` (restores worktree to parents[0] state — do **not** commit anything).
2. `git worktree remove --force <wt>`.
3. `git branch -D <branch>` — the synthetic branch is deleted so it does not linger pointing at `parents[0]`, silently indistinguishable from a real unedited child.
4. `round_recorder.append_round(...)` writes a `history.jsonl` row **before** any further cleanup:

```json
{"round_idx": 12,
 "round_attempt_id": "8b4f2c1e-...",
 "status": "compose_conflict",
 "fork_from": ["exp/.../02-gated-mlp", "exp/.../04-neural-bp"],
 "fork_from_canonical": "exp/.../02-gated-mlp|exp/.../04-neural-bp",
 "compose_mode": "pure",
 "conflicting_files": ["autoqec/decoders/modules/gnn.py"],
 "timestamp": "2026-04-22T14:37:22Z",
 "commit_sha": null,
 "branch": null}
```

Per §15.2 identifier invariants: `round_attempt_id` is the row key; `commit_sha` and `branch` are null because no commit was made. `fork_graph.py` surfaces this row as a `FAILED_compose` node so the Ideator's next-round prompt knows the canonical parent set has been tried and failed. There is no residual git state to clean up later.

#### 15.6.4 Paired comparison protocol (mandatory for any composition claim)

Because training is seed-driven and `delta_vs_parent` carries training-variance noise, making composition claims requires **all three branches — parent A, parent B, and the composed child — to be re-evaluated on the same canonical holdout bundle** before any conclusion is drawn:

- Same holdout seed range (`env_spec.noise.seed_policy.holdout`, typically 9000–9999).
- Same shot count (`env_spec.eval_protocol.min_shots_verify`).
- Same classical backend + baseline settings (from `env_spec`).
- Same evaluation harness (`/verify-decoder`; no re-training).

`/verify-decoder` records a `paired_eval_bundle_id` in each `VerifyReport`; a compose round's VerifyReport must cite the same `paired_eval_bundle_id` as its parents' most recent verifications, or the Ideator triggers a re-verify of the parents before drawing conclusions. Composition claims that use `delta_vs_parent` alone (training-time number) are rejected at `/verify-decoder` gate.

#### 15.6.5 Outcome interpretation (with paired bundle in hand)

Given paired `delta_vs_baseline_holdout` values `Δ_A`, `Δ_B`, `Δ_child`:

- `Δ_child > 0.8 × (Δ_A + Δ_B)` → composition largely holds; child may dominate either parent on Pareto.
- `0 < Δ_child ≤ 0.8 × (Δ_A + Δ_B)` → partial interaction; child joins Pareto without dominating.
- `Δ_child ≤ 0` or `status=FAILED` → **negative result with scientific value**; recorded as evidence that A and B's mechanisms conflict.

The `0.8` threshold is heuristic and lives in `autoqec/pareto/` config for easy tuning. What is not heuristic: the paired comparison protocol (§15.6.4) must be satisfied before any of these three branches is cited in `/review-log` narrative.

### 15.7 Contract schema delta

Not uniformly additive. Fields split into **required when worktree is active** (reject without them) versus **pure optional add**.

```python
# autoqec/runner/schema.py — RunnerConfig additions
class RunnerConfig(BaseModel):
    # ... existing fields unchanged ...
    code_cwd:            Optional[str] = None          # absolute path to worktree checkout; None = in-process
    branch:              Optional[str] = None          # exp/<run_id>/<N>-<slug>; required when code_cwd is set
    fork_from:           Optional[Union[str, list[str]]] = None  # list → compose round
    fork_from_canonical: Optional[str] = None          # sorted, |-joined; dedup key for compose
    compose_mode:        Optional[Literal["pure", "with_edit"]] = None  # required when fork_from is list

    @model_validator(mode="after")
    def _worktree_fields_consistent(self):
        if self.code_cwd is not None and self.branch is None:
            raise ValueError("branch is required when code_cwd is set")
        if isinstance(self.fork_from, list) and self.compose_mode is None:
            raise ValueError("compose_mode is required for compose rounds")
        return self

# autoqec/runner/schema.py — RoundMetrics additions
class RoundMetrics(BaseModel):
    # ... existing fields unchanged ...
    round_attempt_id:    Optional[str] = None                          # UUID; REQUIRED on worktree path (validator)
    branch:              Optional[str] = None                          # None for legacy in-process path + compose_conflict
    commit_sha:          Optional[str] = None                          # REQUIRED when branch is set (validator)
    fork_from:           Optional[Union[str, list[str]]] = None
    fork_from_canonical: Optional[str] = None
    compose_mode:        Optional[Literal["pure", "with_edit"]] = None
    delta_vs_parent:     Optional[float] = None                        # training-regime Δ; search-guidance only
    parent_ler:          Optional[float] = None
    conflicting_files:   Optional[list[str]] = None                    # set iff status=compose_conflict
    train_seed:          Optional[int] = None                          # seed actually used; for Pareto disambiguation
    # Existing `status` Literal gains one new variant:
    status: Literal["ok", "killed_by_safety", "compile_error", "train_error", "compose_conflict"]

    @model_validator(mode="after")
    def _provenance_integrity(self):
        # round_attempt_id is required whenever the worktree path is active (fork_from set or branch set).
        if (self.branch is not None or self.fork_from is not None) and self.round_attempt_id is None:
            raise ValueError("round_attempt_id is required on the worktree path")
        # commit_sha paired with branch for commit-backed rounds
        if self.branch is not None and self.commit_sha is None:
            raise ValueError("commit_sha is required whenever branch is set")
        # compose_conflict must have neither branch nor commit_sha
        if self.status == "compose_conflict" and (self.branch is not None or self.commit_sha is not None):
            raise ValueError("compose_conflict rows must have branch=None and commit_sha=None")
        return self

# autoqec/eval/schema.py — VerifyReport additions
class VerifyReport(BaseModel):
    # ... existing fields unchanged ...
    branch:                  Optional[str] = None        # None only for legacy non-worktree runs
    commit_sha:              Optional[str] = None        # REQUIRED when branch is set
    delta_vs_baseline_holdout: Optional[float] = None    # paired-bundle canonical delta (§15.6.4)
    paired_eval_bundle_id:   Optional[str] = None        # REQUIRED for compose rounds and any Pareto candidate

    @model_validator(mode="after")
    def _holdout_bundle_required_for_compose(self):
        # Compose round VerifyReports must carry paired_eval_bundle_id so §15.6.4
        # cross-checks against parents. Non-compose rounds may omit it for the
        # search-guidance path but must provide it when admitted to pareto.json.
        return self
```

`agents/schemas.py` — subagent response schemas. The current models use `extra="forbid"`; they must flip to `extra="allow"` **or** explicitly declare the new fields:

```python
# autoqec/agents/schemas.py
class IdeatorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis: str
    expected_delta_ler: float
    expected_cost_s: int
    rationale: str
    dsl_hint: Optional[dict] = None
    fork_from: Union[Literal["baseline"], str, list[str]]   # NEW — required field
    compose_mode: Optional[Literal["pure", "with_edit"]] = None  # NEW — required when fork_from is list

class CoderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # ... existing fields ...
    commit_message: str   # NEW — used for `git commit -m` in step 5 of §15.3
```

`pareto.json` stores the **complete non-dominated set** of VERIFIED branches. Row schema:

```json
{"commit_sha":  "abc12345...",     // canonical provenance key; REQUIRED
 "branch":      "exp/.../02-gated-mlp",  // human-readable alias; mutable
 "delta_vs_baseline_holdout": 4.0e-4,    // from /verify-decoder paired bundle
 "paired_eval_bundle_id":     "bundle-20260422-1500",
 "flops_per_syndrome":        180000,
 "n_params":                  42000,
 "verdict":                   "VERIFIED",
 "fork_from":                 "baseline",
 "fork_from_canonical":       "baseline",
 "compose_mode":              null}
```

`pareto_preview.json` is a derived L2 view (top-5 by `-delta_vs_baseline_holdout`) regenerated after every Pareto mutation. Readers that consume only the preview must not claim to report the full archive.

### 15.8 Runner invocation paths

| `code_cwd` setting | Execution | Rationale |
|---|---|---|
| `None` (default) | `autoqec.runner.runner.run_round(cfg, env)` in-process, as today | Covers every non-worktree path: Demo 2, `cli/autoqec.py run --no-llm`, `scripts/run_quick.py`, all existing tests. Zero behavioural change. |
| absolute path | `autoqec.orchestration.subprocess_runner.run_round_in_subprocess(cfg)` shells `python -m cli.autoqec run-round --code-cwd <path> …` with `cwd=code_cwd` and `PYTHONPATH=code_cwd:$PYTHONPATH` | Python's import cache does not hot-reload edited modules; a subprocess is the only reliable way to pick up worktree-local edits to `autoqec/decoders/modules/*.py`. |

The `run_round` function itself raises `RunnerCallPathError` if called in-process with a non-None `code_cwd`, making the routing mistake impossible to hide.

### 15.9 Impact on existing code (as of main @ `bde19db`)

Split into **hard API changes** (existing clients break without coordinated updates) versus **pure additive** (existing clients keep working because new fields default off).

#### 15.9.1 Hard API changes (break without coordinated updates)

| File | Break | Why |
|---|---|---|
| `autoqec/orchestration/memory.py` | `l3_for_ideator` return payload changes shape from `{env_spec, pareto_front, last_5_hypotheses, knowledge_excerpts, machine_state_hint}` to the fork_graph of §15.4 | Ideator prompt consumers on `main` read `last_5_hypotheses` by name; fork_graph replaces that field, not extends. `test_orchestration_stub.py` and `test_loop_helpers.py` assert the old shape. |
| `autoqec/orchestration/round_recorder.py` | `_PARETO_FIELDS` is replaced; the whole Pareto-update algorithm changes from **"top-5 sorted by `-delta_ler, flops, n_params`"** to **"complete non-dominated set over `(delta_vs_baseline_holdout [higher], flops [lower], n_params [lower])` with the top-5 preview regenerated as `pareto_preview.json`"**. Current sort loses valid low-FLOPs / low-param tradeoff points. | Changes the scientific semantics of `pareto.json`. `test_round_recorder.py` tests the old top-5 assumption. |
| `autoqec/agents/schemas.py` | All three response models today use `ConfigDict(extra="forbid")`; new fields `fork_from`, `compose_mode`, `commit_message` must be added as declared fields (flipping to `extra="allow"` hides contract drift, so declare explicitly). | pydantic with `extra="forbid"` rejects unknown keys; existing JSON responses without the new fields will be rejected once `fork_from` is declared required. |
| `cli/autoqec.py run-round` | Currently positional-only (`env_yaml config_yaml round_dir`); the worktree path adds keyword flags (`--code-cwd`, `--branch`, `--fork-from`, `--compose-mode`). Positional callers keep working; invocations on `main` that hard-code the three positional args continue but miss the new fields. | Any new LLM-driven orchestration call site must migrate from positional to keyword form. Not a silent break, but a migration task. |
| `autoqec/runner/runner.py` | `run_round(cfg, env, safety)` now raises `RunnerCallPathError` when `cfg.code_cwd is not None`. Existing tests that call `run_round` with the old 5-field `RunnerConfig` continue; any new caller passing `code_cwd` in-process hits the error. | Documented routing constraint; not a silent break. |

#### 15.9.2 Pure additive changes (existing callers untouched)

| File | Edit |
|---|---|
| `autoqec/runner/schema.py` | §15.7 field additions (`code_cwd`, `branch`, `fork_from`, `fork_from_canonical`, `compose_mode` on `RunnerConfig`; `branch`, `commit_sha`, `fork_from*`, `compose_mode`, `delta_vs_parent`, `parent_ler`, `conflicting_files` on `RoundMetrics`); new `status` Literal variant `compose_conflict`. All additions default to `None`. |
| `autoqec/eval/schema.py` | `VerifyReport` gains `branch`, `commit_sha`, `delta_vs_baseline_holdout`, `paired_eval_bundle_id` — all default `None`. |
| `autoqec/tools/machine_state.py` | adds `active_worktrees: list[str]` (empty list when no worktree module is loaded). |
| `autoqec/orchestration/loop.py` | `run_round_plan` gains `fork_from` parameter (defaulted to `"baseline"`); Coder ctx carries `worktree_dir` (defaulted to `None`). Callers passing neither keep working. |
| `scripts/run_single_round.py` | accepts `--fork-from` (defaults to `"baseline"`), creates worktree, passes `code_cwd`. Scripts calling the old 3-arg form on `main` keep working via defaults. |
| `.claude/skills/autoqec-run/SKILL.md` | adds fork-from decision + compose-round subflow + `compose_conflict` handling. |
| `.claude/agents/autoqec-ideator.md` | consumes fork_graph; emits `fork_from` + `compose_mode`. Prompt rewrite, not a schema break on the autoqec-ideator side (the .md is just a prompt string). |
| `.claude/agents/autoqec-coder.md` | note: cwd is a worktree; code commits are auto-triggered post-Runner. |
| `.claude/agents/autoqec-analyst.md` | output carries `branch` + `commit_sha`. |
| `docs/contracts/interfaces.md`, `docs/contracts/round_dir_layout.md` | contract bumps — merge under `contract-change` label with 3-of-3 owner sign-off. |

#### 15.9.3 Migration order (required sequencing for CI to stay green)

Land in this order on `feat/worktree-experiment-model`:

1. **pydantic schemas** — `autoqec/runner/schema.py`, `autoqec/eval/schema.py`, `autoqec/agents/schemas.py`. Only add fields; no callers change yet. CI stays green because added fields default to None.
2. **round_recorder Pareto algorithm** — replace top-5 sort with non-dominated filter + derived `pareto_preview.json`. Update `test_round_recorder.py` in the same commit.
3. **CLI flags** — `cli/autoqec.py run-round` gains `--code-cwd` etc. Existing positional callers untouched.
4. **Orchestration prompt shape** — `memory.py` returns fork_graph; `loop.py` threads `fork_from`; `agents/dispatch.py` updates prompt templates. Update `test_orchestration_stub.py`, `test_loop_helpers.py`, `test_run_single_round.py`, `test_e2e_handshake.py` in the same commit.
5. **Worktree / fork_graph / subprocess_runner modules** — the three new orchestration modules come online.
6. **Runner in-process guard** — `run_round` starts raising on `code_cwd is not None`.
7. **Agent markdown + skill updates** — `.claude/agents/*.md` and `.claude/skills/autoqec-run/SKILL.md`.
8. **Contracts** — `docs/contracts/*.md` land last with the full-picture diff for 3-of-3 review.

Tests gated by each step must pass before the next step lands.

**New files**:

| New file | Purpose | Est. LOC |
|---|---|---|
| `autoqec/orchestration/worktree.py` | ported from `problem-reductions/scripts/pipeline_worktree.py` with field renames (issue → round) + `create_compose_worktree` | ~220 |
| `autoqec/orchestration/fork_graph.py` | assembles §15.4 JSON from `git branch --list` + `history.jsonl` + `pareto.json` | ~150 |
| `autoqec/orchestration/subprocess_runner.py` | `run_round_in_subprocess(cfg) → RoundMetrics` — shells `cli.autoqec run-round`, captures JSON, validates | ~80 |
| `tests/test_worktree.py` | temp git repo; covers create / compose / conflict / cleanup | ~150 |
| `tests/test_fork_graph.py` | tree serialization, empty-run fallback, Pareto field presence | ~80 |
| `tests/test_subprocess_runner.py` | dev-profile round in a worktree using `gnn_small.yaml` | ~60 |

**Untouched** (explicit list, for reviewer confidence):

- `autoqec/runner/runner.py`'s training / evaluation algorithm
- `autoqec/decoders/dsl_schema.py`, `dsl_compiler.py`, `custom_fn_validator.py`, `backend_adapter.py`, `baselines/pymatching_wrap.py`, all of `modules/*.py`
- `autoqec/envs/schema.py` and all `envs/builtin/*.yaml`
- `autoqec/example_db/*.yaml`
- `cli/autoqec.py run` (no-LLM loop) and `cli/autoqec.py add-env`
- `scripts/run_quick.py`, `scripts/benchmark_surface_baseline.py`, `scripts/e2e_handshake.py`, `scripts/generate_surface_circuit.py`, `scripts/scout_bb72.py`
- `demos/demo-2-bb72/run.sh` and its snapshot
- `.claude/skills/add-env/SKILL.md`
- Existing tests that don't set `code_cwd`: 12 files (`test_dsl_*`, `test_gnn_*`, `test_neural_bp_*`, `test_runner_smoke`, `test_runner_safety`, `test_pymatching_baseline`, `test_seed_templates`, `test_surface_assets`, `test_backend_adapter`, `test_custom_fn_validator`, `test_machine_state`, `test_surface_baseline_benchmark`)

**Existing tests requiring updates** (because the memory / loop / recorder schema bump):

- `tests/test_orchestration_stub.py`, `tests/test_round_recorder.py`, `tests/test_loop_helpers.py`, `tests/test_run_single_round.py`, `tests/test_run_quick.py`, `tests/test_e2e_handshake.py` — each expects ~15 LOC of maintenance to accept the new fork_graph / `_PARETO_FIELDS` shape.

### 15.10 Safety, cleanup, and crash recovery

**Runtime safety**:

- `RunnerSafety.WALL_CLOCK_HARD_CUTOFF_S` governs training as before; an additional `WORKTREE_DISK_BUDGET_GB = 5` (implemented in `worktree.py`) checks available disk before every `create_round_worktree` and raises if exceeded.
- Simultaneous worktrees on disk are bounded to `MAX_ACTIVE_WORKTREES = 3` for MVP (current round + at most one compose scratch + one buffer); exceeding this raises and the round is logged as `killed_by_safety`.
- After every round, `git worktree remove --force <path>` frees the checkout. Branches persist (tiny on-disk cost: just the commit object graph).

**Startup reconciliation** (`fork_graph.py` runs at every orchestrator start before the first round). Must distinguish "empty synthetic branch from a crash before code commit" from "committed round whose history row never made it to disk" — the two demand opposite actions:

1. List `git branch --list 'exp/<run_id>/*'` → set `B`.
2. Parse `history.jsonl` → set `H = {row["branch"] for row in rows if row.get("branch")}`.
3. For each `b ∈ B \ H` (branch without history row), inspect git state to classify:
   a. **Empty synthetic branch** — `git rev-list --count <b> ^origin/main` reports 0, no pointer file on HEAD, no artifacts under `runs/<id>/round_N/` matching the branch slug. The lifecycle crashed between step 3 (worktree create) and step 5 (first commit). Action: `git branch -D <b>` + `git worktree prune`. Logged as `reaped: <branch> (empty)`.
   b. **Committed round without history** — ≥1 commit on the branch or a pointer file exists or artifacts are on disk. The lifecycle crashed between step 5 and step 10. Action: **quarantine, do not delete**. Rename to `quarantine/<run_id>/<N>-<slug>`, write a `history.jsonl` row with `status=orphaned_branch`, `round_attempt_id=<newly minted UUID>`, and links to the available commit_sha + pointer file + artifacts. Operator reviews whether to manually promote or drop.
4. For each `h ∈ H \ B` (history row with non-null `branch` but no live branch): the branch was manually deleted after recording. Action: append a `status_reason="branch_manually_deleted"` follow-up row referencing the original `round_attempt_id`; do not touch the existing history row.
5. For every Pareto member, verify the commit is still retrievable via `git rev-parse --verify <commit_sha>^{commit}` (reachable object check, not reflog — reflogs expire). A missing commit pauses the run for human review; do not silently drop from Pareto.

**Branch naming collisions**:

- The branch name `exp/<run_id>/<N>-<slug>` is deterministic from `(run_id, round_idx, slug)`. `run_id` is a timestamp (`YYYYMMDD-HHMMSS`) minted once per orchestrator startup, so `<N>-<slug>` only collides within a single run.
- Within a run, `round_idx` is strictly monotonic. The `slug` derives from the Ideator's hypothesis and is sanitized via `re.sub(r"[^A-Za-z0-9]+", "-", slug).lower()[:40]` (matches the `sanitize_component` helper ported from `problem-reductions/scripts/pipeline_worktree.py`).
- On a genuine collision (same round_idx already a branch, which indicates state corruption or a restart race): the orchestrator refuses to create a new branch and halts for human review, rather than silently overwriting.

**Stale worktrees from crashed runs**: reaped at startup via `git worktree prune` (run before step 1 of reconciliation above), then the filesystem-level `.worktrees/` directory is scanned for orphaned subdirectories whose corresponding branches/worktree records are gone; those are `shutil.rmtree`'d.

### 15.11 MVP scope and post-MVP

**In MVP** (this PR's scope):
- Serial execution (`MAX_PARALLEL_ROUNDS = 1`) — one research round at a time; worktrees give branch isolation and history, not throughput.
- Compose rounds with `conflict = FAIL` handling (compose_pure and compose_with_edit flavors).
- Full fork graph serialization to the Ideator (no pruning).
- Pydantic schema fields land as `Optional` to sequence the 8-step rollout (§15.9.3) — but worktree mode itself is **not** backward-compatible. It introduces the hard API changes listed in §15.9.1 (memory.py payload shape, round_recorder Pareto algorithm, agents/schemas.py required fields under `extra="forbid"`, CLI keyword migration).
- Fixed training-seed regime per round (one model per branch, seed derived from `(run_id, round_idx)`). Training-seed variance across branches is acknowledged as a systematic limitation; post-MVP adds multi-seed retraining for Pareto members (§15.11 post-MVP below).

**Deferred to post-MVP**:
- `MAX_PARALLEL_ROUNDS > 1`: concurrent worktrees + GPU scheduling. Requires adding `asyncio.gather` inside the round dispatcher and a lightweight GPU reservation queue.
- LLM-assisted merge-conflict resolution: when `compose_conflict` fires, dispatch Coder with both parent diffs to author a manual reconciliation. Currently the conflict is terminal.
- Tier-1-only short-circuit: detect when Coder made **zero edits outside `autoqec/example_db/*.yaml`** and skip the subprocess, running Runner in-process for a 5–10× speedup. The policy check is one-line, the risk is a missed code edit silently reverting to main's copy — hold until tests prove the detection is reliable.
- Automatic **canonical champion** promotion: after a run completes, identify the Pareto-best branch, run it through a multi-env recertification, and open a human-review PR to `main`.
- **Multi-seed retraining for Pareto members**: MVP trains one model per branch with a deterministic seed. Post-MVP adds a "re-seed" round that re-trains a VERIFIED Pareto member under seeds `{1001, 2001, 3001}` and reports `delta_vs_baseline_holdout` median + IQR so training-seed variance stops being a hidden confound in composition claims.

---

## Appendix A — Tier-1 DSL complete schema

```yaml
# autoqec/decoders/dsl_schema.yaml — pydantic-validated
predecoder:
  type:
    enum: [gnn, neural_bp]
    required: true

  output_mode:
    enum: [hard_flip, soft_priors]
    required: true

  # GNN family (only when type=gnn)
  gnn:
    layers:        {type: int, min: 1}
    hidden_dim:    {type: int, min: 4}
    message_fn:
      anyOf:
        - enum: [mlp, gated_mlp, attention, gru_cell, edge_attention,
                 geometric_attention, residual_mlp, normalized_mlp]
        - type: custom_fn
          signature: "(x_src, x_dst, e_ij) -> Tensor[hidden]"
    aggregation:
      anyOf:
        - enum: [sum, mean, max, attention_pool, set_transformer, gated_sum]
        - type: custom_fn
          signature: "(messages, edge_index) -> Tensor[n_nodes, hidden]"
    normalization: {enum: [none, layer, batch, edge_norm, graph_norm]}
    residual:      {type: bool}
    edge_features: {type: list, items: [syndrome_bit, round_idx,
                                         stabilizer_type, distance,
                                         prior_weight]}

  # Neural-BP family (only when type=neural_bp)
  neural_bp:
    iterations:    {type: int, min: 1}
    weight_sharing: {enum: [none, per_layer, per_check]}
    damping:       {enum: [fixed, learnable_scalar, learnable_per_iter]}
    attention_aug: {type: bool}
    attention_heads: {type: int, min: 1}

  head:
    anyOf:
      - enum: [linear, mlp_small]
      - type: custom_fn
        signature: "(hidden_state) -> Tensor[output_dim]"

  training:
    learning_rate: {type: float, min: 0}
    batch_size:    {type: int, min: 1}
    epochs:        {type: int, min: 1}
    loss:          {enum: [bce, focal, weighted_bce]}
    profile:       {enum: [dev, prod]}
```

## Appendix B — Subagent prompt skeletons

Full prompts in `.claude/agents/autoqec-*.md`. Abbreviated structure here:

```
autoqec-ideator.md (~300 tokens)
  You are the Ideator in AutoQEC's multi-round research loop.
  Input: env_spec, pareto_front, last_5_hypotheses, knowledge_excerpts.
  Tool: machine_state (CALL FIRST — use it to estimate compute budget).
  Output: JSON {hypothesis, expected_delta_ler, expected_cost_s, rationale}.
  Constraints: no re-proposing past hypotheses; respect budget.

autoqec-coder.md (~250 tokens)
  You are the Coder. Emit valid Tier-1 DSL YAML for the given hypothesis.
  Escalate to Tier-2 custom_fn only when Tier-1 provably cannot express.
  Tools: Read, Write, Edit. No Bash.

autoqec-analyst.md (~200 tokens)
  You are the Analyst. Read metrics.json; write a 3-sentence round report
  noting Δ_LER, key trade-offs, and delta from previous round.
  Tools: Read. No modification of anything.
```

## Appendix C — Key changes from v1

| v1 | v2 | Reason |
|---|---|---|
| Full multi-agent DAG with Reviewer | 3 subagents (Reviewer removed) | 1-week budget |
| DSL + PyTorch free-form C-path | Tier-1 rich DSL + Tier-2 custom_fn | 42% failure rate for C-path |
| Tree-search upgrades scheduled | All deferred post-MVP | 1-week budget |
| Two reference envs as equal MVP | `surface_d5` core + `bb72` stretch | 1-week budget |
| Inline + background both MVP | Inline primary, background supported | Codex-cli on server = background |
| Hard DSL limits (max_params, max_layers) | No hard limits; `machine_state` tool + RunnerSafety | Agent self-awareness |
| 6 fair-baseline guards | 3 MVP + 3 deferred | 1-week budget |
| claude-cli backend primary | codex-cli backend primary | Server deployment preference |
| Contributor authorship / add-decoder-template skill | Deferred | 1-week budget |
| Baseline: pymatching + bposd | pymatching + bposd + **relay_bp** | DECODER_ROADMAP identifies Relay-BP as must-beat |
| Framing A ("SOTA-competitive") | Framing B ("Pareto discovery across triples") | STRATEGIC_ASSESSMENT compute-gap analysis |
| 6 developer skills | 5 user-facing skills | Project deliverable requirement + "only LLM-reasoning tasks are skills" philosophy |
| Makefile minimal | Makefile with per-agent backend/model overrides | Ablation-friendly |
| 5-day delivery plan | **3-day delivery timeline (v2.1)** | Project reality |
| Skills/UX work distributed throughout the week | **Core infrastructure front-loaded into Days 1-2; skills moved to Day-3 wrappers** | Front-load novelty-carrying work before packaging |
| Skills written throughout the week | **Skills = Day-3 thin wrappers over working CLIs** | CLIs first, UX second |
| All 5 demos equal priority | **Demos tagged P0/P1/P2** (1+4 must-ship; 2 must-attempt; 3+5 nice-to-have) | 3-day realism |
