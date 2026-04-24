# CLAUDE.md

Developer guidance for Claude Code (claude.ai/code) when working on this repository.
For project overview, team ownership, and deliverable roster, see `README.md`.
For the authoritative design, see `docs/superpowers/specs/2026-04-20-autoqec-design.md`.

## Build & Test

```bash
./.venv/bin/pip install -e '.[dev]'      # install in editable mode
./.venv/bin/pytest tests/ -m "not integration" -v    # unit tests (CPU)
./.venv/bin/pytest tests/ -m "not integration" -v --run-slow  # include slow subprocess-heavy tests
./.venv/bin/pytest tests/ -m integration --run-integration --model-path ...  # GPU tests
./.venv/bin/pytest tests/test_round_recorder.py -v   # one file
./.venv/bin/pytest tests/ -k "pareto"                # by substring
./.venv/bin/ruff check autoqec cli tests scripts     # lint
```

Tests marked `@pytest.mark.slow` are skipped unless `--run-slow` is passed,
even if you explicitly invoke `pytest -m "not integration"`.

Python ≥ 3.12, pydantic v2, torch ≥ 2.11, stim ≥ 1.15, pymatching ≥ 2.3, ldpc ≥ 2.4.

Agent backends are selected per-role via env vars: `AUTOQEC_{IDEATOR,CODER,ANALYST}_BACKEND ∈ {codex-cli, claude-cli}` and `..._MODEL`. Defaults: Ideator=gpt-5.4 (codex), Coder=gpt-5.4-codex, Analyst=claude-haiku-4-5. Swap via `make run AUTOQEC_IDEATOR_BACKEND=claude-cli`.

## Architecture

### Core Loop

`cli/autoqec.py run ENV` → `RunMemory(run_dir)` → per round: Ideator (hypothesis) → Coder (DSL config) → Runner (train + classical-backend eval) → Analyst (one-line verdict) → optional Verifier (holdout) → `record_round(...)` → update `pareto.json` + `fork_graph.json`. Each round optionally pins to its own `exp/<run_id>/<NN>-<slug>` git branch in `.worktrees/<id>/`.

### Entry Points

| Entry | Purpose |
|---|---|
| `cli/autoqec.py run ENV` | Full research loop (N rounds, live LLMs) — `click` CLI |
| `cli/autoqec.py run-round ENV CFG ROUND_DIR` | Single round via RunnerConfig (supports `--code-cwd/--branch/--fork-from/--compose-mode/--round-attempt-id`) |
| `cli/autoqec.py add-env PATH` | Validate and register a new environment YAML |
| `scripts/run_single_round.py` | Bypass orchestrator; run one round directly |
| `scripts/run_quick.py` | No-LLM smoke path (useful for CI) |
| `scripts/e2e_handshake.py` | Cross-component handshake smoke test |
| `scripts/benchmark_surface_baseline.py` | Surface-code baseline sanity |

### Subpackages

| Module | Purpose |
|---|---|
| `autoqec.agents` | `dispatch.build_prompt(role, ctx)` assembles subagent prompts. `schemas.py` — `IdeatorResponse`/`CoderResponse`/`AnalystResponse` pydantic models. |
| `autoqec.decoders` | `dsl_schema.py` — `PredecoderDSL` pydantic model (Tier 1 canonical). `dsl_compiler.py` — DSL → nn.Module. `custom_fn_validator.py` + `custom_fn_rules.py` — Tier-2 escape hatch AST + smoke guard. `backend_adapter.py` — MWPM/OSD classical backend glue. `baselines/`, `modules/` — reference predecoders. |
| `autoqec.envs` | `schema.py` — `EnvSpec` (code + noise + constraints triple). `builtin/` — `surface_d5_depol.yaml`, `bb72_depol.yaml`. |
| `autoqec.eval` | `schema.py` — `VerifyReport` (holdout verdict, `delta_vs_baseline_holdout`, `paired_eval_bundle_id`, `branch`, `commit_sha`). |
| `autoqec.orchestration` | Research-loop driver. `memory.py` — `RunMemory` (L1/L2/L3). `loop.py` — `run_round_plan` assembles prompts. `round_recorder.py` — `record_round` + non-dominated Pareto filter. `worktree.py` — git-worktree CRUD. `subprocess_runner.py` — shell-out Runner for worktree mode. `fork_graph.py` — `build_fork_graph(history, pareto, run_id)` assembles the Ideator's branch-aware view. `reconcile.py` — startup `git branches ↔ history.jsonl` reconciliation per §15.10. |
| `autoqec.runner` | `runner.py` — `run_round(cfg)` train + eval entrypoint. `schema.py` — `RunnerConfig` + `RoundMetrics` pydantic models (with §15.7 worktree fields). `data.py` — syndrome sampling. `flops.py` — fvcore-based FLOP counting. `safety.py` — tool whitelisting. |
| `autoqec.tools` | `machine_state.py` — LLM-callable tool returning GPU VRAM / recent-round timings / active worktrees. |

### Key Design Decisions

- **Two-stage decoder**: AI predecoder (trainable GNN / neural-BP) → fixed classical backend (MWPM for surface codes, OSD for qLDPC). The classical backend guarantees structural validity; the predecoder contributes a single `Δ_LER = LER(plain) − LER(predecoder + classical)` number per round.
- **DSL tiers**: Tier 1 is a canonical YAML (`PredecoderDSL` pydantic model — type ∈ {gnn, neural_bp}, fixed hyperparameters). Tier 2 is `custom_fn` — raw Python source validated by AST whitelist (`ALLOWED_TOP_IMPORTS`, `ALLOWED_FROM_IMPORTS`, `FORBIDDEN_NAMES`, `SLOT_SIGNATURES` in `custom_fn_rules.py`) plus a smoke-test execution. Tier 2 is the escape hatch for novel architectures; Tier 1 is the default.
- **3-subagent DAG**: Ideator (proposes hypothesis + `fork_from` + `compose_mode`) → Coder (emits DSL config + `commit_message`) → Runner (executes) → Analyst (one-line verdict + classifies `candidate` | `ignore`). Optional Verifier downstream for VERIFIED → Pareto admission.
- **3-layer memory**: **L1** on disk (`history.jsonl`, `log.md`, `pareto.json`, `pareto_preview.json`, per-round artifacts) — the single source of truth; **L2** is an orchestrator-side snapshot rebuilt each round from L1 (never stored separately); **L3** is assembled per subagent role on dispatch (`l3_for_{ideator, coder, analyst}`). Never bypass L1 — all rounds append there, including synthetic reconciliation rows.
- **Worktree-based experiment model (spec §15)**: each round runs on branch `exp/<run_id>/<NN>-<slug>` in `.worktrees/<id>/`. The `pareto.json` archive is the **non-dominated set of VERIFIED branches** (not a top-5 truncation), keyed on `(delta_vs_baseline_holdout, flops_per_syndrome, n_params)`. `pareto_preview.json` is the top-5 projection sorted by `-delta_vs_baseline_holdout`. Compose rounds try `git merge parent-A parent-B` as a first-class scientific probe: on conflict the row is written with `status="compose_conflict"`, `branch=None`, `commit_sha=None`.
- **Subprocess Runner when `code_cwd` is set**: `run_round(cfg)` in-process raises `RunnerCallPathError` if `cfg.code_cwd is not None` — the worktree path must go through `autoqec.orchestration.subprocess_runner.run_round_in_subprocess`, which shells out to `python -m cli.autoqec run-round --code-cwd PATH ...`. Python's module-import cache makes in-process cwd switching unsound. The child hop uses the hidden `--_internal-execute-locally` flag so the CLI bypasses the dispatch branch and calls `run_round` directly — without this flag the subprocess would recurse into itself.
- **`round_attempt_id` vs `reconcile_id`** (mutually exclusive per §15.2): normal rounds carry `round_attempt_id` (UUID emitted by the Ideator at hypothesis time); synthetic rows written by `reconcile_at_startup` for recovered orphans carry `reconcile_id`. Never both. Never neither.
- **Startup reconciliation** (`autoqec.orchestration.reconcile`, §15.10): on every `cli/autoqec.py run`, diff `git branch --list 'exp/<run_id>/*'` against `history.jsonl` branches. `B\H` (branch without row): read `round_N_pointer.json` via `git show <branch>:round_<NN>/round_<NN>_pointer.json` → if recoverable, auto-heal (synthetic `orphaned_branch` row with **preserved** `round_attempt_id`); if pointer missing/malformed → emit `pause` action (no rename, no row — human review). `H\B` (history row without branch): emit idempotent `branch_manually_deleted` row (skipped if already present — schema allows `commit_sha=None` only for this status).
- **Tool whitelisting as reward-hacking defense** (`autoqec.runner.safety`): the Runner refuses to import anything touching training-set syndromes at eval time. Physical isolation, not policy, because LLM agents will find their way around prose-only constraints.
- **`machine_state` as self-awareness**: `autoqec.tools.machine_state` is an LLM-callable tool that returns `{gpu_vram_gb, recent_round_wallclocks, params_vs_time_scatter, active_worktrees}`. The Ideator uses it to avoid proposing rounds that don't fit the remaining compute budget.
- **Ideator's fork_graph view**: `l3_for_ideator(run_id=...)` returns a `fork_graph` dict with nodes `{branch, parent, delta_vs_parent, status, hypothesis_1line, on_pareto}`. Replaces the pre-§15 `last_5_hypotheses` shape. Fork decisions (`fork_from ∈ "baseline" | str | list[str]`) are explicit in `IdeatorResponse`.
- **Contract files require owner sign-off**: edits to `docs/contracts/interfaces.md` or `docs/contracts/round_dir_layout.md` need 3-of-3 sign-off (Chen / Xie / Lin) — these freeze the cross-component boundary. Add the `contract-change` label on PRs that touch them.

### Files Created Per Run (`runs/<run_id>/`)

- `history.jsonl` — append-only, one superset-`RoundMetrics` line per round. Written by `RunMemory.append_round`.
- `log.md` — human-readable narrative. Written by `RunMemory.append_log`.
- `pareto.json` — full non-dominated archive of VERIFIED branches. Replaced atomically by `RunMemory.update_pareto`.
- `pareto_preview.json` — top-5 projection sorted by `-delta_vs_baseline_holdout`.
- `round_<N>/config.yaml` — dump of `RunnerConfig.predecoder_config` (the DSL dict).
- `round_<N>/train.log` — `<step>\t<loss>` per line.
- `round_<N>/checkpoint.pt` — `{class_name, state_dict, output_mode, dsl_config}`.
- `round_<N>/metrics.json` — exactly `RoundMetrics` (§2.2 of interfaces.md).
- `round_<N>/round_<N>_pointer.json` — authoritative provenance, read by `reconcile.py`. Writer is the Runner (producer side not yet implemented — reconcile handles absence via `pause`).
- `round_<N>/verification_report.md` — prose from the Verifier (when Analyst verdict is `candidate`).

Orchestration writes **only** at the run root; the Runner writes **only** inside `round_<N>/`. The two sides never race.

### Runtime Branch Namespace

- `exp/<run_id>/<NN>-<slug>` — live experiment branches (one per round).
- `quarantine/<run_id>/...` — reserved for future use; current reconcile policy does NOT auto-rename (pause action only).
- `.worktrees/exp-<run_id>-<NN>-<slug>/` — per-round checkout (git-ignored).

## Documentation

- `README.md` — project overview, team ownership, deliverable roster
- `docs/superpowers/specs/2026-04-20-autoqec-design.md` — authoritative design spec (v2.3)
- `docs/superpowers/plans/*.md` — implementation plans (one per feature or per owner)
- `docs/contracts/interfaces.md` — pydantic schema contracts (`EnvSpec`, `RunnerConfig`, `RoundMetrics`, `VerifyReport`, predecoder I/O, subagent message format). **Sign-off required to edit.**
- `docs/contracts/round_dir_layout.md` — run-dir / round-dir file layout. **Sign-off required to edit.**
- `knowledge/` — 81-paper index + 3 synthesis documents (roadmap, strategic assessment, autoresearch patterns). Read-only reference material.

## Skills & Subagents

- `.claude/skills/autoqec-run/SKILL.md` — driver recipe for the full research loop. Invoked as `/autoqec-run`.
- `.claude/skills/read-zulip/SKILL.md` — read hackathon Zulip history for off-repo context recovery. Invoked as `/read-zulip`.
- `.claude/skills/add-env/SKILL.md` — wizard for adding a new environment YAML. Invoked as `/add-env`.
- `.claude/agents/autoqec-ideator.md` — Ideator subagent prompt (consumes `fork_graph`, emits `fork_from` + `compose_mode`).
- `.claude/agents/autoqec-coder.md` — Coder subagent prompt (aware that cwd is a worktree; emits `commit_message`).
- `.claude/agents/autoqec-analyst.md` — Analyst subagent prompt (output echoes `branch` + `commit_sha` when present).

When developing a new skill or subagent: put the `.md` alongside the existing ones; read from `memory.py::l3_for_<role>` for context assembly; use `autoqec.agents.dispatch.build_prompt` so the Tier-2 validator rules and schema constraints are surfaced to the model.

## Commit Convention

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`, `ci:`.

Scope tags used on this repo: `feat(schema)`, `feat(pareto)`, `feat(cli)`, `feat(memory)`, `feat(loop)`, `feat(worktree)`, `feat(subprocess_runner)`, `feat(fork_graph)`, `feat(runner)`, `feat(reconcile)`, `docs(agents)`, `docs(skill)`, `docs(contracts)`, `fix(cli)`, `fix(pareto)`, `fix(reconcile)`, `fix(schema)`, `chore(qa)`.

- One commit per logical change. TDD-style (failing test → implementation → passing test → commit).
- PRs touching `docs/contracts/*` must carry the `contract-change` label and collect 3-of-3 owner sign-off before merge.
- Never commit `runs/`, `.worktrees/`, `autoqec.egg-info/`, or local PDFs (all gitignored).
- Do not commit implementation plans until the PR that realizes them is ready to merge — keep `docs/superpowers/plans/` for design docs that capture key decisions.
