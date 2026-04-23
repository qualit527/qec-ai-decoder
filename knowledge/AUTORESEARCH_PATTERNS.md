# AutoQEC Code-Borrowing Recipe

Companion to `knowledge/autoresearch_refs/INDEX.md` (which catalogs each repo) and `docs/specs/2026-04-20-autoqec-design.md` (which defines the AutoQEC modules). This file is the **implementation-time decision guide**: for each AutoQEC module, which reference project do we lift code from, how much work is the port, and where does our domain force a divergence. Written for a 3-person team making "adapt vs reinvent" calls as they start coding.

Cross-references: `knowledge/DECODER_ROADMAP.md` (QEC decoder landscape), `knowledge/STRATEGIC_ASSESSMENT.md` (why this project, scoped risks) — treat their conclusions as given.

---

## 1. Executive summary

Three direct lifts that save us the most time, in order of ROI:

1. **`aide/aide/journal.py` (192 LOC, MIT) → `autoqec/orchestration/solution_tree.py`.** A ready-made `Node` + `Journal` dataclass pair with debug/improve/draft stage inference, buggy-flag tracking, `get_best_node`, metric history, JSON serialization via `dataclasses_json`. Swap the single-metric `MetricValue` for a multi-objective Pareto score and we have ~80% of Layer B. ≈150 LOC lifted, ≈60 LOC added.
2. **`open-coscientist/coscientist/framework.py` (467 LOC, MIT) → `autoqec/orchestration/dag.py`.** A working LangGraph state graph with typed `global_state.py` and Pydantic `custom_types.py` for inter-node messages. This is our stack choice already — keep the wiring, replace the node bodies with QEC subagent calls. ≈300 LOC lifted.
3. **`aide/aide/interpreter.py` (311 LOC, MIT) → `autoqec/runner/interpreter.py`.** Sandboxed Python subprocess with timeout, stdout/stderr capture, exception extraction, `ExecutionResult` dataclass. Drop-in for our non-LLM Runner (§4.5 of design spec). ≈280 LOC lifted.

**One architectural pattern to adopt wholesale:** AIDE's `search_policy` taxonomy — `draft / debug / improve` as three distinct LLM prompt templates driven by a policy function over the solution tree. It is more principled than the "one big prompt with lots of IF" pattern every other repo falls into, and maps cleanly to our Coder subagent's inner loop.

**Most disappointing repo:** `mlgym` (400 MB, Meta FAIR, CC-BY-NC 4.0). Reputation suggests a clean Gym-like environment abstraction we could copy; reality is a sprawling benchmark harness with task-specific scaffolding entangled with the abstraction, a non-commercial license, and the "clean `Environment` class" is thinner than what `rd-agent/core/scenario.py` (187 LOC, MIT) gives you in one file. Skim `mlgym/configs/` for YAML schema inspiration and move on.

---

## 2. Module-level borrowing recipe

Legend for **Effort**: **S** = <4 hrs, mostly copy + rename; **M** = 4–16 hrs, real adaptation; **L** = >16 hrs, port the pattern but rewrite for our domain.

| AutoQEC module | Donor repo → file | LOC to port | Effort | What must change for QEC |
|---|---|---|---|---|
| `autoqec/orchestration/dag.py` (LangGraph DAG, inline + background) | `open-coscientist/coscientist/framework.py` | ~300 of 467 | **M** | Replace hypothesis-generation / review / ranking nodes with Ideator / Coder / Runner / Reviewer / Pareto nodes. Add a `mode: Literal["inline","background"]` switch that routes to either a blocking loop (inline CC skill) or a detached subprocess with JSONL progress events. Keep their typed-state + Pydantic message pattern verbatim. |
| `autoqec/orchestration/state.py` (typed LangGraph state) | `open-coscientist/coscientist/global_state.py` + `custom_types.py` | ~150 | **S** | Swap their `Hypothesis` / `Ranking` message types for our `EnvSpec`, `DecoderArtifact`, `RunResult`, `ParetoPoint`. Keep the reducer pattern for list-appending fields (`history: Annotated[list, operator.add]`). |
| `autoqec/orchestration/solution_tree.py` (Coder↔Runner inner tree) | `aide/aide/journal.py` | ~150 of 192 | **S** | Replace `MetricValue` (scalar) with a `Score` dataclass containing `(ler, flops, params, train_time_s, is_pareto_dominated)`. Keep `Node.stage_name` property (`draft`/`debug`/`improve`), `debug_depth`, `get_best_node` (generalize to `get_pareto_front`), `draft_nodes` / `buggy_nodes` / `good_nodes` accessors. |
| `autoqec/agents/` (Python wrappers around Agent SDK / subprocess) | `agent-laboratory/agents.py` + `ai_lab_repo.py` | ~100 (prompt skeletons only) | **M** | Agent-Laboratory invokes LLMs via their own `agents.py` role classes; we replace the LLM invocation with a `subprocess.Popen(["claude", ...])` or `["codex", ...])` via our router. Port the role-prompt scaffolding and the `round`-based dispatch loop from `ai_lab_repo.py`'s main loop; drop all paper-drafting branches. |
| `autoqec/runner/interpreter.py` (non-LLM train+eval sandbox) | `aide/aide/interpreter.py` | ~280 of 311 | **S** | Add a resource-limit pre-exec (ulimit / psutil memory cap), add Stim + our QEC harness to the subprocess `PYTHONPATH`, surface FLOPs from our profiler as a structured field. Keep their timeout + exception-info capture verbatim. |
| `autoqec/runner/train_eval.py` (orchestrates training a decoder + computing LER/FLOPs) | `karpathy-autoresearch/train.py` (inner loop shape) | ~80 (shape, not code) | **M** | Karpathy's single-file loop is the right *shape* — read `train()` and `eval()` top-to-bottom, port the "each round: LLM proposes edit → execute → metric → plot" skeleton. All content replaced: our Runner is **non-LLM** and purely functional (input: DSL YAML or PyTorch script; output: `RunResult`). |
| `autoqec/eval/independent_eval.py` (reward-hacking defense) | `ml-master/grading_server.py` + AIDE's `review_func_spec` | ~200 | **M** | Run evaluation in a separate process from the Coder, with its own deterministic seeds and held-out noise realizations. Adopt AIDE's `submit_review` JSON schema (`is_bug`, `summary`, `metric`, `lower_is_better`) as the contract our Reviewer subagent returns; we extend with `(ler_holdout, variance_across_seeds, suspicious_pattern_flags)`. Cross-check: Coder's reported LER vs independent LER must match within tolerance. |
| `autoqec/decoders/dsl_schema.yaml` + `dsl_compiler.py` | `ai-scientist/templates/` structure (as inspiration) + `agent-laboratory/experiment_configs/` YAML schema | — | **L** | No direct code port — DSL is QEC-specific. Borrow only the *convention*: each template is a YAML file + a Python driver that instantiates a PyTorch module. Study `ai-scientist/templates/nanoGPT/experiment.py` for how they parameterize a model from a dict. |
| `autoqec/decoders/pytorch_templates/` (free-form C-PyTorch templates) | `ai-scientist-v2` template-free pattern | — | **M** | v2 abandoned templates, but we keep them for safety. Use their pattern of a minimal `train.py` harness + a model file the Coder freely edits. Borrow the file-layout convention from `ai-scientist-v2/ai_scientist/treesearch/interpreter.py`. |
| `autoqec/envs/base.py` (`EnvSpec`, `CodeSpec`, `NoiseSpec`, `Constraints`) | `rd-agent/rdagent/core/scenario.py` (interface) + `mlgym/configs/` (YAML schema only) | ~100 | **S** | `rd-agent`'s `Scenario` is an abstract base with `background`, `source_data`, `rich_style_description` properties — adapt directly. Our `EnvSpec` extends it with QEC fields (`code_family`, `distance`, `rounds`, `noise_model_name`, `p_phys`, `constraint_budget`). |
| `autoqec/envs/builtin/` (reference envs: `surface_sd`, `surface_circuit`) | none — greenfield | — | **L** | No port. Build two YAML envs by hand. Each references a `code_builder.py` that returns a Stim circuit + detector layout. Borrow the *pattern* of `problem-reductions/src/` where each canonical problem is a small module. |
| `autoqec/example_db/` (canonical decoder builders) | `openevolve/examples/` + `problem-reductions/examples/` layout | — | **S** | One builder module per known decoder family (MWPM, BP-OSD, small Transformer, small GNN). Each exposes `build(env: EnvSpec) -> nn.Module`. Mirrors `openevolve/examples/*/initial_program.py` convention — each example is self-contained. |
| `autoqec/llm/router.py` (claude-cli / codex-cli backends) | `aide/aide/backend/` (provider abstraction shape) | ~80 | **S** | `aide/backend/__init__.py` exposes `query(...)` with provider dispatch based on model name. Adapt to subprocess-based dispatch: `if model.startswith("claude-"): subprocess(["claude", ...])`, elif `codex-`: `subprocess(["codex", ...])`. Add a dry-run cost estimator (we have budget limits per run). |
| `autoqec/logging/lab_notebook.py` (markdown lab notebook) | `aide/aide/journal2report.py` | ~100 | **S** | Port verbatim, retarget template to our Pareto + history structure. One markdown file per run: hypotheses tried, code diffs, metrics, Pareto front snapshots. |
| `autoqec/pareto/front.py` + `topology.py` | `openevolve/openevolve/database.py` (MAP-Elites archive) | ~250 of ~400 | **M** | Their `Database` is a MAP-Elites grid — exactly the structure we need for "Pareto front with diverse architectures." Use their `(feature_descriptor → best_program)` binning, but our feature descriptor is `(param_count_bin, flops_bin)` and our quality score is LER. Keep their novelty-judge hook for the Ideator. |
| `.claude/agents/autoqec-*.md` (subagent prompts) | `agent-laboratory/agents.py` + `ai-scientist/ai_scientist/generate_ideas.py` prompt structure + `open-coscientist/coscientist/generation_agent.py` | ~200 words per agent | **M** | Five subagent files (`ideator`, `coder`, `runner-caller`, `reviewer`, `pareto-curator`). Borrow prompt *sections* from Agent-Laboratory roles (PhD/Postdoc/Engineer/Reviewer map 1:1), borrow the novelty-check loop from `ai_scientist/generate_ideas.py` (lines 100–250), borrow the constraint-enforcement framing from `open-coscientist/coscientist/configuration_agent.py`. |
| `.claude/skills/autoqec-run/SKILL.md` (inline orchestrator) | `problem-reductions/.claude/skills/run-pipeline/SKILL.md` | — | **S** | `problem-reductions` already has a `run-pipeline` skill with a polished "pick work item → dispatch → track" structure. Steal their shape: precondition check → state setup → loop dispatch → on-error recovery → final summary. Retarget to our env-selection + DAG-launch flow. |
| `.claude/skills/{add-env,verify-decoder,review-log}/SKILL.md` | `problem-reductions/.claude/skills/` (14 other skills as templates) | — | **S** | `problem-reductions` has `add-model`, `add-rule`, `verify-reduction`, `review-paper`, `review-pipeline`, `review-structural` — use each as a structural template for our analogs. Especially `verify-reduction/SKILL.md` → our `verify-decoder/SKILL.md`, same pattern of running a held-out check in isolation. |
| `Makefile` (`run`, `run-forever`, `run-with-codex`, `run-cheap`) | `problem-reductions/Makefile` | ~60 of ~200 | **S** | Their Makefile defines `run-pipeline`, `run-pipeline-forever`, `run-review`, `run-review-forever` with `RUNNER ?= codex` / `CLAUDE_MODEL ?= opus` knobs. Adopt the variable pattern and the forever-loop target structure verbatim. Substitute our phase names: `run` = one env, `run-forever` = poll env queue, `run-with-codex` sets `RUNNER=codex`, `run-cheap` sets `CLAUDE_MODEL=haiku`. |

**Total estimated lift:** ~1,600 LOC direct + ~1,000 LOC adapted, out of ~10,000 LOC target for the MVP. Rough rule: **16% direct copy, 10% adapt, 74% new code**. New code concentrated in `autoqec/decoders/` (DSL + templates), `autoqec/envs/builtin/` (reference envs), `autoqec/runner/train_eval.py` (QEC-specific training), and subagent prompts.

---

## 3. Cross-cutting patterns

### Adopt wholesale

**AIDE's draft/improve/debug step policies (`aide/agent.py` lines 61–92).** The `search_policy` function picks one of three LLM prompts based on tree state: draft if under-drafted, debug with probability `p_debug` if buggy leaves exist, else improve the greedy-best. It's 30 LOC of clean control flow that every other ref-project reinvents worse. Use as the Coder subagent's inner loop literally — the only change is our "metric" is a Pareto dominance check, not a scalar.

**`problem-reductions`' skill-as-pipeline convention + `make run-pipeline-forever`.** Their repo is the single best existing example of "a research project whose workflow IS a set of Claude Code skills." Every human-invokable action (`run-plan`, `run-issue`, `run-pipeline`, `run-review`) is both a Make target and a skill. The forever targets are tight polling loops that call the one-shot targets. Mirror their directory layout and naming.

**`open-coscientist` Pydantic-typed LangGraph state.** Their `global_state.py` uses `TypedDict` with `Annotated[list, operator.add]` reducers for append-only fields, and `custom_types.py` has Pydantic models for every inter-node message. This eliminates an entire class of bugs (wrong-shape state hand-offs) that torture every agent framework that uses `dict[str, Any]`. Non-negotiable for our DAG.

### Adopt partially

**`ai-scientist-v2` BFTS config schema (`bfts_config.yaml`).** For the Layer C tree-search upgrade (post-MVP in the design spec), port their config keys: `max_nodes`, `branching_factor`, `expansion_budget`, `pruning_threshold`, `parallel_branches`. Do **not** port their agent-manager complexity upfront — it's 1,221 LOC and most of that handles their LaTeX/plot-review path we don't need.

**RD-Agent's `EvolvingFramework` (`rdagent/core/evolving_framework.py`, 187 LOC).** Read it for the abstract interfaces (`EvolvableSubjects`, `EvolvingStrategy.evolve_iter()` as a generator, `IterEvaluator.evaluate_iter()`), then write our own instantiation. Don't try to use their class hierarchy directly — it's built around their `KnowledgeBase` + `Feedback` object graph which is heavier than we need. The value is the **interface vocabulary**, not the code.

**`openevolve` MAP-Elites archive (`openevolve/database.py`).** Adopt for our Pareto-with-diversity module (bin by `(param_count, flops)`, keep the best LER per bin). Skip the novelty-judge prompt contents — they're domain-specific. Skip their `process_parallel.py` — it's Ray-based and heavier than we want.

**`paper-qa` `Docs` class.** For MVP, the Ideator reads `knowledge/INDEX.md` directly in its context window (small enough to fit). Post-MVP, when the corpus outgrows context, wrap `paperqa.Docs.aquery()` as a tool exposed to the Ideator subagent. Don't build RAG infrastructure until we need it.

### Reject

**RD-Agent's `EvolvingFramework` as runtime.** See above — adopt the interfaces, reject the implementation.

**`agent-laboratory`'s `papersolver.py` + paper-drafting roles.** We don't auto-write papers. Skip.

**`mlgym`'s benchmark-task coupling.** Their `Environment` abstraction is useful on paper but in practice is entangled with their benchmark task data. Take the YAML schema idea and leave the code.

**`ai-scientist` v1's template-rigidity.** v1 required a hand-authored template per experiment domain. We inherit their *separation of concerns* (ideator / experimentalist / reviewer) but follow v2's template-free pattern for C-PyTorch decoders while keeping structured templates only for B-DSL.

---

## 4. License & dependency audit

**License inventory** (verified by reading `LICENSE` files at each repo root, 2026-04-21):

| Repo | License | OK to lift code? |
|---|---|---|
| `aide` | MIT | Yes |
| `agent-laboratory` | MIT | Yes |
| `rd-agent` | MIT (Microsoft) | Yes |
| `open-coscientist` | MIT | Yes |
| `problem-reductions` | MIT | Yes |
| `openevolve` | Apache-2.0 | Yes (include NOTICE) |
| `paper-qa` | Apache-2.0 | Yes (include NOTICE) |
| `sciagents` | Apache-2.0 | Yes (include NOTICE) |
| `dolphin` | CC-BY-NC 4.0 | **Inspiration only** — do not lift code |
| `mlgym` | CC-BY-NC 4.0 | **Inspiration only** — do not lift code |
| `ai-scientist` v1 | "AI Scientist Source Code License" (custom, RAIL-derived) | **Do not lift** — non-OSI, use-restricted |
| `ai-scientist-v2` | Same custom license as v1 | **Do not lift** — non-OSI |
| `karpathy-autoresearch` | No LICENSE file | **Inspiration only** — shape reads from blog post, but no explicit license means no legal redistribution rights |
| `ml-master` | No LICENSE file | **Inspiration only** — same reasoning |
| `internagent` | No LICENSE file | **Inspiration only** |

**Practical implication:** every direct-lift in §2 targets MIT or Apache-2.0 code. The most valuable repos (AIDE, open-coscientist, RD-Agent, problem-reductions, openevolve) are all MIT or Apache-2.0 — the borrowing plan is legally clean. CC-BY-NC and the custom AI-Scientist license explicitly forbid commercial use or impose use restrictions; we read them for architecture ideas only.

**No GPL-class licenses in the set** — nothing forces our code open-source by linkage. Copyleft avoided by luck rather than design; continue to double-check if adding more refs later.

**Notable heavy dependencies to isolate or avoid at the top level:**

- `openevolve` pulls in `ray` for parallel eval — we don't need Ray; keep their `database.py` pure-Python code and reimplement the parallel layer with `concurrent.futures` or asyncio.
- `ai-scientist-v2` uses `vllm` for local LLM inference — irrelevant to us (we use CLI backends).
- `paper-qa` depends on specific `torch` + sentence-transformers versions — fine to pull in via optional extras (`autoqec[rag]`), don't make it a core dep.
- `mlgym` has 400 MB of benchmark data — never clone deeper than we already did.

---

## 5. Implementation ordering

**Reading order before touching code** (~4 hours of reading):

1. `aide/aide/journal.py` (192) → `aide/aide/agent.py` first 200 LOC → `aide/aide/interpreter.py` (311). This gives you the inner-loop mental model.
2. `open-coscientist/coscientist/framework.py` (467) → `global_state.py` → `custom_types.py`. This gives you the LangGraph outer-loop mental model.
3. `rd-agent/rdagent/core/evolving_framework.py` (187) → `scenario.py`. This gives you the right abstraction vocabulary for `EnvSpec`.
4. `problem-reductions/.claude/skills/run-pipeline/SKILL.md` → `problem-reductions/Makefile` (top 100 lines). This gives you the harness-level skill + Make integration pattern.

**Build order (MVP, ~3 weeks for a team of 3):**

**Week 1 — scaffolding.** Port `aide/journal.py` → `autoqec/orchestration/solution_tree.py` (S, ~1 day). Port `aide/interpreter.py` → `autoqec/runner/interpreter.py` (S, ~1 day). Write `autoqec/envs/base.py` modeled on `rd-agent/core/scenario.py` (S, ~1 day). Write minimal `autoqec/llm/router.py` (subprocess dispatch only, no cost estimation yet; S, ~1 day). Build one reference env (`surface_sd`) end-to-end by hand with a hand-written MWPM wrapper (L, ~4 days).

**Week 2 — DAG + subagents.** Port `open-coscientist/framework.py` skeleton → `autoqec/orchestration/dag.py` (M, ~3 days). Write `autoqec/orchestration/state.py` (S, ~1 day). Write the five subagent `.md` files, adapting Agent-Laboratory prompts (M, ~3 days). Hook up the `autoqec-run` skill, modeled on `problem-reductions/run-pipeline` (S, ~1 day). Makefile (S, ~half day).

**Week 3 — independent eval + Pareto + polish.** Build `autoqec/eval/independent_eval.py` with AIDE's `review_func_spec` schema + ML-Master's isolated-grader pattern (M, ~2 days). Port `openevolve/database.py` → `autoqec/pareto/front.py` (M, ~2 days). Port `aide/journal2report.py` → `autoqec/logging/lab_notebook.py` (S, ~1 day). Second reference env (`surface_circuit`), reusing all plumbing (S, ~1 day). Testing + bugfixes (~2 days).

**Critical path = week 1's reference env.** Everything after that is wiring; the reference env is where QEC specificity concentrates, and it blocks testing the DAG end-to-end. Start it day 1.

---

## 6. Anti-patterns to avoid

**All of them hardcode LLM API keys and provider names.** Every repo in the set dispatches LLMs via provider-name strings (`"openai"`, `"anthropic"`, `"mistral"`) with API-key env vars. We intentionally use **CLI subprocess dispatch** (`claude`, `codex`), so key management is out of our process. Don't regress to API-key-based dispatch "for convenience" — it undoes a major safety property (the user's existing CLI auth flows stay the authority).

**Most of them conflate Coder and Runner.** In AIDE, Agent-Laboratory, ML-Master, the "agent" both writes code and executes it in-process. That's why they need elaborate sandboxing. Our design spec makes Runner **non-LLM and out-of-process**; preserve that boundary even when it's tempting to fold in.

**All of them silently drop failed branches.** When a code-gen attempt has an unparseable LLM output or a timeout, most refs log a warning and move on. For a scientific harness this is reward-hacking surface — a subtly-failing Coder could make the rest of the pipeline look more successful than it is. Our independent evaluator must treat dropped branches as **explicit failures**, counted and reported, not filtered out of summaries.

**Agent-Laboratory and ai-scientist v1 have a "Reviewer role" that re-scores the Coder's outputs using the same LLM.** This is a known reward-hacking vector (model can learn to write code + prose that appeals to its own critic). Our `verify-decoder` runs a **non-LLM** check on held-out noise seeds, completely separate from the Reviewer subagent that summarizes findings. Don't let the elegance of "just ask the LLM to review" slide.

**Several repos (especially `mlgym`, `internagent`) over-generalize the `Environment` abstraction.** `EnvSpec` should stay minimal: a YAML config + a Python `build()` function, no methods the Coder must override, no per-env agent classes. Every generalization we copy makes `autoqec/envs/contribute.md` harder to write.

**`karpathy-autoresearch/train.py` and `dolphin/launch_dolphin.py` are both single-metric.** Do not let their elegance tempt you into collapsing our multi-objective (LER, FLOPs, params) signal into a scalar "for simplicity." The multi-objective structure is load-bearing for the Pareto module, the Reviewer's prompts, and the `verify-decoder` cross-check.

**`ai-scientist-v2` has 2,368 LOC in one file (`parallel_agent.py`).** Don't mirror that. Our LangGraph DAG stays in 400–600 LOC; if a node grows past 200 LOC, extract it. The 800-LOC-file limit from `common/coding-style.md` is not negotiable here.

**The "infinite research loop" temptation.** Several repos (InternAgent, Dolphin, openevolve) demo "it just runs forever." For a physics-adjacent tool that's reviewed by skeptical domain experts, determinism and reproducibility matter more than runtime. Make every run seeded, bounded by explicit node budgets, and fully replayable from the lab notebook. Don't copy the "let it cook overnight" framing into our docs.
