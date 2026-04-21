# Autoresearch Reference Projects

Generated: 2026-04-21.

Open-source LLM-agent-driven auto-research systems, cloned shallow (`--depth 1`) for reference. This collection informs the AutoQEC design — we borrow patterns rather than reinvent.

## How to use

For each repo below:
- **Key files** point to the concrete code you should read before writing the AutoQEC equivalent.
- **What to steal** lists the specific patterns worth adopting.
- **What to skip** calls out the tied-to-their-domain parts we do NOT adopt.

See also `knowledge/INDEX.md` for the paper literature, and `docs/superpowers/specs/2026-04-20-autoqec-design.md` for how these inform AutoQEC.

---

## 1. `ai-scientist/` — Sakana AI Scientist v1 (2024)

**Paper**: arXiv:2408.06292 • **Repo**: https://github.com/SakanaAI/AI-Scientist • **Clone size**: 48 MB

Linear pipeline: Ideation → Experiment → Writer → Reviewer. Relies on human-authored code templates for each experiment domain.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `ai_scientist/generate_ideas.py` | 546 | **Ideator** — loop that generates + filters hypotheses. Good pattern for our `autoqec-ideator`. |
| `ai_scientist/perform_experiments.py` | — | **Runner** — executes the coded experiment from a template. |
| `ai_scientist/perform_review.py` | — | **Reviewer** — LLM-driven paper review (we skip this). |
| `ai_scientist/perform_writeup.py` | — | Paper writer (we skip). |
| `ai_scientist/llm.py` | — | Thin LLM provider abstraction (Anthropic/OpenAI/Mistral). |
| `launch_scientist.py` | — | Top-level CLI orchestrator. |
| `templates/` | — | Per-experiment seed templates. **Analog for us**: DSL templates in `autoqec/decoders/dsl_schema.yaml`. |

### What to steal
- `generate_ideas.py`'s "novelty check" loop (re-prompts until idea passes a similarity filter)
- Template-based experiment seeds (AutoQEC's DSL is our version)
- Clean separation of ideator / experimentalist / reviewer phases — **this is the original DAG pattern we inherit**

### What to skip
- Writer stack (we don't auto-generate papers)
- The `experimental/` branch — abandoned experiments, not canonical

---

## 2. `ai-scientist-v2/` — Sakana AI Scientist v2 (2025) ⭐

**Paper**: arXiv:2504.08066 • **Repo**: https://github.com/SakanaAI/AI-Scientist-v2 • **Clone size**: 11 MB

Major rewrite: **Agentic Tree Search** + **Experiment Manager** + parallel branches + VLM figure review. No templates — generates all code from scratch per idea.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `ai_scientist/treesearch/parallel_agent.py` | 2368 | **Main DAG core**. Parallel exploration of ideas as tree nodes, scoring, pruning. |
| `ai_scientist/treesearch/agent_manager.py` | 1221 | **Orchestrator**. Manages concurrent agent branches, kills failed ones, resource budget. |
| `ai_scientist/treesearch/bfts_utils.py` | 76 | Best-First Tree Search helpers (heap, node scoring). |
| `ai_scientist/treesearch/perform_experiments_bfts_with_agentmanager.py` | — | The run loop — this is the function analog to our LangGraph DAG. |
| `ai_scientist/treesearch/interpreter.py` | — | Sandboxed Python interpreter for Runner-equivalent (based on Jupyter). |
| `ai_scientist/treesearch/journal.py` | — | Persistence of tree nodes & experiment history. |
| `ai_scientist/perform_ideation_temp_free.py` | — | **Template-free ideation** — v2's big change vs v1. |
| `ai_scientist/perform_vlm_review.py` | — | **VLM reviews figures** to catch nonsense plots. |
| `launch_scientist_bfts.py` | — | Main CLI. |
| `bfts_config.yaml` | — | Tree-search hyperparameters (branching factor, max depth, node budget). |

### What to steal
- **`parallel_agent.py` + `agent_manager.py`** — this is THE reference for AutoQEC Layer C (hypothesis tree search upgrade) and Layer B (solution-tree Coder↔Runner). Read these two before writing our orchestration.
- **`bfts_config.yaml`** schema — adopt nearly verbatim for our `autoqec/orchestration/tree_config.yaml`
- VLM figure review concept → optional post-MVP for validating AutoQEC Pareto plots

### What to skip
- Paper-writer stack (we don't generate papers)
- ICML / ICBINB LaTeX templates
- Few-shot examples tied to ML research domain (we need QEC-specific ones)

---

## 3. `aide/` — AIDE (Weco) ⭐

**Paper**: MLE-bench (arXiv:2410.07095) • **Repo**: https://github.com/WecoAI/aideml • **Clone size**: 2 MB

Single-agent solution-tree for Kaggle-style ML engineering. SOTA on MLE-bench. Compact and elegant.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `aide/agent.py` | 339 | **Core loop**: draft / improve / debug step policies + tree traversal. **Read this first** — most concise reference. |
| `aide/journal.py` | 192 | Tree-of-solutions data structure. Each node = a candidate Python script + its execution result. |
| `aide/interpreter.py` | — | Sandboxed Python subprocess runner with timeout + stdout capture. |
| `aide/backend/` | — | LLM backends (OpenAI, Anthropic, local). |
| `aide/run.py` | — | CLI entry. |
| `aide/journal2report.py` | — | Final markdown report generator. |
| `aide/webui/` | — | Live web dashboard. |

### What to steal ⭐⭐⭐
- **`journal.py`** — almost a drop-in model for our `autoqec/orchestration/solution_tree.py` (Coder↔Runner inner loop, §7 of design spec)
- **`interpreter.py`** — sandboxed execution pattern for our Runner
- **`agent.py` step policies** — "draft / improve / debug" as three LLM prompts is cleaner than our current plan; steal the taxonomy
- Webui for live monitoring — nice-to-have post-MVP

### What to skip
- MLE-bench-specific loaders (Kaggle datasets)
- Their specific scoring rubric (we have LER + FLOPs + params)

---

## 4. `agent-laboratory/` — Agent Laboratory / AgentRxiv (Schmidgall et al., MIT, 2025)

**Paper**: arXiv:2501.04227 • **Repo**: https://github.com/SamuelSchmidgall/AgentLaboratory • **Clone size**: 3.1 MB

Role-based multi-agent (PhD Student, Postdoc, Engineer, Reviewer). Closest prior art to our multi-agent DAG structure. `mlesolver.py` is a nested AIDE-like engineer loop.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `ai_lab_repo.py` | 891 | **Orchestrator** — role dispatch + round loop. Closest analog to our `autoqec-run` skill / orchestrator. |
| `agents.py` | — | Role prompts (PhD, Postdoc, SW Engineer, ML Engineer, Reviewer). **Analog for us**: `.claude/agents/autoqec-*.md`. |
| `mlesolver.py` | — | Nested engineer loop (draft / edit / debug) for the ML engineering subtask. AIDE-inspired. |
| `papersolver.py` | — | Paper drafting subagent (skip). |
| `tools.py` | — | Shared tool definitions (`arxiv_search`, `python_eval`, `write_file`). |
| `experiment_configs/` | — | YAML configs per research project. |

### What to steal
- **Role decomposition** (PhD = Ideator, Postdoc = Reviewer, SW+ML Engineer = Coder+Runner, Reviewer = Reviewer). Direct 1:1 mapping to our subagents.
- `experiment_configs/*.yaml` schema — influences our `envs/*.yaml` design
- `mlesolver.py`'s dual-pass (idea-stage, implementation-stage) pattern

### What to skip
- Paper drafting (`papersolver.py`)
- arxiv MCP integration tied to their specific auth flow

---

## 5. `karpathy-autoresearch/` — Karpathy's minimal autoresearch loop (2026-03)

**Repo**: https://github.com/karpathy/autoresearch • **Clone size**: 1.2 MB

The "terminus" of minimalism — one file, one metric, one loop. ~630 LOC. Reference for how simple the DSL-only path can be.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `train.py` | 630 | **Entire system** — orchestrator, LLM call, file edits, metric tracking, plot. |
| `program.md` | — | Natural-language task spec (fed to the LLM). |
| `prepare.py` | — | Data prep. |
| `analysis.ipynb` | — | Post-hoc analysis notebook. |

### What to steal
- The ~30-line main loop in `train.py` — use as literal template for our MVP linear inner loop (before upgrading to AIDE tree)
- `program.md` idea — a natural-language task brief that the LLM reads each round, instead of a Python config. **Our analog**: `envs/*.yaml` read as text block to subagent.

### What to skip
- Fixed single-metric assumption — we're multi-objective (Pareto)
- nanochat-specific scaffolding

---

## 6. `rd-agent/` — Microsoft R&D-Agent (2024+)

**Paper**: arXiv:2407.18690 • **Repo**: https://github.com/microsoft/RD-Agent • **Clone size**: 5.2 MB

Enterprise-grade auto-research framework. Designed for quant finance and general ML R&D. Has an "evolving framework" abstraction that's more formal than v2's tree search.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `rdagent/core/evolving_framework.py` | 187 | **Abstract loop**: `EvolvingStrategy`, `Evolvable`, `EvolvingFeedback`. Cleanest abstract DAG interface I've seen. |
| `rdagent/core/proposal.py` | — | Hypothesis proposal abstractions. |
| `rdagent/core/evaluation.py` | — | Eval + feedback abstractions. |
| `rdagent/core/developer.py` | — | "Developer" agent role (= our Coder). |
| `rdagent/core/scenario.py` | — | Domain abstraction (= our EnvSpec). |
| `rdagent/core/knowledge_base.py` | — | Cross-round memory (= our `history` + Pareto). |
| `rdagent/scenarios/data_science/` | — | Data-science-specific scenario (Kaggle-style). |
| `rdagent/scenarios/qlib/` | — | Quant-finance scenario. |
| `rdagent/scenarios/rl/` | — | RL scenario. |

### What to steal
- **`evolving_framework.py`** — 187 LOC of pure abstraction. Use it as the *interface* for our LangGraph DAG, even if we ultimately use LangGraph for the implementation.
- `scenario.py` interface — directly informs our `EnvSpec` base class design
- `knowledge_base.py` — our Reviewer needs exactly this (cross-round accumulated memory)

### What to skip
- Enterprise UI / auth layer
- qlib (quant finance) specifics

---

## Summary — which repo to read in which order

For someone implementing AutoQEC from the design spec:

1. **Read first**: `aide/aide/agent.py` + `aide/aide/journal.py` (531 LOC total) — most digestible full system, inner loop pattern
2. **Read next**: `rd-agent/rdagent/core/evolving_framework.py` + `proposal.py` + `scenario.py` — clean abstractions for our `EnvSpec` and DAG interfaces
3. **Read before implementing tree search**: `ai-scientist-v2/ai_scientist/treesearch/parallel_agent.py` + `agent_manager.py` — parallel hypothesis expansion, best-first pruning
4. **Read for role decomposition**: `agent-laboratory/agents.py` + `ai_lab_repo.py` — how to split one pipeline across role prompts
5. **Read for minimalist inspiration**: `karpathy-autoresearch/train.py` — when scope creep threatens, re-read this

## Update policy

Each of these was cloned `--depth 1` on 2026-04-21. To refresh:

```bash
cd knowledge/autoresearch_refs
for d in */; do (cd "$d" && git pull --ff-only); done
```

Full clones (`--depth 1` without history) live under `.git/` inside each dir; delete `.git/` if you want pure-code snapshots (but then `git pull` no longer works).

---

## Additional projects (pending research agent)

A parallel research agent is surveying GitHub/web for other 2025-2026 auto-research repos worth cloning (MLE-STAR, MLE-Solver, ResearchAgent, Co-Scientist, etc.). When it returns, this section will be populated and additional clones added to this folder.
