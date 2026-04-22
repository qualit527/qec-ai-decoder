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

---

## 7. `open-coscientist/` — LangGraph reimplementation of Google Co-Scientist ⭐⭐⭐

**Repo**: https://github.com/conradry/open-coscientist-agents • **Clone size**: 13 MB

Open-source LangGraph + GPT-Researcher reimplementation of Google Co-Scientist's hypothesis-evolve-rank loop. **Directly matches our stack choice** (LangGraph DAG). Most aligned reference we have.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `coscientist/framework.py` | 467 | **Main LangGraph state graph** — nodes, edges, state reducers. Direct template for our `autoqec/orchestration/dag.py`. |
| `coscientist/generation_agent.py` | 282 | Generation agent node. Analog: our Ideator. |
| `coscientist/evolution_agent.py` | 181 | Evolution operator (crossover/mutate on hypotheses). |
| `coscientist/literature_review_agent.py` | — | Lit-review node backed by Tavily/web search — analog to our knowledge/INDEX reference. |
| `coscientist/configuration_agent.py` | — | Per-hypothesis configuration planner. |
| `coscientist/final_report_agent.py` | — | Report synthesis. |
| `coscientist/global_state.py` | — | Typed LangGraph state schema. |
| `coscientist/custom_types.py` | — | Pydantic models for messages between nodes. |

### What to steal ⭐⭐⭐
- **`framework.py` LangGraph layout** — use nearly verbatim as the skeleton for our DAG; only swap out the agents/state fields for QEC ones
- `global_state.py` typed-state pattern → `autoqec/orchestration/state.py`
- `custom_types.py` Pydantic message schemas between nodes → our agent message contracts
- Evolution operator ideas — later, post-MVP tree search upgrade

### What to skip
- GPT-Researcher literature integration (we have our own paper corpus)
- Final-report node (we write our own markdown log)

---

## 8. `mlgym/` — Meta MLGym framework (2025)

**Paper**: https://ai.meta.com/research/publications/mlgym/ • **Repo**: https://github.com/facebookresearch/MLGym • **Clone size**: 400 MB (heavy benchmark data)

Meta's Gym-style framework + benchmark for AI research agents. Clean `Environment` / `Agent` / `Backend` separation we should steal for `EnvSpec` design.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `mlgym/environment/` | — | Environment abstraction — tasks expose `step()` / `reset()` / `reward()`. Directly informs `autoqec/envs/base.py`. |
| `mlgym/agent/` | — | Agent abstraction. |
| `mlgym/backend/` | — | LLM provider adapters. |
| `mlgym/evaluation/` | — | Metric loggers. |
| `mlgym/tools/` | — | Tool registry pattern. |
| `configs/` | — | YAML-driven experiment configs — YAML template for our `envs/*.yaml`. |

### What to steal
- **Environment / Agent separation** as a hard boundary — our `EnvSpec`↔Coder boundary is the same pattern
- YAML config schema design in `configs/`
- Tool registry layout in `mlgym/tools/`

### What to skip
- Benchmark task data (data/, demonstrations/) — large and ML-specific, not QEC

---

## 9. `paper-qa/` — Future-House high-accuracy RAG over scientific PDFs

**Repo**: https://github.com/Future-House/paper-qa • **Clone size**: 60 MB

Production-grade RAG system for scientific papers. Drop-in candidate for grounding our Ideator in the QEC literature we collected (81 markdown papers).

### Key files
| File | LOC | Purpose |
|---|---|---|
| `src/paperqa/docs.py` | 721 | **`Docs` store** — PDF → chunks → embeddings → retrieval pipeline. |
| `src/paperqa/core.py` | 400 | Core retrieval + synthesis loop. |
| `src/paperqa/agents/` | — | Agent-based query planning. |
| `src/paperqa/readers.py` | — | PDF / markdown / HTML readers. |
| `src/paperqa/clients/` | — | Literature provider clients (Crossref, Semantic Scholar, OpenAlex). |
| `src/paperqa/settings.py` | — | Pydantic settings hierarchy — nice config pattern. |

### What to steal
- **`Docs` class** — if we want RAG over `knowledge/papers_md/`, this is essentially turnkey. Our Ideator subagent could call a tool that invokes paper-qa `Docs.aquery()` on a hypothesis.
- `clients/` API abstractions for later post-MVP automatic paper harvesting

### What to skip
- For MVP, we don't need RAG (Ideator reads `INDEX.md` directly). Revisit when corpus grows past what fits in one subagent context.

---

## 10. `openevolve/` — AlphaEvolve open-source reimplementation

**Repo**: https://github.com/codelion/openevolve • **Clone size**: 12 MB

Evolutionary program search with LLM mutation operator. A **different paradigm** from chain-of-thought or tree-search: maintain a population of program variants, LLM mutates/crossovers, fitness-based selection. Highly relevant for "evolve a decoder architecture over many generations."

### Key files
| File | LOC | Purpose |
|---|---|---|
| `openevolve/controller.py` | 583 | Main evolution loop — population maintenance, generation scheduling. |
| `openevolve/iteration.py` | 211 | Single iteration logic (sample parents, mutate, evaluate, insert). |
| `openevolve/evaluator.py` | 727 | Fitness evaluation with sandboxing. |
| `openevolve/database.py` | — | Population / archive store (MAP-Elites-like). |
| `openevolve/novelty_judge.py` | — | Prevents convergence to trivial solutions. |
| `openevolve/process_parallel.py` | — | Parallelism for batched evaluations. |
| `configs/` | — | Evolution hyperparams per example. |
| `examples/` | — | Worked examples (circle-packing, symbolic regression, etc.). |

### What to steal
- **Evolutionary search paradigm** — AutoQEC post-MVP could add a Layer D (population-based search) alongside B/C (tree search). For decoders, crossing over two architectures is non-trivial but doable with DSL.
- `novelty_judge.py` — prevent Ideator from proposing near-duplicates of past hypotheses
- `database.py` MAP-Elites archive — excellent for maintaining Pareto fronts with diverse architectures

### What to skip
- For MVP, tree search is sufficient — evolutionary search is stretch

---

## 11. `ml-master/` — SJTU ML-Master (MLE-Bench SOTA, 2026-04)

**Repo**: https://github.com/sjtu-sai-agents/ML-Master • **Clone size**: 26 MB

Current #1 on MLE-Bench. Hierarchical MCTS + Cognitive Caching. Relevant for the "long-horizon R&D" aspect.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `main_mcts.py` | 214 | MCTS launch script — entry point. |
| `agent/mcts_agent.py` | 960 | **Full MCTS agent implementation** — state, action, rollout, backprop. |
| `interpreter/` | — | Sandboxed Python execution. |
| `backend/` | — | LLM + judging backends. |
| `grading_server.py` | — | Isolated grader process (reward-hacking defense). |

### What to steal
- **`mcts_agent.py`** — more rigorous than AIDE's greedy best-first. Upgrade path for Layer C if BFTS from AI-Scientist-v2 doesn't scale.
- `grading_server.py` isolated-grader pattern → our `/verify-decoder` uses same idea

### What to skip
- Dataset loaders tied to Kaggle

---

## 12. `internagent/` — InternScience InternAgent v1.5 (2026)

**Repo**: https://github.com/InternScience/InternAgent • **Clone size**: 17 MB

Unified long-horizon scientific discovery framework across multiple domains. Most complete "end-to-end scientific research agent" OS reference.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `launch_discovery.py` | 324 | **Main CLI** — top-to-bottom flow through stages. |
| `internagent/mas/` | — | Multi-agent system module. |
| `internagent/stage.py` | — | **Stage abstraction** — each research phase is a pluggable stage. Very close to our pipeline-stages-as-skills design. |
| `internagent/experiments_utils_aider.py` | — | Aider-based code editing integration. |
| `internagent/prompts.py` | — | All role prompts centralized. |
| `internagent/vis_tree.py` | — | Research tree visualization — useful for our research log. |
| `tasks/` | — | Per-domain task YAML configs. |

### What to steal
- **`stage.py`** — clean "stage" abstraction mirrors our `.claude/skills/` pattern at the Python layer
- `vis_tree.py` — we want something like this for the final Pareto+hypothesis visualization

### What to skip
- Aider-specific integration (we use Claude Code / Codex CLI instead)

---

## 13. `dolphin/` — ACL'25 closed-loop auto-research (InternScience)

**Repo**: https://github.com/InternScience/Dolphin • **Clone size**: 16 MB

Minimal closed-loop design (think / practice / feedback). Smaller and more readable than InternAgent, good for understanding the core loop.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `launch_dolphin.py` | 267 | Main entry — full loop in <300 LOC. |
| `dolphin_utils/generate_ideas.py` | 630 | Idea generation with RAG. |
| `dolphin_utils/rag_tools/` + `rag_utils.py` | — | Lightweight RAG over a paper corpus. |
| `dolphin_utils/experiments_utils.py` | — | Experiment execution wrappers. |
| `dolphin_utils/prompts.py` | — | Prompt library. |
| `examples/` | — | Worked closed-loop examples. |

### What to steal
- **`launch_dolphin.py`** (267 LOC) as a "minimal closed loop" reference alongside `karpathy-autoresearch/train.py`
- Lightweight RAG pattern if we decide not to pull in paper-qa

---

## 14. `sciagents/` — MIT SciAgentsDiscovery (materials science, 2024-25)

**Paper**: Multiple Buehler-group publications • **Repo**: https://github.com/lamm-mit/SciAgentsDiscovery • **Clone size**: 223 KB

Knowledge-graph-driven multi-agent discovery for materials science — **closest physical-science analog to QEC** in our collection.

### Key files
| File | LOC | Purpose |
|---|---|---|
| `ScienceDiscovery/agents.py` | 406 | Multi-agent role definitions (materials domain). |
| `ScienceDiscovery/graph.py` | 19 | Graph / ontology utilities. |
| `ScienceDiscovery/llm_config.py` | — | LLM configs. |
| `Notebooks/` | — | End-to-end discovery notebooks. |

### What to steal
- Approach of using a **domain knowledge graph** to ground the Ideator — for QEC, we could build a graph of (code, noise, decoder_family) relationships over time
- Multi-agent role layout — lighter weight than Agent Laboratory

### What to skip
- Materials-specific prompts / ontology

---

## Additional repos surveyed but NOT cloned

The research agent found 13 more projects we chose not to clone. Listed here with reasoning:

| Repo | Reason for skip |
|---|---|
| `MASWorks/ML-Agent` | RL-trained Qwen agent — niche; paradigm covered by ml-master |
| `MLSysOps/MLE-agent` | Mature pair-programming agent — overlaps with AIDE |
| `openai/mle-bench` | Benchmark (not a harness) — useful as future eval reference but not MVP |
| `openai/frontier-evals` (PaperBench) | Replication benchmark, not a harness |
| `MLE-Dojo/MLE-Dojo` | Benchmark only |
| `aibuildai/AI-Build-AI` | Proprietary-ish recent project, limited docs |
| `HKUDS/AI-Researcher` | Covered by InternAgent + ai-scientist-v2 combined |
| `Just-Curieous/Curie` | Overlaps with other scientific agents we have |
| `ulab-uiuc/research-town` | Community simulator — interesting but orthogonal to MVP |
| `JinheonBaek/ResearchAgent` | 33 stars, small subset of what InternAgent covers |
| `gomesgroup/coscientist` | Wet-chemistry lab — too domain-specific |
| `IntologyAI/Zochi` | Closed-source aspects |
| `OSU-NLP-Group/ScienceAgentBench` | Benchmark |
| `Future-House/aviary` | Gym-like; overlaps with MLGym |
| `findalexli/ai-scientist-v3` | 12-star personal fork |
| `google-deepmind/funsearch` | Superseded by openevolve (which reimplements it more completely) |

All are mentioned here so anyone looking back knows they were considered.

---

## Summary — which repo to read in which order

For someone implementing AutoQEC from the design spec:

1. **Read first**: `aide/aide/agent.py` + `aide/aide/journal.py` (531 LOC total) — most digestible full system, inner-loop pattern
2. **Read second**: `open-coscientist/coscientist/framework.py` (467 LOC) — LangGraph DAG skeleton we'll copy most directly
3. **Read third**: `rd-agent/rdagent/core/evolving_framework.py` + `proposal.py` + `scenario.py` — cleanest abstractions for our `EnvSpec` and DAG interfaces
4. **Read before tree search upgrade**: `ai-scientist-v2/ai_scientist/treesearch/parallel_agent.py` + `agent_manager.py` — parallel hypothesis expansion, best-first pruning
5. **Read for role decomposition**: `agent-laboratory/agents.py` + `ai_lab_repo.py` — role-split across prompts
6. **Read for minimalist inspiration**: `karpathy-autoresearch/train.py` + `dolphin/launch_dolphin.py` — when scope creep threatens
7. **Read for evolutionary upgrade (post-MVP)**: `openevolve/controller.py` + `iteration.py` + `database.py`
8. **Read for MCTS upgrade (post-MVP)**: `ml-master/agent/mcts_agent.py`
9. **Read for RAG integration (post-MVP)**: `paper-qa/src/paperqa/docs.py` + `core.py`

## Update policy

Each of these was cloned `--depth 1` on 2026-04-21. To refresh:

```bash
cd knowledge/autoresearch_refs
for d in */; do (cd "$d" && git pull --ff-only); done
```

Full clones (`--depth 1` without history) live under `.git/` inside each dir; delete `.git/` if you want pure-code snapshots (but then `git pull` no longer works).

---

## Quick reference table

| # | Name | Size | Paradigm | Stars (Apr 2026) | Why worth reading |
|---|---|---|---|---|---|
| 1 | ai-scientist (v1) | 315 MB | Linear pipeline | — | Historical / generate_ideas pattern |
| 2 | ai-scientist-v2 | 11 MB | BFTS + agent mgr | — | Tree search reference |
| 3 | aide | 2 MB | Solution tree | — | ⭐ Best inner-loop ref |
| 4 | agent-laboratory | 3 MB | Role multi-agent | — | Role decomposition |
| 5 | karpathy-autoresearch | 1 MB | Minimal single-file | — | Minimalism reference |
| 6 | rd-agent | 21 MB | Abstract framework | — | Cleanest abstractions |
| 7 | problem-reductions | 18 MB | Skills-as-stages | — | Developer workflow ref |
| 8 | open-coscientist | 13 MB | LangGraph DAG | — | ⭐⭐⭐ LangGraph template |
| 9 | mlgym | 400 MB | Gym env/agent | 595 | Env abstraction |
| 10 | paper-qa | 60 MB | RAG over PDFs | 8397 | Optional RAG backend |
| 11 | openevolve | 12 MB | Evolutionary search | 6057 | Post-MVP paradigm |
| 12 | ml-master | 26 MB | MCTS | 398 | MCTS upgrade ref |
| 13 | internagent | 17 MB | Stage-based | 1277 | Most complete harness |
| 14 | dolphin | 16 MB | Minimal closed loop | 42 | Readable reference |
| 15 | sciagents | 223 KB | Knowledge-graph agents | 605 | Closest physical-science analog |

Total: ~915 MB. All cloned `--depth 1`. Regenerate via `bash clone_refs.sh`.
