# /review-framework skill — design

**Status:** approved 2026-04-25
**Owner:** Chen Jiahan
**Implements:** a sixth Claude Code skill for the AutoQEC repo

## Motivation

The repo has skills for *running* the loop (`/autoqec-run`), *reviewing the
research narrative* (`/review-log`), *auditing a single round*
(`/diagnose-failure`, `/verify-decoder`), and orchestration helpers
(`/add-env`, `/read-zulip`, `/demo-presenter`). What is missing is a
**framework retrospective**: an agent that reads the artifacts of a
completed run and proposes improvements **to the framework code itself** —
DSL gaps, weak baselines, miscalibrated safety thresholds, prompt drift,
env limitations, tool/orchestration friction — anything that would let the
*next* run reach a higher Δ_LER ceiling.

`/review-log` answers "did this run go well as research?". `/review-framework`
answers "what should we change in the codebase before the next run?".

## Scope

**In scope (v1):**
- Single-run input: one `runs/<id>/`.
- Read-only on the framework. Writes exactly one markdown file.
- LLM-driven: the agent decides what to look at, what counts as a finding,
  and how to organise the report.

**Out of scope (deliberate, YAGNI):**
- Cross-run aggregation (re-invoke the skill per run for now).
- Auto-applied edits or auto-handoff to `superpowers:writing-plans`. The
  output is advisory; the user decides what to do with it.
- A built-in taxonomy of weakness categories or a fixed scoring rubric.
  Anti-pattern explicitly rejected during brainstorming — pre-defined
  categories bias the agent toward what the spec author imagined and away
  from genuinely surprising findings.

## Contract

| Field | Value |
|---|---|
| Slash command | `/review-framework` |
| File | `.claude/skills/review-framework/SKILL.md` |
| Inputs | `run_dir` — `runs/<id>/` |
| Output | `runs/<id>/framework_review.md` (the single file the skill writes) |
| Tools allowed | `Read`, `Grep`, `Glob` everywhere; `Write` only to the output path; `Bash` only for `python -m cli.autoqec review-log <run_dir>` to fetch mechanical stats |
| Tools forbidden | `Edit` / `Write` to any framework source under `autoqec/`, `cli/`, `.claude/agents/`, `autoqec/envs/builtin/`, `docs/contracts/` — the skill is advisory |

## Behaviour (handed to the agent, not prescribed)

1. Read the run's artifacts: `history.jsonl`, `pareto.json`, `log.md`,
   `orchestrator_trace.md`, and any per-round `diagnosis.md` /
   `failure_diagnosis.md` / `verification_report.json`.
2. Cross-read project source freely to ground findings in real code.
3. Identify framework weaknesses or bugs.
4. Write recommendations to `runs/<id>/framework_review.md`.

The skill does NOT prescribe a weakness taxonomy, evidence-to-source-file
mapping, scoring rubric, or output section layout. The agent decides per
run.

## The only content requirement

Every recommendation must include:

1. **What's wrong** — the finding itself, in one or two sentences.
2. **Evidence** — concrete pointers the user can re-open. File paths with
   line numbers (`history.jsonl:42`, `round_07/train.log:118`), trace
   quotes, or pinned source lines. Without this the user has to re-derive
   the agent's reasoning, and the report becomes untrustworthy.
3. **Proposed change** — pinned to a specific file. Prose is acceptable; a
   diff or YAML/code snippet is better. "Improve the DSL" is not actionable.

Everything else — grouping, ranking, executive summary, confidence labels,
whether to bundle related findings — is the agent's call.

## Guardrail

The skill must never recommend weakening seed isolation, the safety
whitelist (`autoqec.runner.safety`), or any reward-hacking guard. These
exist because LLM agents *will* find their way around prose-only
constraints; the framework's defense is physical isolation, not policy.

## Relationship to existing skills

| Skill | Asks |
|---|---|
| `/review-log` | "Was this run coherent as research?" |
| `/diagnose-failure` | "Why did this single round break?" |
| `/verify-decoder` | "Is this checkpoint's Δ_LER real?" |
| `/review-framework` (new) | "What should we change in the codebase before the next run?" |

The naming parallel `/review-log` ↔ `/review-framework` is intentional.

## Acceptance

- `.claude/skills/review-framework/SKILL.md` exists with a frontmatter
  block matching the shape of sibling skills (`name`, `description`).
- The SKILL.md body fits on roughly one screen — anything longer is a
  signal we've started prescribing the analysis again.
- Manual smoke test on an existing `runs/<id>/` produces a
  `framework_review.md` whose findings each carry the three required
  pieces (problem, evidence pointers, proposed change pinned to a file).
