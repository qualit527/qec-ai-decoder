---
name: review-framework
description: Read the artifacts of a completed AutoQEC run and propose improvements to the framework code itself — DSL gaps, weak baselines, miscalibrated safety thresholds, prompt drift, env limitations, orchestration friction. Advisory only; never edits framework files. Use after a run completes when the user asks "what should we change before the next run?" or feels the system is plateauing.
---

# /review-framework

## When to use
- A run has completed and the user wants to know what to **change in the codebase** before the next run.
- The user feels Δ_LER is plateauing across runs and asks what is holding the framework back.

This is the framework-code counterpart to `/review-log` (which reviews the
research narrative). For single-round bug triage use `/diagnose-failure`;
for checkpoint audits use `/verify-decoder`.

## Inputs
- `run_dir` — path to `runs/<id>/`.

## Goal
Read everything in `run_dir` that documents what happened, cross-read the
project source freely, find weaknesses or bugs in the current framework,
and propose improvements. The shape of the analysis — what to look for,
how to group findings, how to rank them — is yours to decide per run.

Suggested starting points (not a checklist):
- `history.jsonl`, `pareto.json`, `log.md`, `orchestrator_trace.md`
- per-round artifacts under `round_<N>/` (`config.yaml`, `train.log`,
  `metrics.json`, any `diagnosis.md` / `failure_diagnosis.md` /
  `verification_report.json`)
- `python -m cli.autoqec review-log <run_dir>` for mechanical stats
- project source under `autoqec/`, `cli/`, `.claude/agents/`,
  `autoqec/envs/builtin/`, `docs/contracts/`

## Output
Write `runs/<id>/framework_review.md`. Structure is your call.

Each finding must include:
1. **What's wrong** — one or two sentences.
2. **Evidence** — concrete pointers the user can re-open: file paths with
   line numbers (`history.jsonl:42`, `round_07/train.log:118`), trace
   quotes, or pinned source lines.
3. **Proposed change** — pinned to a specific file. A diff or YAML/code
   snippet is better than prose. "Improve the DSL" is not actionable.

A finding without all three is unfinished — drop it or finish it.

## Tool-use rules
- `Read`, `Grep`, `Glob` — anywhere.
- `Write` — only to `runs/<id>/framework_review.md`.
- `Bash` — only `python -m cli.autoqec review-log <run_dir>`.
- Never `Edit` or `Write` framework source under `autoqec/`, `cli/`,
  `.claude/agents/`, `autoqec/envs/builtin/`, or `docs/contracts/`. The
  user applies recommendations themselves.

## Guardrail
Never recommend weakening seed isolation, the safety whitelist
(`autoqec.runner.safety`), or any reward-hacking guard. These are
physical-isolation defenses against LLM agents that find their way around
prose-only constraints; loosening them defeats the purpose of the project.
