---
name: review-log
description: Read an entire runs/<id>/log.md and assess research narrative coherence, identify stuck hypotheses, detect overfitting signs, and write a review markdown. Use after a full run completes or when a loop has been stuck for many rounds.
---

# /review-log

## When to use
- A run of 10+ rounds has completed.
- User asks for a "research review" or "is the agent stuck?"

## Inputs
- `run_dir`: path to `runs/<id>/`

## Behavior
1. Run `python -m cli.autoqec review-log <run_dir>` to get structured stats (round count, Pareto size, top hypotheses, killed_by_safety count).
2. Read `log.md` fully.
3. LLM-reason:
   - Narrative coherence: does each round build on previous findings?
   - Stuck patterns: ≥3 rounds with near-identical hypotheses?
   - Overfitting signs: Δ_LER monotonically improving on train seeds but not holdout?
   - Safety-kill clustering: are VRAM or wall-clock kills bunched around one hypothesis family?
4. Write `runs/<id>/review.md` with: summary (5 sentences), top 3 concerns, recommended next actions.

## Output
- Path to `review.md`.
