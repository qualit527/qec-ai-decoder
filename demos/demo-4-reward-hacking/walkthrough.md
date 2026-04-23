# Demo 4 walkthrough

## What this demo proves
The minimum-viable MVP guarantee: any predecoder that looks good on train/val
but cannot generalize to fresh holdout seeds will NOT be admitted into the
Pareto front.

## Why this matters for publishability
Without this guard, auto-research harnesses can produce "great results" that
are actually hidden memorization — an artifact, not a decoder. See
STRATEGIC_ASSESSMENT.md §4.

## What to look at in the output
- `verification_report.md`: verdict + evidence.
- `verification_report.json`: full JSON with CIs and ablation data.
