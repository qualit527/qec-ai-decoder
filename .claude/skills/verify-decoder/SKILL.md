---
name: verify-decoder
description: Audit a predecoder checkpoint against holdout seeds. Runs independent_eval (3 fair-baseline guards) and interprets borderline cases with LLM reasoning. Use when a round produces a promising Δ_LER and the user wants to confirm it is not a reward-hacking artifact.
---

# /verify-decoder

## When to use
- An AutoQEC round produced `delta_ler > 0` and the user wants final sign-off.
- User asks to "verify this checkpoint" or "audit this round".

## Inputs
- `round_dir`: path to `runs/<id>/round_N/` (must contain `checkpoint.pt` + `config.yaml`)
- `env_yaml`: env used by the round (defaults to `config.yaml`'s `env_name`)

## Behavior
1. Run `python -m cli.autoqec verify <round_dir> --env <env_yaml>`.
2. Read `verification_report.json` + `training.log`.
3. LLM-reason over:
   - If `verdict=VERIFIED`: write one paragraph noting key confidence intervals and whether delta is within ablation threshold. Output "APPROVED".
   - If `verdict=SUSPICIOUS`: read training.log, the DSL config, the history of previous rounds; diagnose whether this is (a) genuine small improvement, (b) overfitting, (c) reward-hacking. Recommend next action: accept, re-run with more shots, or reject.
   - If `verdict=FAILED`: produce a diagnostic report matching the failure pattern (seed leak, ablation failure, negative delta). Archive into `round_N/failure_diagnosis.md`.

## Tool-use rules
- Read: training.log, config.yaml, verification_report.md, previous round metrics.json files.
- Bash: `python -m cli.autoqec verify ...` only. No other commands.

## Output
- Short decision block: APPROVED / SUSPICIOUS_KEEP / SUSPICIOUS_REJECT / FAILED_REJECT.
- Diagnostic paragraph saved to `round_N/decision.md`.
