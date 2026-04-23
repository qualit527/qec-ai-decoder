---
name: diagnose-failure
description: Inspect a stalled or failed run, identify root cause (bad hyperparameter / NaN pattern / OOM / env misconfig), and recommend a fix. Does NOT apply fixes autonomously.
---

# /diagnose-failure

## When to use
- A round's status was `killed_by_safety`, `compile_error`, or `train_error`.
- User asks "why did this break?" or provides a bad config.

## Inputs
- `run_dir` OR `round_dir`

## Behavior
1. Run `python -m cli.autoqec diagnose <run_dir>` for mechanical stats.
2. Read `train.log`, `config.yaml`, `metrics.json`, the latest 2–3 rounds of `history.jsonl`.
3. LLM-reason over known failure modes:
   - `killed_by_safety` + wall_clock → params too large for budget → suggest smaller `hidden_dim` / `layers`.
   - `killed_by_safety` + NaN rate → loss instability → suggest lower learning rate, gradient clipping, or switch `loss` to `focal`.
   - `compile_error` → DSL syntax → show offending field + schema expected value.
   - `train_error` + VRAM → OOM → suggest smaller batch_size or profile=dev.
4. Write `round_N/diagnosis.md` with:
   - Root cause (1 sentence)
   - Evidence citation (log line numbers)
   - Recommended fix (as a patched DSL YAML snippet)
   - **Do NOT apply** the fix — the user runs it themselves.

## Tool-use
- Read only. Never Edit or Write outside the `diagnosis.md` file.
