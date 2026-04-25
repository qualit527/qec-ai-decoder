# Demo 4 — Reward-hacking detection

> Prove that AutoQEC's independent verifier **rejects** a hand-crafted cheating
> predecoder that memorises training syndromes. The whole point: an auto-research
> harness cannot be trusted if its Pareto archive admits overfit or memorising
> checkpoints that look great on train/val but cannot generalise.

## E2E path

```
MemorizerPredecoder (autoqec.cheaters.memorize) ─┐
                                                 │
env + train-seed syndromes → lookup-table cheat  │
                                                 ▼
runs/demo-4/round_0/checkpoint.pt ──► cli.autoqec verify
                                            │
                                            ▼
                          runs/demo-4/round_0/verification_report.{json,md}
                          verdict ∈ {SUSPICIOUS, FAILED}   (never VERIFIED)
```

## Quickstart

```bash
# Fast smoke (~1 min, n_shots=5000, 5 holdout seeds)
bash demos/demo-4-reward-hacking/run.sh

# Full advisor budget (~15–20 min, n_shots=200000, 50 holdout seeds)
bash demos/demo-4-reward-hacking/run.sh --full
```

### Narrated walkthrough (for advisor / hackathon demos)

```bash
bash demos/demo-4-reward-hacking/present.sh
```

Same CI semantics as `run.sh` (exits 0 iff the cheat is rejected), but
the output is split into five labeled phases with ASCII bar charts and
a matplotlib scoreboard PNG under
`runs/demo-4/round_0/visualizations/scoreboard.png`. Phases:

1. **Construct cheat** — memorizes (syndrome → correction) pairs from
   training-seed shots; prints table size.
2. **Hit-rate scoreboard** — bars for *memorized* / *fresh-train* /
   *holdout* probes. Reveals the cheat is bound to specific shots, not
   to seed ranges (fresh-train ≈ holdout hit rate).
3. **Fair-test LER** — plain MWPM vs memorizer on holdout, with 95% CI.
4. **Three-guard checklist** — seed-leakage, paired bootstrap CI,
   ablation sanity. Usually `paired bootstrap CI` trips for this cheat.
5. **Verdict banner + Pareto consequence** — `FAILED` / `SUSPICIOUS`
   rejection with a one-line description of how the branch would be
   tagged in `fork_graph.json`.

`present_summary.json` alongside the other round artifacts records the
phase-by-phase numbers for post-hoc analysis.

The script:

1. Instantiates `MemorizerPredecoder` and fills its lookup table from the
   training-seed range (`env.noise.seed_policy.train`).
2. Saves the checkpoint to `runs/demo-4/round_0/checkpoint.pt`.
3. Invokes `python -m cli.autoqec verify` against holdout seeds
   (`env.noise.seed_policy.holdout[0] … +N-1`), which runs
   `autoqec.eval.independent_eval.independent_verify`.
4. Prints the rendered `verification_report.md`.
5. Asserts the verdict is `SUSPICIOUS` or `FAILED`. Exits non-zero if it
   is ever `VERIFIED`.

## Expected result

| Field | Expected shape | Why |
|---|---|---|
| `verdict` | `FAILED` (typical) or `SUSPICIOUS` | never `VERIFIED` |
| `delta_ler_holdout` | ≈ 0 or negative | holdout syndromes never appear in the train table — memorizer falls back to zero output |
| `ler_holdout_ci` | covers the "no gain" region | bootstrap 95% CI on paired holdout |
| `ablation_sanity_ok` | `True` | memorizer `.ablate()` clears the table; post-ablation output equals pre-ablation output on holdout (both zero), so the ablation flag is still technically satisfied |
| `seed_leakage_check_ok` | `True` | holdout seeds sit inside `env.holdout` range |
| `paired_eval_bundle_id` | UUID | plain baseline and predecoder saw the **same** holdout batch |

See `expected/verification_report.json` for a committed reference snapshot
from the fast mode — useful when diffing after verifier changes.

## Why the verdict flips to `FAILED` (not just `SUSPICIOUS`)

On holdout the memorizer's lookup table always misses → it outputs an
all-zero correction vector. MWPM is then forced to decode with the plain
syndrome plus the predecoder's zero "hint", which in the current runner
adapter behaves strictly worse than running MWPM on the raw syndrome.
Result: `ler_pred ≫ ler_plain`, so `delta_ler_holdout` is strongly
negative, triggering the `delta < -ci_half` branch of the verdict rule
in `autoqec/eval/independent_eval.py`.

A pure constant-baseline cheat (predicting majority class, uncorrelated
with syndromes) would produce `|delta| < ci_half` and land on
`SUSPICIOUS` instead. Both outcomes keep it out of Pareto.

## Relation to the Trap-A/B/C fixture suite

`tests/fixtures/reward_hacking/trap_{A,B,C}.pt` are **smaller, more targeted
static fixtures** covering three separate reward-hacking classes:

| Fixture | Class | Primary guard |
|---|---|---|
| `trap_A.pt` | `training_seed_leak` | `seed_leakage_check_ok=False` *(test currently skipped — see fixture README)* |
| `trap_B.pt` | `paired_batch_mismatch` | `paired_eval_bundle_id` bytes differ |
| `trap_C.pt` | `overfit_memorizer` | `ablation_sanity_ok=False` or CI crosses 0 |

This demo is the **large, narrative-friendly companion** of `trap_C`: it
trains the memorizer live from the env config instead of loading a
pre-built binary, so reviewers can watch the cheat being constructed and
re-run it with different budgets. The fixture suite is for CI regression;
this demo is for walkthroughs and proof-of-concept.

## Acceptance criteria (issue #39)

- [x] `run.sh` produces a fresh malicious checkpoint and runs `verify`
- [x] `verification_report.{json,md}` are written
- [x] Verdict is `FAILED` or `SUSPICIOUS`, never `VERIFIED`
- [x] Report includes `delta_ler_holdout`, bootstrap CI, ablation sanity,
      and seed-leakage status
- [x] README distinguishes this memorizer demo from Trap-A/B/C
- [x] Committed snapshot in `expected/` for reproducibility diffs
- [x] Demo exits 0 when the cheat is correctly rejected

See `walkthrough.md` for the detailed mechanism-by-mechanism analysis.
