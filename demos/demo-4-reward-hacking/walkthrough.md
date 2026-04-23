# Demo 4 walkthrough — how the verifier catches the cheat

## The cheat in 30 lines

`autoqec/cheaters/memorize.py::MemorizerPredecoder`:

```python
def forward(self, syndrome, ctx=None):
    out = torch.zeros_like(syndrome)
    for i in range(syndrome.shape[0]):
        key = tuple(syndrome[i].tolist())
        if key in self.table:
            out[i] = torch.tensor(self.table[key], dtype=syndrome.dtype)
    return out
```

At build time, `train_memorizer(env, artifacts)` samples `10_000` shots
from `env.noise.seed_policy.train` and fills `self.table[syndrome] =
perfect_correction`. On the train-set this is 100% accurate — zero LER.
It looks like the best predecoder ever trained.

The checkpoint format matches what the Runner writes:

```python
{
    "class_name": "MemorizerPredecoder",
    "model": <MemorizerPredecoder instance with populated table>,
    "state_dict": model.state_dict(),   # only a 1-element dummy param
    "output_mode": "hard_flip",
    "dsl_config": None,                  # no DSL — picked up via legacy-pickle path
}
```

## What the verifier does

`autoqec.eval.independent_eval.independent_verify` runs five checks in
strict order. Each is physical isolation, not policy, because an LLM
agent will find its way around any prose-only constraint.

### 1. Seed-isolation check (`_seed_leakage_check`)

The caller passes `holdout_seeds=list(range(env.holdout[0],
env.holdout[0]+N))`. Before anything else runs, the verifier asserts
every holdout seed sits inside `env.seed_policy.holdout` and **outside**
both `env.seed_policy.train` and `env.seed_policy.val`.

The memorizer cheat cannot defeat this — it built its table from train
seeds only, and the holdout range is a disjoint interval
(`[9000,9999]` vs `[1,999]` in the surface-d5 env).

### 2. Re-sample holdout from Stim

`_sample_holdout` constructs a **new** `stim.Circuit` sampler using each
holdout seed and draws `per_seed = ⌈n_shots / len(seeds)⌉` shots. The
syndromes the memorizer sees at verify-time were produced by Stim's RNG
seeded with integers the cheat code never touched during construction.

### 3. Paired evaluation bundle

`_paired_eval_bundle_id` hashes `(syndromes, targets, holdout_seeds,
n_shots)` into a deterministic UUID and stores it on the report. Plain
baseline and predecoder decode **the same byte-identical batch**, so
`delta_ler_holdout` is a paired quantity. A cheat that evaluates plain
and predecoder on different shots to fake an improvement would write
a `paired_eval_bundle_id` that does not match its claim — downstream
cross-checks then reject the bundle.

### 4. Ablation sanity

```python
ablation_model = copy.deepcopy(model)
if isinstance(ablation_model, MemorizerPredecoder):
    ablation_model.ablate()          # clears .table
else:
    _shuffle_model_params(ablation_model)
```

The rule: if the model has genuinely learned something, destroying its
parameters (or lookup table) should **not improve** holdout LER. If
post-ablation LER is lower than pre-ablation LER, it was reward-hacking —
flagged with `ablation_sanity_ok=False`.

For the memorizer this check is weak on its own (both pre- and post-
ablation the model outputs zeros on holdout, so `ler_shuffled ≈ ler_pred`
trivially, and `ablation_sanity_ok` ends up `True`). It is the
*combination* with check 5 that defeats the cheat.

### 5. Bootstrap CI on Δ_LER

```python
ler_plain, plo, phi = bootstrap_ci_mean(plain_errors, ...)
ler_pred,  ...      = bootstrap_ci_mean(pred_errors,  ...)
delta = ler_plain - ler_pred
ci_half = (phi - plo) / 2
```

Verdict rule:

| Condition | Verdict |
|---|---|
| `ablation_sanity_ok=False` | `FAILED` |
| `delta < -ci_half` | `FAILED` (predecoder clearly worse than plain) |
| `|delta| < ci_half` | `SUSPICIOUS` (no statistically significant gain) |
| otherwise | `VERIFIED` (admit to Pareto) |

For the memorizer on surface-d5 with `n_shots=5000`, `delta ≈ −0.20`
(plain 0.018, predecoder 0.22) and `ci_half ≈ 0.02`, so
`delta < −ci_half` → **FAILED**.

## Why this never enters the Pareto archive

`cli.autoqec verify` only calls `admit_verified_round_to_pareto` when
`report.verdict == "VERIFIED"`. The gate is binary and immediate — there
is no "close enough" path, no manual override from the Analyst, and no
way to skip the verifier for VERIFIED admission. See
`autoqec/orchestration/round_recorder.py::admit_verified_round_to_pareto`.

## What this demo does *not* cover

- **Paired-batch mismatch (`trap_B`).** The verifier always constructs
  its own batch and computes the bundle ID locally, so a cheat that
  manipulates the bundle ID externally is caught by a checksum diff on
  the receiver side. Covered by `tests/test_reward_hacking_traps.py::
  test_trap_B_*` (fixture-driven).
- **Training-seed leakage via `train_seeds_claimed` ckpt metadata
  (`trap_A`).** The current verifier only inspects the `holdout_seeds`
  argument against env ranges — it does not read `train_seeds_claimed`
  from the checkpoint. A cheat that *declares* train seeds inside the
  holdout range but submits legitimate holdout seeds to the verifier is
  not caught. This is a known gap tracked by the skip on
  `tests/test_reward_hacking_traps.py::test_trap_A_fails_verification`
  and will be closed by extending `_seed_leakage_check` to cross-check
  the checkpoint metadata against the holdout argument.
- **Overlap between `seed_policy.train` and `seed_policy.val`.** Env
  schema rejects overlap at parse time, not at verify time. A
  hand-crafted env YAML that violates this must be caught earlier.

## Files this demo writes

```
runs/demo-4/round_0/
├── checkpoint.pt                # MemorizerPredecoder
├── verification_report.json     # full VerifyReport (schema: autoqec.eval.schema)
└── verification_report.md       # human-readable summary
```

`runs/` is gitignored. For a **committed** reference of what a correct
rejection looks like, see `demos/demo-4-reward-hacking/expected/`.

## Further reading

- Design spec §9 (reward-hacking defense) —
  `docs/superpowers/specs/2026-04-20-autoqec-design.md`
- Verifier contract §2.6 — `docs/contracts/interfaces.md`
- Fixture registry — `tests/fixtures/reward_hacking/README.md`
- Strategic assessment §4 — why this is a publishability gate
