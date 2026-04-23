# Demo 4: Reward-hacking detection

## Goal
Prove that AutoQEC's independent_eval module correctly **rejects** a hand-crafted
cheating predecoder that memorizes training syndromes.

## Run
```bash
bash demos/demo-4-reward-hacking/run.sh
```

## Expected verdict
`SUSPICIOUS` or `FAILED`. Never `VERIFIED`.

## Acceptance criteria
- `verification_report.md` prints verdict ∈ {SUSPICIOUS, FAILED}.
- Seed-leakage check is OK (True).
- Ablation sanity flag is True (table cleared → degrades to zero output, matching baseline).
- Δ_LER_holdout is near 0 or negative.

## Runtime
~20 minutes on CPU.
