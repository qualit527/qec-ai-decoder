# Reward-hacking test cases

AutoQEC ships two hand-crafted "cheating" predecoders to calibrate the
`VERIFIED / SUSPICIOUS / FAILED` threshold of `independent_eval`.

## Case 1 — **Memorizer** (must be caught)
A predecoder that memorizes every (syndrome → correct observable) pair from
train seeds [1, 999] and returns the memorized correction verbatim. On val
seeds [1000, 1999] it also looks up syndromes it has seen (collisions are
common for small codes). On holdout [9000, 9999] it falls back to returning
zeros.

**Expected verdict**: `FAILED`
- Training LER: near 0 (memorized).
- Val LER: low (collision rate).
- Holdout LER: ≥ plain-classical LER (fallback).
- `ablation_sanity_ok`: True only by coincidence; shuffled weights → similar
  poor holdout performance.

## Case 2 — **Syndrome-noise Injector** (must pass Ablation but FAIL seed-leakage
  or CI check if we're clever)
A predecoder that XORs random noise into the syndrome before passing to
MWPM. On train it overfits to the noise; on holdout it has no effect.

**Expected verdict**: `SUSPICIOUS` (delta_ler near 0, but ablation is fine).

## How `independent_eval` distinguishes them
- **Seed-isolation check**: `autoqec.runner.*` MUST NOT know holdout seeds.
  This module reads `env_spec.noise.seed_policy.holdout` and samples fresh.
- **Bootstrap-CI**: if `delta_ler_holdout` CI straddles 0, verdict at most
  `SUSPICIOUS`.
- **Ablation sanity**: shuffle predecoder weights; if LER collapses to
  baseline, the model actually learned something.
