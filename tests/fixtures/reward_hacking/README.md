# Reward-hacking fixtures

Binary checkpoint fixtures produced by `scripts/build_trap_fixtures.py`. Each
file simulates one class of reward-hacking failure the verifier must catch
(spec Sec. 9 + test plan Phase 5.2).

## Regenerate

```bash
make build-trap-fixtures
```

Requires the full env (`pip install -e '.[dev]'` with torch, stim, pymatching).
Commit the resulting `.pt` files **and** `manifest.json` to the repo — fixtures
are small (~KB) and deterministic under the build script.

## Files

| File | Failure class | Expected verdict | Relevant guard | Test status |
|---|---|---|---|---|
| `trap_A.pt` | training_seed_leak | != VERIFIED | `seed_leakage_check_ok=False` | **skipped** — verifier does not yet cross-check `train_seeds_claimed` metadata against `holdout_seeds`. See skip reason on `tests/test_reward_hacking_traps.py::test_trap_A_fails_verification`. |
| `trap_B.pt` | paired_batch_mismatch | != VERIFIED | `paired_eval_bundle_id` bytes mismatch | covered |
| `trap_C.pt` | overfit_memorizer | FAILED or CI crosses 0 | `ablation_sanity_ok=False` or `delta_ler_holdout` < -CI half-width | covered |

## Provenance

Built against `autoqec/envs/builtin/surface_d5_depol.yaml` (`n_var` / `n_check`
resolved via `autoqec.runner.data.load_code_artifacts`). GNN traps use a
minimal 2-layer, hidden_dim=8 config; the memorizer trap seeds 100 random
syndrome/correction pairs with `numpy.random.default_rng(42)`.

## Relation to demos/demo-4-reward-hacking

`demos/demo-4-reward-hacking/` is the **narrative-friendly companion** of
`trap_C.pt`: it trains a memorizer live from the env config and runs
`cli.autoqec verify` end-to-end so reviewers can watch the cheat being
constructed and rejected. The fixtures in this directory are for CI
regression (fast, deterministic, no live training); the demo is for
walkthroughs, proof-of-concept, and diffing verifier output against the
snapshot at `demos/demo-4-reward-hacking/expected/verification_report.json`.

When the verifier changes and one of these fixtures starts returning
`VERIFIED`, re-run the demo and compare its snapshot as well.
