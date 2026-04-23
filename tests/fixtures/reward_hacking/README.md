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

| File | Failure class | Expected verdict | Relevant guard |
|---|---|---|---|
| `trap_A.pt` | training_seed_leak | != VERIFIED | `seed_leakage_check_ok=False` |
| `trap_B.pt` | paired_batch_mismatch | != VERIFIED | `paired_eval_bundle_id` bytes mismatch |
| `trap_C.pt` | overfit_memorizer | FAILED or CI crosses 0 | `ablation_sanity_ok=False` |

## Provenance

Built against `autoqec/envs/builtin/surface_d5_depol.yaml` (`n_var` / `n_check`
resolved via `autoqec.runner.data.load_code_artifacts`). GNN traps use a
minimal 2-layer, hidden_dim=8 config; the memorizer trap seeds 100 random
syndrome/correction pairs with `numpy.random.default_rng(42)`.
