# Public API JSON fixtures

These fixtures record small, reviewable input/output contracts for public
interfaces that tests currently verify in three places:

- CLI discoverability contracts
- Pydantic schema field inventories
- Artifact manifest payload validation

## Rules

- Keep each fixture human-reviewable and scoped to one public contract.
- Prefer exact field/value contracts for stable APIs and artifacts.
- Use smoke fixtures only for output that is intentionally partial, such as
  `--help` discoverability checks.
- Add both the fixture file and a consuming test in the same change.
- Update `manifest.json` when adding or removing a fixture.

## Stability labels

- `contract`: exact public contract that downstream tests should treat as
  stable unless the interface intentionally changes.
- `smoke`: lightweight snapshot or substring contract that verifies
  discoverability without freezing every byte of output.
