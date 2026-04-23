# CI Coverage Design (Phase 1)

## Goal

Add a minimal code coverage report to the existing GitHub Actions CI pipeline so each run shows how much code is exercised by the current non-integration test suite.

The first version should:

- keep the existing CI structure intact
- keep local `make test` focused on fast non-integration runs
- add an explicit coverage entry point for CI and opt-in local use
- print a coverage percentage and per-file summary in CI logs

It should not add external services, dashboards, or enforce a minimum threshold yet.

## Scope

### In scope

- add `pytest-cov` to the development dependencies
- add a `make coverage` command that enables coverage explicitly
- update `.github/workflows/ci.yml` to call the coverage-aware target
- move shared pytest defaults and coverage settings into `pyproject.toml`
- report coverage for the main Python code under `autoqec` and `cli`

### Out of scope

- Codecov or other third-party reporting services
- PR comment bots
- HTML or XML coverage artifacts
- `fail-under` thresholds
- integration-test coverage

## Current Repository Constraints

The repository already has a stable CI path:

- local development already relies on `make test`
- CI already creates `.venv` and installs `.[dev]`
- the repository already uses `pyproject.toml` for pytest markers and Python path setup

The main trade-off is local ergonomics versus command reuse. Always attaching coverage to `make test` keeps one command everywhere, but it makes the default local test path slower and noisier.

## Proposed Approach

Use `pytest-cov`, keep `make test` as the fast default command, and add an explicit `make coverage` target for CI and opt-in local runs.

Recommended command shape:

```bash
make test
make coverage
```

Supporting configuration:

- `pyproject.toml` owns the default pytest selection (`tests`, `-m "not integration"`, `-v`)
- `pyproject.toml` owns coverage source and reporting behavior
- `make coverage` only turns coverage collection on

## Why This Approach

- local development keeps a fast, lower-noise default command
- CI still uses a single Make target
- pytest and coverage defaults live in configuration instead of being duplicated on the command line
- developers can immediately see total coverage and the least-covered files in the CI logs

## Reporting Behavior

The coverage report should:

- show a total coverage percentage
- show file-level uncovered lines for files that are not fully covered
- avoid clutter from files that are already fully covered

Using `show_missing = true` and `skip_covered = true` in coverage config keeps the report readable while still highlighting gaps.

## Target Files

Coverage should include:

- `autoqec`
- `cli`

Coverage should not target:

- `tests`
- `docs`
- generated runtime outputs such as `runs/`

## CI Behavior

The existing workflow:

1. installs dev dependencies
2. runs `make coverage`

After this change, CI prints coverage information into the GitHub Actions log, while local developers can still run `make test` without coverage overhead.

## Failure Model

The CI job should still fail only when tests fail or the test command itself errors.

Coverage percentage by itself should not fail CI in this phase.

This is important because the team does not yet have an agreed baseline or target threshold.

## Design Choices

### Why not use `coverage.py` directly

`coverage.py` would work, but it would require either a second command or more custom scripting. `pytest-cov` is simpler because it integrates with pytest while still allowing a dedicated `make coverage` entry point.

### Why not add a threshold now

A threshold before measuring the natural baseline would be arbitrary and likely disruptive. The first step is visibility, not enforcement.

### Why not upload artifacts now

Artifacts are useful later, but the first requirement is simply “show coverage in CI.” Text output already satisfies that with the least complexity.

## Expected Outcome

After this change:

- local `make test` will stay focused on fast non-integration runs
- local `make coverage` will show coverage on demand
- GitHub Actions logs will show coverage
- the team can discuss future thresholds using real numbers

## Future Extensions

Likely next steps after Phase 1:

1. add `coverage.xml` output as a CI artifact
2. define a baseline and introduce `--cov-fail-under=<N>`
3. optionally split unit-test coverage and integration-test coverage
4. optionally publish coverage to a third-party service if the team wants historical tracking
