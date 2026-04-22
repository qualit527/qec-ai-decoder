# CI Coverage Design (Phase 1)

## Goal

Add a minimal code coverage report to the existing GitHub Actions CI pipeline so each run shows how much code is exercised by the current non-integration test suite.

The first version should:

- keep the existing CI structure intact
- reuse the current `make test` entry point
- print a coverage percentage and per-file summary in CI logs

It should not add external services, dashboards, or enforce a minimum threshold yet.

## Scope

### In scope

- add `pytest-cov` to the development dependencies
- update the `make test` command to emit coverage output
- keep `.github/workflows/ci.yml` unchanged except insofar as it already runs `make test`
- report coverage for the main Python code under `autoqec` and `cli`

### Out of scope

- Codecov or other third-party reporting services
- PR comment bots
- HTML or XML coverage artifacts
- `fail-under` thresholds
- integration-test coverage

## Current Repository Constraints

The repository already has a stable CI path:

- local and CI both rely on `make test`
- CI already creates `.venv`, installs `.[dev]`, and runs `make test`
- `make test` currently runs `pytest tests/ -m "not integration" -v`

This means the lowest-risk approach is to extend `make test` instead of adding a second coverage-specific CI command.

## Proposed Approach

Use `pytest-cov` and extend the existing test command to print a terminal coverage summary.

Recommended command shape:

```bash
pytest tests/ -m "not integration" -v \
  --cov=autoqec \
  --cov=cli \
  --cov-report=term-missing:skip-covered
```

## Why This Approach

- smallest possible change to the current workflow
- local development and CI continue to use the same command
- no extra CI step is required
- developers can immediately see total coverage and the least-covered files in the CI logs

## Reporting Behavior

The coverage report should:

- show a total coverage percentage
- show file-level uncovered lines for files that are not fully covered
- avoid clutter from files that are already fully covered

Using `term-missing:skip-covered` is preferred for the first version because it keeps the report readable while still highlighting gaps.

## Target Files

Coverage should include:

- `autoqec`
- `cli`

Coverage should not target:

- `tests`
- `docs`
- generated runtime outputs such as `runs/`

## CI Behavior

No new workflow step is needed.

The existing workflow:

1. installs dev dependencies
2. runs `make test`

After this change, the same step will automatically print coverage information into the GitHub Actions log.

## Failure Model

The CI job should still fail only when tests fail or the test command itself errors.

Coverage percentage by itself should not fail CI in this phase.

This is important because the team does not yet have an agreed baseline or target threshold.

## Design Choices

### Why not use `coverage.py` directly

`coverage.py` would work, but it would require either a second command or more custom scripting. `pytest-cov` is simpler because it integrates directly with the existing pytest entry point.

### Why not add a threshold now

A threshold before measuring the natural baseline would be arbitrary and likely disruptive. The first step is visibility, not enforcement.

### Why not upload artifacts now

Artifacts are useful later, but the first requirement is simply “show coverage in CI.” Text output already satisfies that with the least complexity.

## Expected Outcome

After this change:

- local `make test` will show coverage
- GitHub Actions logs will show coverage
- the team can discuss future thresholds using real numbers

## Future Extensions

Likely next steps after Phase 1:

1. add `coverage.xml` output as a CI artifact
2. define a baseline and introduce `--cov-fail-under=<N>`
3. optionally split unit-test coverage and integration-test coverage
4. optionally publish coverage to a third-party service if the team wants historical tracking
