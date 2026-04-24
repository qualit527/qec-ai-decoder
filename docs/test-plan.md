# Test Plan

This repository has two intentionally separate test layers:

- Default gate: `make lint` and `make test`
- Integration layer: `make test-integration`

The default pytest path also skips tests marked `slow`. Use `--run-slow`
to opt back into those wall-clock-heavy checks when you need their unique
coverage.

## Canonical integration entrypoint

Use the Make target below as the stable repository-level entry:

```bash
make test-integration
```

It expands to the existing pytest contract:

```bash
pytest tests/ -m "integration" -v --run-integration
```

The `integration` marker selects end-to-end tests, and `--run-integration`
is required by `tests/conftest.py` to opt into running them.

## CI policy

The default pull-request gate stays unchanged:

```bash
make lint
make test
```

By default, both local `make test` and CI's `make coverage` skip
`@pytest.mark.slow` tests. Integration tests remain manual for now. The
repository does not run them in the default GitHub Actions workflow unless
that workflow is explicitly expanded in a later change.

## Canonical slow-test opt-in

Use the pytest flag below whenever you want the slower subprocess-heavy
checks back in scope:

```bash
pytest tests/ -m "not integration" -v --run-slow
```

`tests/conftest.py` enforces this opt-in even if you pass
`-m "not integration"` explicitly, so the wall-clock budget for the default
gate stays stable.

## Live-LLM resume

The live-LLM path supports resuming an existing run directory:

```bash
python -m cli.autoqec run <env-yaml> --run-dir runs/<run_id>
```

Resume treats round `N` as complete only when both conditions hold:

- `history.jsonl` contains a row with `round == N`
- `round_N/metrics.json` exists and contains parseable JSON

If both surfaces also carry a non-empty `round_attempt_id`, the IDs must
match. There is no separate marker file. In practice `history.jsonl` is the
orchestration source of truth and `metrics.json` proves the Runner finished
the same attempt. Linux SIGINT coverage lives in
`tests/test_llm_loop_resume.py`; Windows should exercise the same helper
semantics without relying on POSIX signal delivery.

## When to run integration tests

Run `make test-integration` before merging changes that affect:

- `autoqec.runner`
- `cli/autoqec.py`
- `autoqec.orchestration.subprocess_runner`
- worktree dispatch, artifact layout, or end-to-end demo paths

Also run it before recording demo artifacts or diagnosing failures that
only reproduce in a full round execution path.

## Discoverability

- `Makefile` exposes `test-integration` as the canonical command
- `tests/conftest.py` requires `--run-slow` before wall-clock-heavy tests run
- `README.md` points contributors here for integration guidance
- `AGENTS.md` points coding agents here for the same contract
