# AGENTS.md

Developer guidance for Codex and similar coding agents when working on this
repository. For project overview, goals, and demos, see `README.md`.

## Build & Test

```bash
make install          # install project + dev dependencies into .venv
make lint             # ruff check autoqec cli tests scripts
make test             # default non-integration pytest run
make coverage         # pytest with coverage for autoqec + cli
pytest tests/ -m "not integration" -v
pytest tests/test_dsl_schema.py -v
pytest tests/test_runner_smoke.py -v -m integration --run-integration
```

## Make Targets

```bash
make run              # run cli.autoqec with env/profile defaults
make run-nollm        # run cli.autoqec without LLM-backed subagents
make demo-2           # run demos/demo-2-bb72/run.sh
make run-all-claude   # force Claude-backed ideator/coder
make run-cheap        # swap ideator model to a cheaper default
```

Key environment knobs:

- `ENV`: env YAML passed to `cli.autoqec run`
- `ROUNDS`: number of rounds for `make run`
- `PROFILE`: `dev` or `prod`
- `AUTOQEC_*_{BACKEND,MODEL}`: backend/model routing for the agent roles

## Architecture

### Core directories

| Path | Purpose |
|------|---------|
| `autoqec/envs/` | `EnvSpec` schema and builtin environment YAML loaders |
| `autoqec/decoders/` | DSL schema/compiler, backend adapters, neural modules, seed templates |
| `autoqec/runner/` | training/eval loop, artifacts, FLOPs, safety checks, round metrics |
| `autoqec/orchestration/` | lightweight loop/memory helpers used to assemble prompts and carry run state |
| `autoqec/tools/` | analysis helpers like `machine_state` |
| `cli/autoqec.py` | click CLI entrypoint |
| `tests/` | pytest suite; integration tests require explicit opt-in |
| `docs/superpowers/` | design specs and implementation plans |

### Main execution path

`cli.autoqec run` loads an env YAML, picks a predecoder config, calls the
Runner, and writes run artifacts under `runs/<timestamp>/`.

`cli.autoqec run-round` is the lower-level entrypoint for executing one round
from a specific env/config/round-dir tuple.

### Important schemas

- `autoqec.envs.schema.EnvSpec`
- `autoqec.runner.schema.RunnerConfig`
- `autoqec.runner.schema.RoundMetrics`
- `autoqec.decoders.dsl_schema.PredecoderDSL`

## Repository Conventions

- Python version: `3.12`
- Test runner: `pytest`
- Lint: `ruff`
- Coverage config lives in `pyproject.toml`
- Prefer `make` targets over handwritten command variants when both exist
- Default test path is non-integration only; integration tests require
  `--run-integration`

## Files Created at Runtime

Common generated files and directories:

- `.venv/` — local virtualenv
- `runs/` — run outputs, metrics, checkpoints, logs
- `.pytest_cache/`
- `.coverage`
- `autoqec.egg-info/`
- `build/`

Do not commit runtime artifacts, coverage outputs, or packaging byproducts.

## Editing Guidance

- Keep changes aligned with the current branch's architecture. Do not assume a
  feature from another branch exists locally.
- Start by reading the nearest test and the relevant `Makefile` target.
- Prefer small, targeted patches over broad refactors.
- When changing behavior, add or update tests first when practical.
- Update docs only when they are directly affected by the change.
- Follow existing naming and file layout instead of inventing new structure.

## Documentation

- `README.md` should stay as a project overview and quickstart.
- Detailed implementation notes belong in `docs/superpowers/specs/` and
  `docs/superpowers/plans/`.
- Demo-specific instructions belong under `demos/<demo-name>/README.md`.

## Commit Convention

Use concise conventional-style commit prefixes:

- `feat:`
- `fix:`
- `docs:`
- `test:`
- `chore:`
- `ci:`

Do not commit temporary plans or generated runtime artifacts unless the task
explicitly requires them.
