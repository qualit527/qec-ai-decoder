# AGENTS.md

Developer guidance for Codex and similar coding agents when working on this
repository. For project overview, goals, and demos, see `README.md`.

## Build & Test

```bash
pip install -e '.[dev]'                     # install with dev deps
python -m pytest tests/ -m "not integration" -v   # unit tests (CI uses this)
python -m pytest tests/ -m "integration" -v --run-integration  # integration tests (manual entry)
python -m pytest tests/test_bootstrap.py -v        # single test file
python -m pytest tests/test_bootstrap.py::test_bootstrap_ci_basic -v  # single test
ruff check autoqec cli tests scripts              # lint (CI uses this)
ruff check --fix autoqec cli tests scripts        # lint with auto-fix
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml --rounds 1 --profile dev --no-llm  # smoke test
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
make install        # pip install -e '.[dev]'
make test           # pytest tests/ -m "not integration" -v
make test-integration # pytest tests/ -m "integration" -v --run-integration
make coverage       # pytest with coverage for autoqec + cli
make lint           # ruff check autoqec cli tests scripts
make run            # run cli.autoqec with env/profile defaults
make run-nollm      # N-round random-template smoke loop (no LLM needed)
make demo-2         # run bb72 qLDPC demo
make run-all-claude # force Claude-backed ideator/coder
make run-cheap      # swap ideator model to a cheaper default
```

Integration execution guidance lives in `docs/test-plan.md`. The default
repo gate stays `make lint` + `make test`; use `make test-integration`
only when you explicitly want the end-to-end layer.

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

## GitHub Workflow

- Use GitHub MCP tools for read-only context gathering only.
- Use the local `gh` CLI for GitHub write operations such as creating PRs,
  pushing follow-up metadata, commenting on issues, and other state-changing
  actions.
- When opening a PR, request the repository owner `qualit527` as a reviewer.
- If the PR author is the repository owner, request `tengxianglin` as the
  reviewer instead.

## Zulip Context

- When task context may depend on prior hackathon discussion, use the repo
  skill at `.claude/skills/read-zulip/SKILL.md` before making assumptions.
- Default Zulip context for this repository:
  - site: `https://quantum-info.zulipchat.com`
  - stream/channel: `VibeYoga-Hackathon-QEC` (`591576`)
  - topic: `general`
  - URL:
    `https://quantum-info.zulipchat.com/#channels/591576/VibeYoga-Hackathon-QEC/general`
  - If `general` returns no relevant messages, retry the same stream without a
    topic narrow, then check the `channel events` topic.
- Authentication should come from `ZULIPRC`, `~/.zuliprc`, or
  `~/.config/zulip/zuliprc`.
- Never store Zulip credentials in the repository.

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
