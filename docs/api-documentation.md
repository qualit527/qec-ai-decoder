# AutoQEC API Documentation

This document describes the public developer-facing interfaces currently
supported on `main`. It is intended as a stable entry point for contributors
who need to call the CLI, understand the core schemas, or consume run/round
artifacts without reading multiple plans and implementation files first.

For exact frozen field shapes, see:

- `docs/contracts/interfaces.md`
- `docs/contracts/round_dir_layout.md`

This file is a practical guide, not a replacement for those contracts.

## 1. Public CLI Interfaces

### 1.1 `python -m cli.autoqec run`

Primary purpose: run a multi-round no-LLM smoke loop using bundled seed
templates and write run-level artifacts under `runs/<timestamp>/`.

Example:

```bash
python -m cli.autoqec run autoqec/envs/builtin/surface_d5_depol.yaml \
  --rounds 3 \
  --profile dev \
  --no-llm
```

Inputs:

- `env_yaml`: path to an `EnvSpec` YAML file
- `--rounds`: number of rounds to execute
- `--profile`: `dev` or `prod`
- `--no-llm`: required on the current public path; LLM orchestration remains
  a skill-driven flow rather than a plain CLI feature on `main`

Behavior:

- Loads the environment via `load_env_yaml`
- Selects bundled decoder templates from `autoqec/example_db`
- Runs one `Runner` invocation per round
- Writes run-level and round-level artifacts
- Prints a machine-readable JSON summary containing `run_dir`, `rounds`, and
  `candidate_pareto_path`

Notes:

- The current public CLI path is a no-LLM smoke path
- `candidate_pareto.json` is demo/reporting output only; it is not the
  authoritative verifier-owned Pareto front

### 1.2 `python -m cli.autoqec run-round`

Primary purpose: run one round from a specific env/config/round-dir tuple and
emit `RoundMetrics` as JSON on stdout.

Example:

```bash
python -m cli.autoqec run-round \
  autoqec/envs/builtin/surface_d5_depol.yaml \
  autoqec/example_db/gnn_small.yaml \
  runs/manual/round_1 \
  --profile dev
```

Inputs:

- `env_yaml`: path to an `EnvSpec` YAML file
- `config_yaml`: path to a decoder config YAML
- `round_dir`: output directory for round artifacts
- `--profile`: `dev` or `prod`

Worktree-aware options:

- `--code-cwd`: absolute path to a worktree checkout
- `--branch`: branch name; required when `--code-cwd` is set
- `--fork-from`: parent branch or JSON list of parent branches
- `--compose-mode`: required when `--fork-from` is a list
- `--round-attempt-id`: UUID minted at attempt creation time

Behavior:

- Loads the env and decoder config
- Validates worktree-related metadata through `RunnerConfig`
- Runs the round in-process or via the subprocess/worktree path
- Prints a `RoundMetrics` JSON payload to stdout

Internal-only command:

- `python -m cli.autoqec run-round-internal`

This is a hidden internal entrypoint used by the subprocess bridge. It is not
part of the public CLI surface and should not be called directly by users.

## 2. Core Schemas

### 2.1 `EnvSpec`

`EnvSpec` defines the experiment environment. It answers: which code, which
noise model, which resource constraints, and which classical backend.

Top-level fields:

- `name`
- `code`
- `noise`
- `constraints`
- `baseline_decoders`
- `classical_backend`
- `eval_protocol`

Key practical notes:

- `code.source` is the asset path consumed by the Runner
- builtin env YAMLs live under `autoqec/envs/builtin/`
- loaders may resolve relative paths against the env file location or repo root

### 2.2 `RunnerConfig`

`RunnerConfig` defines how one round should execute.

Core fields:

- `env_name`
- `predecoder_config`
- `training_profile`
- `seed`
- `round_dir`

Worktree-path additions:

- `code_cwd`
- `branch`
- `fork_from`
- `fork_from_canonical`
- `fork_from_ordered`
- `compose_mode`

Practical interpretation:

- legacy path: only the core fields are required
- worktree path: provenance and branch-routing fields become part of the
  contract and are validated together

### 2.3 `RoundMetrics`

`RoundMetrics` is the canonical per-round result payload. It is written to
`metrics.json` and also emitted by `run-round`.

Key result fields:

- `status`
- `status_reason`
- `ler_plain_classical`
- `ler_predecoder`
- `delta_ler`
- `flops_per_syndrome`
- `n_params`
- `train_wallclock_s`
- `eval_wallclock_s`
- `vram_peak_gb`
- `checkpoint_path`
- `training_log_path`

Worktree/provenance additions:

- `round_attempt_id`
- `reconcile_id`
- `branch`
- `commit_sha`
- `fork_from`
- `fork_from_canonical`
- `fork_from_ordered`
- `compose_mode`
- `delta_vs_parent`
- `parent_ler`
- `conflicting_files`
- `train_seed`

Practical interpretation:

- `RoundMetrics` is the output contract consumed by artifact readers,
  analysis helpers, and orchestration-side bookkeeping
- any downstream tool that reads `metrics.json` should assume this schema
  rather than ad-hoc JSON keys

## 3. Output Artifacts and Their Consumers

### 3.1 Run-level artifacts

Common run-root files:

- `history.jsonl`
- `history.json`
- `log.md`
- `pareto.json`
- `candidate_pareto.json`

Meaning:

- `history.jsonl`: append-only per-round records
- `history.json`: aggregated run summary for simpler consumers
- `log.md`: orchestration narrative
- `pareto.json`: authoritative verifier-owned Pareto front
- `candidate_pareto.json`: unverified demo/reporting front for CLI smoke runs

### 3.2 Round-level artifacts

Per-round files:

- `config.yaml`
- `train.log`
- `checkpoint.pt`
- `metrics.json`
- `verification_report.md` when verification exists

Meaning:

- `config.yaml`: exact decoder config used for the round
- `train.log`: per-step loss trace
- `checkpoint.pt`: saved trained model bundle
- `metrics.json`: `RoundMetrics` JSON payload
- `verification_report.md`: verifier output when available

### 3.3 Main consumers

- Analyst subagent reads `metrics.json`
- Ideator context reads `history.jsonl` and `pareto.json`
- Verifier reads `checkpoint.pt` and `config.yaml`
- `machine_state` reads `history.jsonl`

For the authoritative layout and ownership rules, use
`docs/contracts/round_dir_layout.md` as the source of truth.

## 4. Source of Truth and Update Rules

When the public interface changes, this document should be updated together
with the relevant source-of-truth files.

Update this document when any of the following changes:

- CLI command shape for `run` or `run-round`
- top-level `EnvSpec`, `RunnerConfig`, or `RoundMetrics` usage contracts
- run / round artifact names, meanings, or consumer relationships

Also update, as applicable:

- `docs/contracts/interfaces.md`
- `docs/contracts/round_dir_layout.md`
- `README.md`
