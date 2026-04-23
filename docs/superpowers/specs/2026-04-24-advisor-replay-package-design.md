# Advisor Replay Package Design

**Status:** approved design for issue #41
**Date:** 2026-04-24
**Issue:** `demo: advisor replay and reproducibility package`

## Goal

Deliver an offline advisor replay demo that proves a completed AutoQEC run can
be packaged, moved to a clean environment, and independently verified without
live LLM calls or network access.

The replay input is a `runs/<run_id>.tar.gz` package plus the repository
checkout at the manifest's repo SHA. The replay output is a fresh
`verification_report.json` / `verification_report.md` for a selected
`round_<N>/`, produced by the existing `python -m cli.autoqec verify` path.

## Non-Goals

- Do not build a new verifier. Replay reuses `cli.autoqec verify`.
- Do not make training deterministic. Training artifacts are packaged; replay
  re-runs holdout verification only.
- Do not require live Codex, Claude, or any other LLM backend.
- Do not require outbound network during replay.
- Do not claim bit-for-bit identical verification unless the same holdout seeds
  and shot count are used. Sampling variance remains documented.

## Package Format

The package is a gzip-compressed tar archive named:

```text
runs/<run_id>.tar.gz
```

The archive contains the run directory at its root:

```text
<run_id>/
├── history.jsonl
├── history.json                # optional, no-LLM path
├── candidate_pareto.json       # optional, no-LLM path
├── pareto.json                 # optional, verifier-owned path
├── log.md                      # optional, LLM/orchestration path
└── round_<N>/
    ├── artifact_manifest.json
    ├── checkpoint.pt
    ├── config.yaml
    ├── metrics.json
    ├── train.log
    ├── verification_report.json      # optional original report
    └── verification_report.md        # optional original report
```

Only files under the selected run directory are archived. The package command
rejects paths that are not directories, paths without at least one
`round_<N>/`, and round directories missing `artifact_manifest.json`.

## Artifact Manifest

Each `round_<N>/artifact_manifest.json` is the replay contract. Minimum fields:

```json
{
  "schema_version": 1,
  "created_at": "2026-04-24T00:00:00Z",
  "repo": {
    "commit_sha": "40-hex-or-null",
    "branch": "branch-name-or-null",
    "dirty": false
  },
  "environment": {
    "env_yaml_path": "autoqec/envs/builtin/surface_d5_depol.yaml",
    "env_yaml_sha256": "sha256"
  },
  "round": {
    "run_id": "20260424-120000",
    "round_dir": "round_1",
    "round": 1,
    "dsl_config_sha256": "sha256",
    "command_line": ["python", "-m", "cli.autoqec", "run", "..."]
  },
  "artifacts": {
    "config_yaml": "config.yaml",
    "checkpoint": "checkpoint.pt",
    "metrics": "metrics.json",
    "train_log": "train.log"
  },
  "packages": {
    "python": "3.12.3",
    "torch": "2.x-or-unavailable",
    "stim": "version-or-unavailable",
    "pymatching": "version-or-unavailable",
    "ldpc": "version-or-unavailable"
  }
}
```

The manifest stores paths relative to the round directory or repository root so
it survives extraction to another machine. Absolute paths can appear only in
`metrics.json`, where the existing `RoundMetrics` contract already allows them.

If Git metadata cannot be read, `repo.commit_sha` may be `null`, but the package
is not advisor-grade. For issue 41 scope, `package-run` must fail when the repo
SHA cannot be captured.

## CLI And Script Interfaces

### `package-run`

Add a small CLI entry point:

```bash
python -m cli.autoqec package-run runs/<run_id>
```

Behavior:

- validates every `round_<N>/artifact_manifest.json`
- writes `runs/<run_id>.tar.gz`
- prints a machine-readable `AUTOQEC_RESULT_JSON=` payload with
  `package_path`, `run_dir`, and discovered rounds
- refuses to overwrite an existing package

### Demo Script

Add:

```text
demos/demo-6-advisor-replay/run.sh
```

The script demonstrates:

1. finding or accepting a run directory
2. packaging it
3. extracting the tarball to a temporary replay directory
4. unsetting `AUTOQEC_IDEATOR_BACKEND`, `AUTOQEC_CODER_BACKEND`, and
   `AUTOQEC_ANALYST_BACKEND`
5. running:

```bash
python -m cli.autoqec verify <extracted_run>/round_1 \
  --env autoqec/envs/builtin/surface_d5_depol.yaml \
  --n-seeds 2 \
  --n-shots 8
```

The demo uses small defaults so CI and local smoke runs finish quickly. The
README documents how to raise `--n-seeds` and `--n-shots` for advisor runs.

## Data Flow

1. `cli.autoqec run --no-llm` or another run path creates `runs/<run_id>/`.
2. For each round, the runner writes core artifacts and
   `artifact_manifest.json`.
3. `package-run` validates manifests and archives the run directory.
4. Advisor extracts the archive in a clean directory.
5. Advisor checks out the manifest's `repo.commit_sha`.
6. Advisor unsets the three `AUTOQEC_*_BACKEND` variables.
7. Advisor re-runs `verify` against the extracted round directory.
8. The new `verification_report.json` is compared to the original report if an
   original exists; otherwise replay success is defined as a valid fresh report.

## Error Handling

- Missing manifest: fail package creation with a clear path-specific message.
- Missing checkpoint/config/metrics/train log: fail package creation.
- Archive path exists: fail without overwrite.
- Extracted replay lacks selected `round_<N>/`: fail before invoking verify.
- Backend env vars set: demo script unsets them before verify.
- Original report absent: replay still succeeds if verify writes a valid fresh
  report; README states no comparison was possible.
- Original and fresh reports differ: compare only stable fields by default
  (`verdict`, holdout seed list, shot count when present). Numeric LER deltas
  are allowed to vary within the documented bootstrap/statistical tolerance.

## Testing Strategy

Use TDD for each behavior change.

- Unit tests for manifest generation:
  - writes required top-level keys
  - records env YAML and DSL SHA-256 digests
  - records package versions without importing optional packages hard-failing
- Unit tests for package validation:
  - fails if a round lacks `artifact_manifest.json`
  - fails if required artifacts are missing
  - creates a tarball with safe relative members
- CLI tests:
  - `package-run` emits `AUTOQEC_RESULT_JSON`
  - `package-run` refuses to overwrite an existing tarball
- Demo replay tests:
  - replay helper unsets all three backend env vars
  - replay verify writes `verification_report.json`

Integration smoke can use tiny `--n-seeds` / `--n-shots` values. The full
advisor path remains manual because statistically meaningful verification can
be slow.

## Acceptance Criteria Mapping

- Package format exists: `runs/<run_id>.tar.gz` from `package-run`.
- Every round has manifest: runner writes `artifact_manifest.json`.
- Demo docs and script exist: `demos/demo-6-advisor-replay/README.md` and
  `run.sh`.
- No-network / no-LLM replay documented: README and script unset backend envs.
- Sample package path: README shows how to create one from Demo 1/2 output.
- `verify` completes after extraction: covered by demo replay test.
- Reproducibility caveats documented: README distinguishes exact artifacts from
  stochastic holdout sampling.
