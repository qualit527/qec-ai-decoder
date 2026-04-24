# BB72 Positive Delta Benchmark Design

## Context

PR #69 stabilizes the live Demo 1 path, but its committed `surface_d5` run
still reports `delta_ler = 0.0` for every round. That is not only a weak
performance claim; in the current implementation it is expected. Demo 1 uses
`surface_d5_depol` with `classical_backend: mwpm`, while the live loop rejects
non-`soft_priors` predecoders. The MWPM adapter ignores `soft_priors` and
decodes the raw syndrome, so a `surface_d5 + mwpm + soft_priors` round cannot
show a real positive `delta_ler`.

The review response should therefore separate two claims:

- Demo 1 proves the live LLM loop can run end to end.
- A new benchmark track proves that AutoQEC rounds can produce positive
  `delta_ler` when the predecoder output actually affects the classical
  backend.

## Scope

Add `experiments/bb72-positive-delta/` as a reviewer-facing performance
benchmark track. It is not a replacement for Demo 1 and is not a verified
scientific holdout claim. Its purpose is to provide a reproducible experiment
showing at least one positive `delta_ler`, ideally with best-so-far improvement
across rounds.

The benchmark uses the BB72/qLDPC path because `classical_backend: osd`
consumes `soft_priors` as `channel_probs`. That gives the neural predecoder a
real path to influence the downstream decoder, unlike the current MWPM surface
path.

## Architecture

The benchmark is a thin orchestration layer over the existing Runner:

- `experiments/bb72-positive-delta/configs/round_*.yaml` stores a fixed
  schedule of candidate `PredecoderDSL` configs.
- `experiments/bb72-positive-delta/run.py` loads a benchmark environment,
  calls `run_round()` once per config, reads each `round_N/metrics.json`, and
  writes a compact summary.
- `experiments/bb72-positive-delta/README.md` explains why the track exists,
  how to reproduce it, and which fields answer the PR review comment.
- `experiments/bb72-positive-delta/expected_output/summary.json` stores a
  compact positive-delta snapshot.
- `autoqec/envs/builtin/bb72_perf.yaml` defines the benchmark environment. It
  keeps the BB72 parity-check artifact and OSD backend, but chooses noise and
  shot settings that make nonzero baseline error observable.

The Runner remains the source of truth for training, evaluation, metrics,
checkpoint format, and artifact layout. The benchmark driver only sequences
rounds and reports results.

## Data Flow

One benchmark run performs the following steps:

1. Load `bb72_perf.yaml`.
2. Create a run root under `runs/<timestamp>/`.
3. For each fixed config, call `run_round()` with deterministic seeds.
4. Read `round_N/metrics.json`.
5. Build per-round rows containing `delta_ler`, `best_delta_ler_so_far`,
   `ler_plain_classical`, `ler_predecoder`, `flops_per_syndrome`, `n_params`,
   and wall-clock fields.
6. Write `summary.json` and `report.md` under the run root.
7. For the committed snapshot, copy only compact expected-output artifacts.
   Do not commit checkpoints or runtime-heavy files.

## Success Gate

The benchmark should fail loudly instead of publishing a misleading result.
It exits nonzero if:

- any round has a non-`ok` Runner status;
- any expected `metrics.json` is missing or malformed;
- no round has `delta_ler > 0`;
- the best round does not improve over round 1.

The reviewer-facing claim is:

> In the BB72/OSD benchmark, at least one AutoQEC round achieved positive
> `delta_ler`, and the best-so-far result improved across the fixed schedule.

The claim must be labeled as benchmark evidence, not verified holdout proof.
The existing verifier remains the authority for `VERIFIED` Pareto admission.

## Testing

Default CI should not run the full benchmark. Tests should cover the new track
without adding slow runtime:

- Unit tests mock `run_round()` and assert that `run.py` writes `summary.json`
  and `report.md`.
- Unit tests assert that `run.py` exits nonzero when all `delta_ler <= 0`.
- Asset tests validate config YAMLs, README claims, and the expected
  `summary.json` schema.
- Manual or opt-in integration execution runs the full benchmark and refreshes
  `expected_output/summary.json`.

## Non-Goals

- Do not claim Demo 1 itself improves decoder performance until MWPM can use
  predecoder output.
- Do not implement weighted MWPM in this PR.
- Do not treat noisy dev-profile smoke output as a performance claim.
- Do not commit checkpoints, generated run directories, or large artifacts.
