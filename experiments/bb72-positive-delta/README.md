# BB72 Positive Delta Benchmark

This track provides reviewer-facing benchmark evidence that AutoQEC rounds can
produce a positive `delta_ler` when the neural predecoder output affects the
classical backend.

It intentionally does not replace Demo 1. Demo 1 proves the live LLM loop can
run end to end on `surface_d5`. The current `surface_d5 + mwpm + soft_priors`
path cannot demonstrate a real positive `delta_ler`, because MWPM decodes the
raw syndrome and ignores `soft_priors`.

## Why BB72/OSD

`bb72_perf.yaml` uses the BB72 parity-check artifact and
`classical_backend: osd`. In this path, `soft_priors` are passed into OSD as
`channel_probs`, so a trained predecoder can influence decoding.

## Run

```bash
python experiments/bb72-positive-delta/run.py
```

The script writes:

```text
runs/YYYYMMDDTHHMMSSZ-bb72-positive-delta/
|- round_1/
|- round_2/
|- round_3/
|- summary.json
`- report.md
```

## Success Criteria

The script exits nonzero unless:

- every round finishes with `status == "ok"`;
- at least one round has `delta_ler > 0`;
- the best round improves over round 1.

The result is benchmark evidence, not a VERIFIED holdout claim. The separate
verification workflow remains authoritative for `VERIFIED` Pareto admission.
