# Demo 2: bb72 research loop

## Goal

Show AutoQEC running a neural predecoder loop on the BB72 qLDPC smoke environment.

## Run

```bash
bash demos/demo-2-bb72/run.sh
MODE=fast bash demos/demo-2-bb72/run.sh
```

## Acceptance criteria

- Prod mode: all rounds complete and emit `metrics.json`.
- Fast mode: all 3 rounds complete and emit `candidate_pareto.json`.

## Known limitations for this branch

- The bb72 env uses a manually constructed parity-check matrix artifact.
- The parity-check path reports an exact-recovery surrogate instead of a true logical error rate because logical-operator metadata is not available in this repo yet.
- The emitted `candidate_pareto.json` is an unverified candidate Pareto summary for demo/reporting; the verification-admitted Pareto front remains owned by the verification slice.
