# Demo 5: Failure recovery

## Goal
Show that `/diagnose-failure` reads broken-run state and recommends a fix.

## Run
```bash
bash demos/demo-5-failure-recovery/run.sh
```

## Acceptance
- CLI prints structured round stats.
- The LLM skill layer (invoked via `/diagnose-failure runs/demo-5/`) would produce
  a `diagnosis.md` identifying `hidden_dim: -1` as the root cause and suggesting
  `hidden_dim: 32`.

## Runtime
~1 minute.
