# Demo 5: Failure Diagnosis and Recovery Walkthrough

## Goal

Show that AutoQEC can read a broken run directory, identify the failure class, and recommend a fix **without applying changes autonomously**.

## Run

```bash
bash demos/demo-5-failure-recovery/run.sh
```

## What it covers

| Round | Failure mode   | Root cause                                        |
|-------|----------------|---------------------------------------------------|
| 0     | compile_error  | `hidden_dim: -1` fails pydantic validation        |
| 1     | NaN loss       | Learning rate too high (10.0) causes NaN cascade  |
| 2     | OOM            | Model too large for available VRAM                |

## CLI commands used

```bash
# Diagnose a specific round
python -m cli.autoqec diagnose runs/demo-5/round_0

# Skill-layer walkthrough
/diagnose-failure runs/demo-5/round_0

# Diagnose a run directory (auto-selects latest round, round_2)
python -m cli.autoqec diagnose runs/demo-5
```

## Acceptance criteria

- [x] Demo exits 0
- [x] `bad_config.yaml` contains a clear schema failure (`hidden_dim: -1`)
- [x] `run.sh` creates synthetic failed `round_0`, `round_1`, `round_2` directories
- [x] CLI `diagnose` outputs structured JSON with path, root_cause, signals, and artifact flags
- [x] CLI `diagnose` identifies the latest round when passed a run directory
- [x] CLI `diagnose` handles direct round-dir input
- [x] `/diagnose-failure` skill contract is exercised against the synthetic run, and `diagnosis.md` is written per round with root cause, evidence citations, and patched YAML
- [x] Walkthrough states the system does not apply fixes automatically
- [x] Covers three failure modes: compile error, NaN loss, OOM

## Runtime

~30 seconds (no training, no GPU, no LLM calls).
