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

# Optional skill-layer walkthrough command
# The demo script prints this suggestion but does not invoke an LLM skill.
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
- [x] `run.sh` prints the `/diagnose-failure` skill command for manual follow-up and writes `diagnosis.md` per round with root cause, evidence citations, and patched YAML via the local generator
- [x] Walkthrough states the system does not apply fixes automatically
- [x] Covers three failure modes: compile error, NaN loss, OOM

## Runtime

~30 seconds (no training, no GPU, no LLM calls).

## Visual Showcase

Run the standalone visual showcase to generate a browser-friendly evidence
dashboard for Demo 5 only:

```bash
bash demos/demo-5-failure-recovery/showcase/run.sh
```

Outputs:

- `runs/demo-5-showcase/report.html`
- `runs/demo-5-showcase/report.md`
- `runs/demo-5-showcase/summary.json`

### Copy-paste Agent Prompt

Paste this prompt into Codex CLI / Claude Code from the repository root:

```text
Please run the Demo 5 visual showcase only.

Requirements:
- Do not call an LLM.
- Do not use the network.
- Do not modify source files.
- Use Python: ./.venv/bin/python (auto-discovered by the script)

Command:
bash demos/demo-5-failure-recovery/showcase/run.sh

After it finishes, tell me:
1. whether Demo 5 passed,
2. the status from runs/demo-5-showcase/summary.json,
3. the absolute path to report.html,
4. the file:// link for report.html,
5. if artifact links do not open, the repo-root http://127.0.0.1 server command and report URL.

If all checks pass, end with:
demo5 showcase healthy
```

If `file://` links are not accessible from your browser, start a read-only
server from the repo root:

```bash
python -m http.server 8767 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8767/runs/demo-5-showcase/report.html
```
