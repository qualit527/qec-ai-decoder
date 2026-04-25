# Demo 6: Advisor Replay and Reproducibility Package

## Goal

Show that AutoQEC can:

1. produce a fresh one-round no-LLM run,
2. verify the resulting checkpoint,
3. package the run directory as `runs/<run_id>.tar.gz` via `python -m cli.autoqec package-run`,
4. extract that package in a clean replay directory, and
5. rerun `verify` offline with all `AUTOQEC_*_BACKEND` variables unset.

## Run

```bash
bash demos/demo-6-advisor-replay/run.sh
```

The script performs these commands in order:

```bash
python -m cli.autoqec run ... --no-llm
python -m cli.autoqec verify <round_dir> --env <yaml> --n-shots <n> --n-seeds <n>
python -m cli.autoqec package-run <run_dir>
python -m autoqec.tools.advisor_replay --run-dir <run_dir> --package-path runs/<run_id>.tar.gz ...
```

## Replay mode

This demo exercises a **no-network / no-LLM replay mode** for the replay
step only.

- `AUTOQEC_IDEATOR_BACKEND`, `AUTOQEC_CODER_BACKEND`,
  `AUTOQEC_ANALYST_BACKEND` are unset before replay verification.
- proxy-related environment variables are removed and `NO_PROXY=*` is set
  before replay verification.
- the replay Python process injects a temporary `sitecustomize.py` that
  blocks socket connections while verification runs.
- the replay step calls only `python -m cli.autoqec verify ...` on local
  artifacts from the extracted package.

## Package an existing Demo 1 / Demo 2 run

If you already have a completed run from Demo 1 or Demo 2, you do not need
to regenerate it. First package the run directory:

```bash
/home/jinguxie/qec-ai-decoder/.venv/bin/python -m cli.autoqec package-run runs/<run_id>
```

Then point the replay helper at that existing `run_dir` and package:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
/home/jinguxie/qec-ai-decoder/.venv/bin/python -m autoqec.tools.advisor_replay \
  --run-dir runs/<run_id> \
  --package-path runs/<run_id>.tar.gz \
  --env autoqec/envs/builtin/surface_d5_depol.yaml \
  --python-bin /home/jinguxie/qec-ai-decoder/.venv/bin/python \
  --n-shots 256 \
  --n-seeds 2 \
  --extract-root runs/demo-6-replay
```

For Demo 2, swap `--env` to `autoqec/envs/builtin/bb72_depol.yaml`.

## What is reproducible

- The packaged run artifacts under `runs/<run_id>.tar.gz`
- The `artifact_manifest.json` contents, including repo SHA, env YAML
  SHA-256, DSL SHA-256, package versions, and original command line
- The offline `verify` execution path on the extracted run
- The replayed `verification_report.json` schema and verdict

## What remains stochastic

The initial fresh run is still a training/evaluation process. Even in
`--no-llm` mode it is not promised to be bitwise identical across machines.
This demo proves replayability of the generated checkpoint and artifacts,
not determinism of the original training loop.

## Offline guarantee

The replay step removes:

- `AUTOQEC_IDEATOR_BACKEND`
- `AUTOQEC_CODER_BACKEND`
- `AUTOQEC_ANALYST_BACKEND`
- proxy-related environment variables (`HTTP_PROXY`, `HTTPS_PROXY`,
  `http_proxy`, `https_proxy`, `ALL_PROXY`, `all_proxy`)

from the environment before invoking `python -m cli.autoqec verify ...`,
sets `NO_PROXY=*`, and prepends a temporary `sitecustomize.py` that raises
if code attempts to open a network socket.

## Outputs

- `runs/<run_id>/round_1/artifact_manifest.json`
- `runs/<run_id>/round_1/verification_report.json`
- `runs/<run_id>.tar.gz`
- `runs/demo-6-replay/<run_id>/round_1/verification_report.original.json`
- `runs/demo-6-replay/<run_id>/round_1/verification_report.json`

## Runtime

Default settings target a demo-scale run:

- `ROUNDS=1`
- `PROFILE=dev`
- `N_SHOTS=256`
- `N_SEEDS=2`

These defaults keep replay practical while still proving the advisor-facing
package/replay workflow.

## Visual Showcase

Run the standalone visual showcase to generate a browser-friendly evidence
dashboard for Demo 6 only. The report focuses on the replay artifact paths and
the field-by-field verifier comparison between original and replay reports:

```bash
bash demos/demo-6-advisor-replay/showcase/run.sh
```

Outputs:

- `runs/demo-6-showcase/report.html`
- `runs/demo-6-showcase/report.md`
- `runs/demo-6-showcase/summary.json`

### Copy-paste Agent Prompt

Paste this prompt into Codex CLI / Claude Code from the repository root:

```text
Please run the Demo 6 visual showcase only.

Requirements:
- Do not call an LLM.
- Do not use the network during the replay step.
- Do not modify source files.
- Use Python: /home/jinguxie/qec-ai-decoder/.venv/bin/python

Command:
PYTHON_BIN=/home/jinguxie/qec-ai-decoder/.venv/bin/python bash demos/demo-6-advisor-replay/showcase/run.sh

After it finishes, tell me:
1. whether Demo 6 passed,
2. the status from runs/demo-6-showcase/summary.json,
3. the reports_match value from runs/demo-6-showcase/summary.json,
4. the comparison_matches / comparison_total field count from summary.json,
5. the package path,
6. the absolute path to report.html,
7. if artifact links do not open, the repo-root http://127.0.0.1 server command and report URL.

If all checks pass, end with:
demo6 showcase healthy
```

If `file://` links are not accessible from your browser, start a read-only
server from the repo root:

```bash
python -m http.server 8768 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8768/runs/demo-6-showcase/report.html
```
