---
name: autoqec-run
description: Run the AutoQEC research loop on a given env YAML. Orchestrates the autoqec-ideator / autoqec-coder / autoqec-analyst subagents via the Agent tool, invokes the Runner CLI for training + evaluation, and writes history.jsonl + pareto.json. Use when the user asks "run AutoQEC on <env>", "start a research round", or provides an EnvSpec YAML.
---

# /autoqec-run

One-shot orchestration recipe for Claude Code chat. The orchestrator (you)
drives the loop; three `autoqec-{ideator,coder,analyst}` subagents live
behind the `Agent` tool.

## When to use

- The user says "run AutoQEC", "start a research round", or provides an
  env YAML path.
- The user wants to see an AI-generated predecoder candidate on an
  existing env (surface_d5_depol, bb72_depol, etc.).

## Required inputs

- `env_yaml` — path to an EnvSpec YAML (e.g.
  `autoqec/envs/builtin/surface_d5_depol.yaml`).
- `rounds` — default `1`. Each round is Ideator → Coder → Runner →
  Analyst and takes ~1–10 min in dev profile.
- `profile` — `dev` (fast, 256/64 shots, 1 epoch) or `prod` (longer
  training). Default `dev`.

## The loop (per round)

Do these steps in order. If any step fails, halt and report the failure
to the user — do not silently skip.

### 1. Prepare the run + round directory

Pick a run id (UTC `YYYYMMDD-HHMMSS`) if the user did not supply one. All
subsequent paths live under `runs/<run_id>/`. Initialise memory via the
Python helper — do NOT write `history.jsonl` / `pareto.json` / `log.md`
by hand:

```bash
python -c "
from autoqec.orchestration.memory import RunMemory
mem = RunMemory('runs/<run_id>')
print('run_dir:', mem.run_dir)
"
```

### 2. Build the Ideator prompt

```bash
python scripts/run_single_round.py \
  --env-yaml <env_yaml> \
  --run-dir runs/<run_id> \
  --round-idx <N> > /tmp/plan_<N>.json
```

Read the JSON and extract `ideator_prompt`. That string is ready to
hand to the subagent.

### 3. Dispatch the Ideator

Use the `Agent` tool with `subagent_type="autoqec-ideator"` and pass the
`ideator_prompt` verbatim. The subagent returns exactly one fenced
```json block. Parse it via:

```bash
python -c "
import sys
from autoqec.agents.dispatch import parse_response
resp = open('/tmp/ideator_raw.txt', encoding='utf-8').read()
print(parse_response('ideator', resp))
"
```

Required keys: `hypothesis`, `expected_delta_ler`, `expected_cost_s`,
`rationale`. Optional: `dsl_hint`. If `parse_response` raises, the
response is malformed — re-dispatch once with a "your last response
failed §2.5 validation; try again" addendum, then give up if it fails
again.

### 4. Build the Coder prompt

```python
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.loop import build_coder_prompt
from pathlib import Path

mem = RunMemory("runs/<run_id>")
dsl_schema_md = Path("docs/superpowers/specs/2026-04-20-autoqec-design.md").read_text(encoding="utf-8")
coder_prompt = build_coder_prompt(
    hypothesis=<ideator-response-dict>,
    mem=mem,
    dsl_schema_md=dsl_schema_md,
)
```

### 5. Dispatch the Coder

`Agent` tool with `subagent_type="autoqec-coder"`. Required response
keys (validated): `tier`, `dsl_config`, `rationale`. `tier` is
`"1"` or `"2"`. `dsl_config` must validate against
`autoqec.decoders.dsl_schema.PredecoderDSL` — verify with:

```python
from autoqec.decoders.dsl_schema import PredecoderDSL
PredecoderDSL(**coder_response["dsl_config"])  # raises on drift
```

If validation fails, re-dispatch once with the pydantic error as
feedback. Give up after one retry.

### 6. Write `round_<N>/config.yaml` and invoke the Runner

```bash
mkdir -p runs/<run_id>/round_<N>
python -c "
import yaml
import sys, json
cfg = json.loads(sys.stdin.read())
with open('runs/<run_id>/round_<N>/config.yaml', 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
" <<< '<json-dump-of-coder.dsl_config>'

python -m cli.autoqec run-round \
  <env_yaml> \
  runs/<run_id>/round_<N>/config.yaml \
  runs/<run_id>/round_<N> \
  --profile <profile>
```

The CLI prints the `RoundMetrics` JSON on stdout. Save it — you also
need it for the Analyst. If `metrics.status != "ok"`, skip the Analyst
for this round; record the round with `verdict="ignore"` and move on.

### 7. Build the Analyst prompt

```python
from autoqec.orchestration.loop import build_analyst_prompt
analyst_prompt = build_analyst_prompt(
    mem=mem,
    round_dir="runs/<run_id>/round_<N>",
    prev_summary="<the previous round's summary_1line, or ''>",
)
```

### 8. Dispatch the Analyst

`Agent` tool with `subagent_type="autoqec-analyst"`. Required response
keys (validated): `summary_1line`, `verdict` (`"candidate"` or
`"ignore"`), `next_hypothesis_seed`.

### 9. Record the round

```python
from autoqec.orchestration.round_recorder import record_round
row = record_round(
    mem=mem,
    round_idx=<N>,
    hypothesis=<ideator-response-dict>["hypothesis"],
    dsl_config=<coder-response-dict>["dsl_config"],
    metrics=<metrics-dict>,
    verdict=<analyst-response-dict>["verdict"],
    summary_1line=<analyst-response-dict>["summary_1line"],
)
```

This writes the round to `history.jsonl`, appends to `log.md`, and
refreshes `pareto.json` (top-5 by Δ LER, candidates only).

### 10. Loop or stop

If the user asked for more than one round, go back to step 2 with
`round_idx+=1`. After the last round, print a short summary:

```
Run complete. Path: runs/<run_id>/
  log.md          narrative
  history.jsonl   <N> rounds
  pareto.json     top-K candidates by Δ LER
```

## Failure handling

- **Malformed subagent response** — retry once with the pydantic error
  appended to the prompt; give up after a second failure.
- **Runner `compile_error`** — record the round with `verdict="ignore"`,
  surface the `status_reason` in the chat, continue.
- **Runner `killed_by_safety`** — same as compile_error but the user
  should be told which safety fired (VRAM, NaN, wall-clock).
- **3 consecutive ignored rounds** — halt and ask the user to inspect;
  likely a DSL schema drift or Runner regression.

## Tool-use rules

- The orchestrator (you) may call `Agent`, `Bash`, `Read`, `Write`, `Edit`.
- `autoqec-ideator` subagent: `Read`, `Grep`, `Glob`.
- `autoqec-coder` subagent: `Read`, `Write`, `Edit`, `Grep`, `Glob`.
- `autoqec-analyst` subagent: `Read`, `Grep`.

## Costs + caveats

- Torch must be installed in the Python env used for `cli.autoqec
  run-round`. The rest of the loop is torch-free.
- Dev profile rounds finish in 1–3 minutes on CPU; prod rounds take
  10–20 min and benefit from GPU.
- Verification (`/verify-decoder`) is **not** part of this skill.
  Candidates emerging on `pareto.json` are Analyst-flagged, not holdout-
  verified.
