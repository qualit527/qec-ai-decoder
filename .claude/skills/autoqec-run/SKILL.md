---
name: autoqec-run
description: Run the AutoQEC research loop on a given env YAML. Orchestrates the autoqec-ideator / autoqec-coder / autoqec-analyst subagents via the Agent tool, invokes the Runner CLI for training + evaluation, and writes history.jsonl + pareto.json. Use when the user asks "run AutoQEC on <env>", "start a research round", or provides an EnvSpec YAML.
---

# /autoqec-run

One-shot orchestration recipe for Claude Code chat. The orchestrator (you)
drives the loop; three `autoqec-{ideator,coder,analyst}` subagents live
behind the `Agent` tool.

All shell commands below are written as `python -c "..."` one-liners so
the recipe is identical on PowerShell (Windows) and bash / zsh
(macOS / Linux). The helpers resolve every path relative to the repo
root via `_REPO_ROOT = Path(__file__).resolve().parents[...]` — no cwd
assumptions.

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

## Working scratch directory

Use Python `tempfile.gettempdir()` when you need to stash intermediate
subagent responses. Never hard-code `/tmp/...`.

```bash
python -c "import tempfile,os; print(os.path.join(tempfile.gettempdir(), 'autoqec-run'))"
```

Create it once per run; the steps below refer to it as `<SCRATCH>`.

## The loop (per round)

Do these steps in order. If any step fails, halt and report the failure
to the user — do not silently skip.

### 1. Prepare the run + round directory

Pick a run id (UTC `YYYYMMDD-HHMMSS`) if the user did not supply one. All
subsequent paths live under `runs/<run_id>/`. Initialise memory via the
Python helper — do NOT write `history.jsonl` / `pareto.json` / `log.md`
by hand:

```bash
python -c "from autoqec.orchestration.memory import RunMemory; print(RunMemory('runs/<run_id>').run_dir)"
```

### 2. Build the Ideator prompt

```bash
python scripts/run_single_round.py --env-yaml <env_yaml> --run-dir runs/<run_id> --round-idx <N>
```

Capture stdout into a JSON variable (the whole plan dict). The
`ideator_prompt` key inside is the string to hand to the subagent.

### 3. Dispatch the Ideator

Use the `Agent` tool with `subagent_type="autoqec-ideator"` and pass the
`ideator_prompt` verbatim. The subagent returns exactly one fenced
```json block. Validate + parse via:

```bash
python -c "
import sys, json
from autoqec.agents.dispatch import parse_response
raw = sys.stdin.read()
parsed = parse_response('ideator', raw)
print(json.dumps(parsed))
"
```

Pipe the subagent's raw response on stdin. Required keys (validated
automatically by `parse_response`): `hypothesis`, `expected_delta_ler`,
`expected_cost_s`, `rationale`. Optional: `dsl_hint`.

If `parse_response` raises `ValueError: Invalid ideator response
payload:`, the response is malformed — re-dispatch once with the
pydantic error appended to the prompt, then give up if it fails again.

### 4. Build the Coder prompt

```bash
python -c "
import json, sys
from pathlib import Path
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.loop import build_coder_prompt
hypothesis = json.loads(sys.stdin.read())
mem = RunMemory('runs/<run_id>')
schema_md = Path('autoqec/decoders/dsl_schema.py').read_text(encoding='utf-8')
print(build_coder_prompt(hypothesis=hypothesis, mem=mem, dsl_schema_md=schema_md))
"
```

Pipe the parsed Ideator JSON on stdin.

### 5. Dispatch the Coder

`Agent` tool with `subagent_type="autoqec-coder"`. Required response
keys (validated by `parse_response`): `tier`, `dsl_config`, `rationale`.
`tier` is `"1"` or `"2"`.

Double-check `dsl_config` against `PredecoderDSL`:

```bash
python -c "
import json, sys
from autoqec.agents.dispatch import parse_response
from autoqec.decoders.dsl_schema import PredecoderDSL
parsed = parse_response('coder', sys.stdin.read())
PredecoderDSL(**parsed['dsl_config'])   # raises on schema drift
print(json.dumps(parsed))
"
```

If validation fails, re-dispatch once with the pydantic error as
feedback. Give up after one retry.

## Fork-from decision (§15.2)

After the Ideator responds, read its `fork_from` field:

| `fork_from` value | Action |
|---|---|
| `"baseline"` | Call `create_round_worktree(repo_root, run_id, N, slug, fork_from="main")`. |
| `"exp/<run_id>/<M>-<slug>"` (string) | Call `create_round_worktree(..., fork_from=<that branch name>)`. |
| list of ≥2 branch names | **Compose round**: call `create_compose_worktree(repo_root, run_id, N, slug, parents=<list>)`. If it returns `status="compose_conflict"`: write a `history.jsonl` row via `record_round` with `status=compose_conflict`, `round_attempt_id=<UUID>`, `conflicting_files=<list>`, and skip to the next round (no Runner invocation). |

Mint `round_attempt_id` as `str(uuid.uuid4())` **before** calling the Ideator prompt is assembled, and pass it through every subsequent call (Coder prompt, Runner subprocess, Analyst prompt, record_round).

## Runner invocation (worktree path)

Use `subprocess_runner.run_round_in_subprocess(cfg, env, round_attempt_id=<uuid>)` with:
- `cfg.code_cwd = <worktree_dir>` (from the worktree plan dict)
- `cfg.branch = <branch>` (from the plan)
- `cfg.fork_from = <Ideator's fork_from value>`
- `cfg.compose_mode = <Ideator's compose_mode>` (only for compose rounds)

On success (metrics.status == "ok"), the subprocess has already committed the pointer file in the worktree. Proceed to Analyst dispatch.

On `status=compose_conflict` or any non-ok status, call `cleanup_round_worktree` (removes the checkout; branch persists unless it was a compose_conflict, in which case `create_compose_worktree` already deleted the branch).

### 6. Write `round_<N>/config.yaml` and invoke the Runner

```bash
python -c "
import json, sys, yaml
from pathlib import Path
parsed = json.loads(sys.stdin.read())
out = Path('runs/<run_id>/round_<N>')
out.mkdir(parents=True, exist_ok=True)
with (out / 'config.yaml').open('w', encoding='utf-8') as f:
    yaml.safe_dump(parsed['dsl_config'], f, sort_keys=False)
"

python -m cli.autoqec run-round <env_yaml> runs/<run_id>/round_<N>/config.yaml runs/<run_id>/round_<N> --profile <profile>
```

The CLI prints the `RoundMetrics` JSON on stdout. Save it — you also
need it for the Analyst.

**If `metrics.status != "ok"`**, skip the Analyst and the verifier for
this round. Do not call `build_analyst_prompt`, do not dispatch the
`autoqec-analyst` subagent, and do not run `independent_verify`. Call
`record_round()` at step 10 with only `round_metrics=<metrics>` (omit
`verify_verdict` and `verify_report`); the recorder synthesises a
fallback summary from `metrics.status` / `status_reason`.

### 7. Build the Analyst prompt (only if `metrics.status == "ok"`)

```bash
python -c "
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.loop import build_analyst_prompt
mem = RunMemory('runs/<run_id>')
print(build_analyst_prompt(mem=mem, round_dir='runs/<run_id>/round_<N>', prev_summary='<prev or empty>'))
"
```

### 8. Dispatch the Analyst

`Agent` tool with `subagent_type="autoqec-analyst"`. Required response
keys (validated): `summary_1line`, `verdict` (`"candidate"` or
`"ignore"`), `next_hypothesis_seed`.

### 9. Run the independent verifier (automatic)

If `metrics.status == "ok"` AND the Analyst verdict is `candidate`:

```python
from pathlib import Path
from autoqec.envs.schema import load_env_yaml
from autoqec.eval.independent_eval import independent_verify

env = load_env_yaml("<ENV_YAML_PATH>")
sp = env.noise.seed_policy
holdout = list(range(sp.holdout[0], sp.holdout[1] + 1))[:50]
report = independent_verify(
    checkpoint=Path("<ROUND_DIR>/checkpoint.pt"),
    env_spec=env,
    holdout_seeds=holdout,
)
Path("<ROUND_DIR>/verification_report.json").write_text(
    report.model_dump_json(indent=2), encoding="utf-8",
)
verify_verdict = report.verdict       # pass into record_round
verify_report = report.model_dump()   # ditto
```

**Skip cases:** `metrics.status != "ok"`, `compose_conflict`, Analyst verdict == `ignore`.

**On verifier crash:** catch, set `verify_verdict="FAILED"`, `verify_report=None`, include the exception in the Analyst summary.

### 10. Record the round

```bash
python -c "
import json, sys
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round
payload = json.loads(sys.stdin.read())
mem = RunMemory('runs/<run_id>')
row = record_round(
    mem=mem,
    round_metrics=payload['round_metrics'],
    verify_verdict=payload.get('verify_verdict'),  # 'VERIFIED' | 'FAILED' | None
    verify_report=payload.get('verify_report'),    # dict or None
)
print(json.dumps(row['summary_1line']))
"
```

Pipe one JSON object on stdin with keys: `round_metrics` (the full
`RoundMetrics` dict including `branch`, `commit_sha`, `fork_from`,
`round_attempt_id`, `status`, `status_reason`, etc.), optional
`verify_verdict` (`"VERIFIED"` / `"FAILED"` / `None`), and optional
`verify_report` (the `VerifyReport` dict from Step 9, or `None`). The
recorder synthesises a fallback `summary_1line` on the runner-failure
path.

This writes the round to `history.jsonl`, appends to `log.md`, and
refreshes `pareto.json` (full non-dominated archive) alongside
`pareto_preview.json` (top-5 projection for humans).

### 11. Loop or stop

If the user asked for more than one round, go back to step 2 with
`round_idx+=1`. After the last round, print a short summary:

```
Run complete. Path: runs/<run_id>/
  log.md          narrative
  history.jsonl   <N> rounds
  pareto.json     top-K candidates by Δ LER
```

## Failure handling

- **Malformed subagent response** — `parse_response` raises with the
  pydantic error in the message. Retry once with that error appended to
  the prompt; give up after a second failure.
- **Runner `compile_error`** — call `record_round(mem, round_metrics=<metrics>)`
  without `verify_verdict`; surface the `status_reason` in the chat;
  continue.
- **Runner `killed_by_safety`** — same as compile_error; the recorder's
  fallback summary shows which safety fired.
- **3 consecutive ignored rounds** — halt and ask the user to inspect;
  likely a DSL schema drift or Runner regression.

## Compose conflict handling (§15.6.3)

If a compose round returns `status="compose_conflict"`:
- `record_round` writes the `compose_conflict` row BEFORE any cleanup
- The synthetic branch is already deleted by `create_compose_worktree`
- Ideator on the next round will see the `FAILED_compose` node in `fork_graph` and must not re-propose the same `fork_from_canonical` set
- Do NOT run the Step 9 verifier for compose_conflict rounds — there's no checkpoint

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
- Verification runs **automatically** in this skill for successful
  rounds with Analyst verdict `candidate` (Step 9). `/verify-decoder`
  remains available for post-hoc re-audits with different holdout seeds.
