# Demo 7 — GPU Positive AI Loop

Shows the live AutoQEC loop running from one shell command:

```bash
bash demos/demo-7-gpu-positive-ai-loop/run.sh
```

That command uses `cli.autoqec run` without `--no-llm`, so the project AI
executes the full loop:

`Ideator -> Coder -> Runner -> Analyst -> history.jsonl + orchestrator_trace.md`

## Backend setup

The launcher respects the standard role backend environment variables. On the
server used for the committed evidence, Claude CLI credentials were valid and
the Codex CLI OpenAI key returned 401, so the run used:

```bash
AUTOQEC_IDEATOR_BACKEND=claude-cli \
AUTOQEC_IDEATOR_MODEL=claude-haiku-4-5 \
AUTOQEC_CODER_BACKEND=claude-cli \
AUTOQEC_CODER_MODEL=claude-haiku-4-5 \
AUTOQEC_ANALYST_BACKEND=claude-cli \
AUTOQEC_ANALYST_MODEL=claude-haiku-4-5 \
bash demos/demo-7-gpu-positive-ai-loop/run.sh
```

Defaults are `ROUNDS=1`, `PROFILE=prod`, and
`ENV_YAML=autoqec/envs/builtin/surface_d5_depol.yaml`.

## Evidence run

The reference run was produced on 2026-04-25 in the `feat/gpu-positive-demo`
worktree on an RTX 4090:

- run id: `20260425-234335`
- trace sections: Ideator response, Coder response, Runner metrics, Analyst
  response, run complete
- status: `ok`
- plain MWPM LER: `0.013916015625`
- neural-predecoder LER: `0.0130615234375`
- `delta_ler`: `+0.0008544921875`
- VRAM peak: `1.94 GB`
- training loss: `0.013585 -> 0.003462`

The compact committed snapshot lives at
`expected_output/live_loop_positive_summary.json`. Full runtime artifacts
remain under `runs/` and are intentionally not committed.

## Why This Demo Exists

Demo 1 and Demo 2 prove the runner and existing surface/qLDPC paths. Demo 7
proves the live AI loop can be started by one command, produce an agent trace,
train on GPU, and record a positive surface-code delta in the normal run
layout.

The Analyst verdict may still be `ignore` when the confidence interval crosses
zero. That is expected: this demo's claim is a positive training/eval result
with complete loop artifacts, not a verified Pareto admission.
