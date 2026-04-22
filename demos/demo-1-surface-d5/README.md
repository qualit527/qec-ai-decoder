# Demo 1 — AutoQEC on `surface_d5_depol`

Shows AutoQEC discovering a neural predecoder for a distance-5 rotated
surface code under circuit-level depolarising noise (`p = 5e-3`), and
reporting Δ LER vs the PyMatching baseline.

## What happens

One **round** is one full cycle:

1. `autoqec-ideator` subagent reads the current Pareto + knowledge excerpts
   and proposes a hypothesis — "try GNN with gated_mlp message_fn, 3 layers".
2. `autoqec-coder` subagent turns it into a Tier-1 `PredecoderDSL` config
   (or Tier-2 if the slot is novel).
3. The Runner (`cli/autoqec run-round`) compiles the config into an
   `nn.Module`, trains it on `seed_policy.train` shots, evaluates on
   `seed_policy.val` shots, and writes `metrics.json` +
   `checkpoint.pt` + `train.log` + `config.yaml` inside `round_<N>/`.
4. `autoqec-analyst` subagent reads `metrics.json` and writes a 3-sentence
   summary + verdict (`candidate` or `ignore`).
5. The orchestrator records the round into `history.jsonl`, updates
   `pareto.json`, and appends a line to `log.md`.

## How to run

### Path A — LLM loop (recommended, inside Claude Code)

In a Claude Code session, invoke the skill:

```
/autoqec-run autoqec/envs/builtin/surface_d5_depol.yaml --rounds 1 --profile dev
```

Claude Code follows `.claude/skills/autoqec-run/SKILL.md` to dispatch
all three subagents via the `Agent` tool and shells out to the Runner
for each round. Torch must be installed in the shell's active Python
env — the skill step that calls `cli.autoqec run-round` will otherwise
fail.

### Path B — no-LLM baseline (fallback, pure CLI)

If you just want to verify the Runner loop without an LLM in play:

```bash
bash demos/demo-1-surface-d5/run_quick.sh
```

Under the hood this is:

```bash
python -m cli.autoqec run \
    autoqec/envs/builtin/surface_d5_depol.yaml \
    --rounds 3 --profile dev --no-llm
```

`--no-llm` makes the CLI pick a random dev-safe template from
`autoqec/example_db/` each round. Useful as a smoke test; **not** the
AutoQEC research experience.

## Expected outputs

After a run:

```
runs/<run_id>/
├── log.md               # narrative, one line per round
├── history.jsonl        # machine-readable, one JSON per round
├── pareto.json          # top-K by Δ LER
└── round_<N>/
    ├── config.yaml      # the DSL config trained this round
    ├── train.log        # per-step loss
    ├── checkpoint.pt    # trained weights + dsl_config
    └── metrics.json     # RoundMetrics (status, Δ LER, FLOPs, n_params, …)
```

Headline metric: compare `round_<N>/metrics.json::ler_predecoder` to
the committed 1M-shot PyMatching reference at
`demos/demo-1-surface-d5/expected_output/baseline_benchmark.json`
(`LER = 0.01394` at `p = 5e-3`, seed 42). `delta_ler > 0` means the
predecoder beat PyMatching on this round's val shots.

## Runtime

- `dev` profile: 1–3 min per round on CPU, < 1 min on GPU.
- `prod` profile: 10–20 min per round on GPU (not recommended for demos).

## Caveats

- **No verifier in this demo.** The Analyst's `verdict = "candidate"`
  is **not** a VERIFIED signal — proper holdout evaluation is
  `/verify-decoder` (Xie's subtree, not yet in main).
- **Dev profile LER is noisy.** Dev shots are small (≤ 256 train / ≤ 64
  val); Δ LER CIs are wide. Use `prod` for publishable numbers, and
  run multiple seeds.
- **Seed templates are weak on surface codes.** Expect Δ LER near 0 or
  slightly negative in the first Tier-1 round. The loop's job is to
  iterate toward something better.

## Sample output

A smoke run (CPU torch, Python 3.10, surface_d5, `--rounds 3 --profile dev
--no-llm`, ~3 min total) is committed for reference:

- `expected_output/sample_run/history.jsonl` — all three rounds, `status=ok`
- `expected_output/sample_run/round_1_metrics.json` — one full `RoundMetrics` dump
- `expected_output/sample_run/round_1_config.yaml` — the DSL config the Runner trained

On that run: `ler_plain_classical = 0.01563` (eval shots = 64, hence
noisy vs the 1M-shot `LER = 0.01394` reference); `delta_ler = 0.0`
across all three dev-profile rounds. Real predecoder improvement shows
up in prod profile, not dev smoke.
