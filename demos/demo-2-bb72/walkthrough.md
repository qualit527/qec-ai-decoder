# Demo 2 walkthrough — one-prompt qLDPC proof

## Demo goal

This demo is not trying to prove that BB72 wins on a single smoke run. It is
trying to prove something more important for a live presentation:

> One natural-language prompt is enough for the agent to launch a reproducible
> qLDPC round, audit the result, and explain why the run is technically
> different from the surface-code path.

The core evidence is:

- the agent runs the demo without hand-held shell work;
- the run uses `classical_backend: osd`;
- the round finishes with `status == "ok"`; and
- the harness writes the expected artifacts under `runs/<run_id>/`.

## Why this lands with each audience

- **Hackathon judges:** this looks like delegated work, not a human driving a
  shell script line by line.
- **QEC researchers:** the claim is concrete: the same harness now routes
  through BB72 + OSD instead of surface code + MWPM.
- **Advisor:** the result is reproducible and inspectable because the run leaves
  behind `metrics.json`, `history.jsonl`, and the round artifacts.

## Preflight

- Start in the repo root with `.venv` already activated and dependencies
  already installed.
- Use the demo wrapper in `fast` mode:

  ```bash
  MODE=fast bash demos/demo-2-bb72/run.sh
  ```

- Prefer the smoke path in a live setting. It is enough to show routing,
  artifact generation, and agent-driven auditing.
- Do not promise a quality gain from this single run. The smoke path is about
  routing and reproducibility, not a headline `delta_ler`.

## 20-second opener

Use something close to this:

> Demo 1 showed that the pipeline runs end to end. Here I switch to a different
> code family, BB72 qLDPC, and I do not hand-drive the verification. I give the
> agent one prompt, it runs the round, proves that the backend changed from the
> surface-code MWPM path to OSD, and brings back the artifacts for inspection.

## One prompt to paste

```text
From the repo root, run Demo 2 and audit the result without modifying source files.

1. Run `MODE=fast bash demos/demo-2-bb72/run.sh`.
2. Identify the new run directory created under `runs/`.
3. Prove that `autoqec/envs/builtin/bb72_depol.yaml` uses `classical_backend: osd`.
4. Prove that the new run completed successfully by checking `round_1/metrics.json` and reporting whether `status == "ok"`.
5. Confirm that the run wrote `history.jsonl`, `candidate_pareto.json`, and `round_1/{config.yaml,train.log,checkpoint.pt,metrics.json}`.
6. Summarize in five bullets:
   - command and wall-clock
   - run directory path
   - proof that this demo routes through OSD rather than the surface-code MWPM path
   - proof that the round finished successfully
   - one short plain-language explanation of why this matters

If the command fails, keep going: show the last 30 lines of output and explain the most likely cause. Do not edit files.
```

## What to point at on screen

Point to these items and say why each matters:

1. The natural-language prompt itself.
   This is the "one-prompt" claim. The audience should see that you are
   delegating a concrete audit task, not manually running a checklist.
2. The command the agent decides to run.
   This grounds the demo in a real executable path.
3. `classical_backend: osd` in `autoqec/envs/builtin/bb72_depol.yaml`.
   This is the technical proof that the route changed away from the surface-code
   MWPM path.
4. `status == "ok"` in the new `round_1/metrics.json`.
   This is the proof that the round actually completed.
5. The generated files in the new `runs/<run_id>/` directory.
   This shows the run is reproducible and inspectable, not just a terminal log.

## What success looks like

- The agent runs the BB72 smoke demo without extra coaching.
- The agent explicitly says `OSD`, not just "a different backend".
- The agent names the new `runs/<run_id>/` directory and the expected files.
- The agent gives one short human explanation of why this is different from the
  surface demo.

If you get all four, the demo has landed.

## If the run is slow

While it is running, say this:

> This is still the cheap smoke path: one dev-profile round on BB72. What I
> care about here is not squeezing out a flashy number. I care that the agent
> can run the task and audit the result for me.

That turns waiting time into part of the story instead of dead air.

## If the run fails

Do not hide the failure. Reframe it:

> Even if the run fails, the useful part is that the task is structured. The
> agent can still tell us which command failed, which file or artifact is
> missing, and whether the backend routing or the run contract broke.

Then point to the agent's reported stderr tail, missing artifact, or failed
status field.

## What not to claim

- Do not say this one prompt "solves QEC."
- Do not say this smoke run proves a better decoder than the baseline.
- Do not present `candidate_pareto.json` as a verified scientific result. The
  demo README explicitly treats it as unverified demo output.
- Do not oversell the demo as a BB72-specific achievement. The point is the
  harness staying stable while the environment and backend change.

## 20-second close

Use something close to this:

> The important thing here is not just that BB72 runs. It is that one prompt is
> enough to delegate a technically meaningful qLDPC check: the agent runs the
> demo, proves that the route switched to OSD, and returns reproducible
> artifacts. That is the core of an agentic research workflow rather than a
> one-off script.
