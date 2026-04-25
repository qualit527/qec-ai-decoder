---
name: demo-presenter
description: Use when preparing an AutoQEC demo walkthrough, advisor presentation, hackathon pitch, recorded narration, or evidence-backed explanation of why the demos matter and why their outputs are correct.
---

# Demo Presenter

## Overview

Create a persuasive, human-voiced AutoQEC demonstration from repository
evidence. The output must distinguish what is merged on `main`, what is
planned or present only on a PR branch, and what was actually verified during
the current session.

## Core Rule

Never invent demo results. Use only repository files, command output, committed
snapshots, or explicitly-labeled planned/PR evidence. If a command is not run,
say it is a scriptable talking point, not a fresh verification.

## Evidence First

Before writing the presentation, collect the current evidence:

1. Read `README.md`, especially the one-prompt review block, feature table,
   demo table, and skill table.
2. Read demo docs:
   - `demos/demo-1-surface-d5/README.md`
   - `demos/demo-2-bb72/README.md`
   - `demos/demo-4-reward-hacking/README.md`
   - `demos/demo-4-reward-hacking/walkthrough.md` when explaining integrity.
   - `demos/demo-5-failure-recovery/README.md`
3. Check current branch state with `git status --short --branch`.
4. If asked for a live presentation, run the README's demo commands in order
   and capture command, wall-clock, pass/fail, artifact paths, and the last
   error lines for failures.
5. If asked for a scripted or dry-run presentation, use committed snapshots and
   label them as snapshots.
6. For the planned worktree demo, inspect `origin/feat/issue-38-worktree-demo`
   without switching away from `main`:

```bash
git show origin/feat/issue-38-worktree-demo:demos/demo-3-worktree-provenance/README.md
git show origin/feat/issue-38-worktree-demo:demos/demo-3-worktree-provenance/expected_output/run_demo.stdout.txt
git diff --name-status main..origin/feat/issue-38-worktree-demo
```

Treat this as planned / PR-only until the files exist on `main`.

## Spoken Audio

If the user wants AI narration, write the `Narration Script` section to a plain
text file and run the bundled TTS helper:

```bash
python .claude/skills/demo-presenter/scripts/speak_demo.py demo_script.txt --output runs/demo_audio.mp3
```

Requirements:

- Set `OPENAI_API_KEY` in the shell.
- If using an OpenAI-compatible gateway, set `OPENAI_BASE_URL` or pass
  `--base-url`.
- Install the optional `openai` Python package if the script reports that it is
  missing.
- Tell listeners the voice is AI-generated.

Useful options:

```bash
python .claude/skills/demo-presenter/scripts/speak_demo.py demo_script.txt \
  --output runs/demo_audio.mp3 \
  --voice coral \
  --model gpt-4o-mini-tts \
  --instructions "Speak like a warm, precise research demo host."
```

OpenAI-compatible gateway example:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.example.com"
python .claude/skills/demo-presenter/scripts/speak_demo.py demo_script.txt --output runs/demo_audio.mp3
```

For long scripts, the helper splits requests. If `ffmpeg` is installed it also
combines the parts into the requested output file; otherwise it leaves numbered
part files plus a playlist next to the requested output.

## Live Narrated Demo

If the user asks to run demos while AI narrates, use the live runner:

```bash
python .claude/skills/demo-presenter/scripts/live_present_demo.py --output-dir runs/demo-presenter-live/manual
```

Behavior:

- Plays or writes an opening narration.
- For each demo, starts the "why it matters" narration and the demo command
  **concurrently** so the listener hears the context while the command actually
  runs. The runner waits for both before moving on.
- Narrates the pass/fail meaning after the command exits (sequential, so the
  listener hears the verdict before the next demo starts).
- Writes per-demo logs and `manifest.json` under the output directory.
- Keeps Demo 3 worktree provenance as planned / PR-only unless
  `--include-pr-worktree` is passed and the demo directory exists locally.

### Audio player requirements for parallel playback

The overlap uses a non-blocking CLI player. The runner auto-detects, in order:
`ffplay`, `afplay` (macOS), `mpv`, `mpg123`, `paplay`, `aplay`. Install one so
the narration does not open a GUI media player:

- **Windows / cross-platform**: install `ffmpeg` (which ships `ffplay`).
- **macOS**: `afplay` is preinstalled.
- **Linux**: `mpv` or PulseAudio's `paplay` are common choices.

If no async player is found, pre-narration falls back to blocking playback (the
platform default, which may open a GUI). Use `--no-overlap` to force this
sequential mode even when an async player is available.

Useful options:

```bash
python .claude/skills/demo-presenter/scripts/live_present_demo.py \
  --output-dir runs/demo-presenter-live/advisor \
  --skip-demo 4 \
  --include-pr-worktree
```

For rehearsal without running long commands or calling TTS:

```bash
python .claude/skills/demo-presenter/scripts/live_present_demo.py \
  --dry-run \
  --no-audio \
  --output-dir runs/demo-presenter-live/rehearsal
```

For generating audio files without attempting playback:

```bash
python .claude/skills/demo-presenter/scripts/live_present_demo.py \
  --no-playback \
  --output-dir runs/demo-presenter-live/audio-only
```

To disable the concurrent playback and keep the old narrate-then-run ordering:

```bash
python .claude/skills/demo-presenter/scripts/live_present_demo.py \
  --no-overlap \
  --output-dir runs/demo-presenter-live/sequential
```

## Demo Order

Use this sequence unless the user asks otherwise:

| Segment | Status label | Command or source | Claim to prove |
|---|---|---|---|
| Opening | Context | `README.md` | AutoQEC is an agentic research harness for neural predecoders, not just a training script. |
| Demo 1: surface_d5 | Merged on main | `bash demos/demo-1-surface-d5/run_quick.sh` | The full DSL -> train -> eval -> artifact loop works on the surface-code/MWPM path. |
| Demo 2: BB72 qLDPC | Merged on main | `MODE=fast bash demos/demo-2-bb72/run.sh` | The same harness swaps to qLDPC/OSD through environment config rather than special-case code. |
| Demo 3: worktree provenance | Planned / PR-only until merged | `origin/feat/issue-38-worktree-demo:demos/demo-3-worktree-provenance/README.md` | Branches-as-Pareto preserves provenance, supports compose rounds, and records merge conflicts as scientific outcomes. |
| Demo 4: reward-hacking rejection | Merged on main | `bash demos/demo-4-reward-hacking/present.sh` | A memorizing cheater is rejected by independent verification, so the Pareto front is guarded. |
| Demo 5: failure recovery | Merged on main | `bash demos/demo-5-failure-recovery/run.sh` | Broken rounds produce machine-readable root cause signals for diagnosis. |
| Closing | Synthesis | Artifacts and pass/fail table | Importance: reproducible automated discovery. Correctness: artifacts, holdout verification, backend switching, provenance, and diagnostics. |

## Presentation Shape

Return a presentation with these sections:

1. **Run Card**: repo branch, commit SHA if available, whether commands were
   executed live or summarized from snapshots, and any PR-only evidence used.
2. **Narration Script**: first-person host voice, concise and human. Speak as
   an expert guide: "Now I want to show why this matters..." Avoid hype that
   outruns the evidence.
3. **Demo Beats**: for each demo, include:
   - what the audience sees,
   - command or source file,
   - pass/fail criterion,
   - artifact path,
   - why it matters,
   - why it supports correctness.
4. **Evidence Table**: demo, status label, command/source, key artifact,
   pass/fail or snapshot-only, claim proven.
5. **Honest Limits**: noisy dev-profile LER, unverified candidate Pareto,
   PR-only worktree demo status, skipped live-LLM path, or failed commands.
6. **Closing Argument**: one paragraph connecting the demos to the core claim:
   AutoQEC is a reproducible, auditable harness for agent-driven QEC decoder
   research.

## Voice Rules

- Use a warm, technically credible presenter voice.
- Explain importance before correctness for each demo.
- Prefer concrete language over slogans: "this writes `metrics.json` and
  `checkpoint.pt`" beats "this proves everything works."
- Say "planned / PR-only" for Demo 3 until it is merged.
- Do not claim a PR-only demo passed on `main`.
- Do not claim VERIFIED status for no-LLM candidate Pareto output unless an
  independent verifier report says so.

## Explanation-mode demos (visualized walkthroughs)

Some demos ship two launchers:

- `run.sh` — CI-style pass/fail, one-shot exit code.
- `present.sh` — narrated, multi-phase output with ASCII visualizations
  and an optional PNG. Same exit semantics as `run.sh`, but stdout is
  structured so a presenter can comment on each phase as it lands.

Prefer `present.sh` whenever a human is watching. `run.sh` stays the
stable CI entry point; do not remove it.

### Demo 4 — how to narrate the five phases

`bash demos/demo-4-reward-hacking/present.sh` prints five phase headers
(`PHASE 1 … PHASE 5`) in order and writes `present_summary.json` plus
`visualizations/scoreboard.png` under `runs/demo-4/round_0/`. When
narrating, do exactly one thing per phase — do not front-load conclusions:

1. **PHASE 1 — Construct the cheater.** Point out the memorized-entry
   count and the training seed range. Say out loud: "this is an
   adversarial example, not a bug — I am building the worst-case reviewer
   submission to check the verifier actually catches it."
2. **PHASE 2 — Hit-rate scoreboard.** Three bars appear (memorized /
   fresh-train / holdout). The scientific punchline is that the cheat is
   bound to *specific shots*, not to seed ranges: fresh train-seed hits
   ≈ holdout hits. Narrate this explicitly — it is the key to why the
   `seed-leakage` guard *passes* while the cheat still fails the fair test.
3. **PHASE 3 — Fair-test LER.** Two bars appear (plain MWPM vs
   memorizer). Read out Δ_LER (holdout) and the 95% CI. Emphasise that
   the memorizer is not merely weaker — it is *worse than doing nothing*
   because missed lookups poison the MWPM backend with a zero hint.
4. **PHASE 4 — Three guards checklist.** For each guard, state which
   failure mode it covers. Call out which guard tripped (usually
   `paired bootstrap CI` for this cheat). Mention that a different
   cheater class (e.g. `trap_A` seed leakage, `trap_C` ablation fail)
   would trip a different guard — the guards are not redundant.
5. **PHASE 5 — Verdict + Pareto consequence.** Read the banner verdict
   (expected: `FAILED`; `SUSPICIOUS` is also acceptable rejection).
   State the Pareto consequence explicitly: the checkpoint is not
   admitted to `pareto.json`; the run's branch would be tagged
   `rejected_by_verifier` in `fork_graph.json`.

Artifacts to cite by path after the walkthrough:

- `runs/demo-4/round_0/verification_report.json` — the machine-readable verdict
- `runs/demo-4/round_0/present_summary.json` — the phase-by-phase log
- `runs/demo-4/round_0/visualizations/scoreboard.png` — 2-panel figure

If matplotlib is unavailable, present.py prints
`(matplotlib unavailable, skipping PNG)` and exits 0; the ASCII output
alone is sufficient — do not treat the missing PNG as a failure.

## Worktree Demo Handling

When including the worktree demo before merge:

1. Identify it as `Demo 3 - Worktree branches-as-Pareto provenance`.
2. State that it currently lives on `origin/feat/issue-38-worktree-demo`, not
   on `main`, unless the demo directory is present locally.
3. Summarize the expected sequence:
   - Round 1 forks Idea A from baseline.
   - Round 2 forks Idea B from baseline.
   - Round 3 composes A and B with a merge worktree.
   - A conflict probe records `status="compose_conflict"` instead of crashing.
   - `reconcile_at_startup` can recover orphaned branch rows from pointer JSON.
4. Cite PR-branch evidence such as:
   - `demos/demo-3-worktree-provenance/README.md`
   - `expected_output/run_demo.stdout.txt`
   - `round_1_pointer.json`
5. Explain why it matters: research candidates become inspectable git branches,
   not opaque rows in a database.
6. Explain correctness: branch names, commit SHAs, pointer JSON, merge graph,
   and conflict status make provenance auditable.

## Failure Handling

If a demo command fails, keep going. In the presentation:

- mark that demo as failed,
- include the last useful stdout/stderr lines,
- explain which claim is not freshly supported,
- distinguish "the command failed in this environment" from "the design claim is false",
- do not hide failures behind narration.

## Quick Prompt Pattern

User: "Use `/demo-presenter` to prepare the advisor walkthrough."

You:

1. Collect evidence.
2. Ask whether to run live commands if the user did not specify live vs dry-run.
3. Produce the run card, narration script, demo beats, evidence table, limits,
   and closing argument.
