# Demo 2 Live Walkthrough Design

**Date**: 2026-04-24
**Status**: Approved design for documentation
**Project**: `qec-ai-decoder`
**Scope**: `demos/demo-2-bb72/`

## 1. Goal

Add a demo-facing document that helps a presenter use `demo-2-bb72` in a live setting.

The document should make `demo-2` land as a one-prompt delegation story:

- the presenter gives the agent one natural-language prompt;
- the agent runs the BB72 smoke demo on its own;
- the agent proves that the run used the qLDPC OSD path rather than the surface-code MWPM path; and
- the agent reports reproducible artifacts and a plain-language conclusion.

The live-demo claim must stay disciplined. The demo should not be framed as "one prompt solves QEC" or as a single-round proof of decoder superiority.

## 2. Audience

The walkthrough must work for three audiences at once:

- Hackathon judges, who care about clear product-style delegation and visible outcomes.
- QEC researchers, who care that the backend genuinely switches from MWPM to OSD and that the claim is technically defensible.
- The project advisor, who cares that the run is reproducible and leaves inspectable artifacts behind.

## 3. Chosen Framing

The recommended message is:

> Given one prompt, the agent can launch a reproducible qLDPC demo round, audit the result, and explain why it matters.

This framing is preferred over:

- "BB72 has a strong metric": the committed smoke snapshot shows `delta_ler == 0.0`, so metric-centric framing is weak.
- "The repo has a convenient shell script": this undersells the agentic part of the project.
- "One prompt solves the decoding problem": this is too broad and is not supported by the demo.

## 4. Deliverable

Create `demos/demo-2-bb72/walkthrough.md`.

The walkthrough should be operational, not aspirational. It must be something the presenter can read or paste during a live demo without further editing.

## 5. Walkthrough Content Requirements

The walkthrough must include:

1. A short statement of what the demo proves.
2. A short audience-specific framing section.
3. Preflight advice for a stable live run.
4. A 20-second opening script.
5. One recommended prompt that tells the agent to:
   - run `MODE=fast bash demos/demo-2-bb72/run.sh`;
   - identify the new run directory;
   - prove `classical_backend: osd` in `autoqec/envs/builtin/bb72_depol.yaml`;
   - prove `round_1/metrics.json.status == "ok"`;
   - confirm the expected run artifacts; and
   - explain the result in plain language.
6. A list of the concrete on-screen cues the presenter should point out.
7. A failure/slow-run fallback script.
8. Explicit guidance on what not to claim.
9. A short closing script.

## 6. Tone and Constraints

- Follow the existing repository documentation style: concise, direct, English-first.
- Optimize for live presentation, not implementation detail.
- Keep evidence tied to files and fields visible in the repo or run output.
- Avoid deep QEC jargon unless it is needed to distinguish OSD from MWPM.
- Do not claim verified Pareto admission or performance gains from this smoke run.

## 7. Acceptance Criteria

The documentation is correct when:

- a presenter can use it without reading source files first;
- it makes the one-prompt delegation story clearer than the current `README.md`;
- it preserves the technical distinction between `surface_d5_depol` and `bb72_depol`;
- it does not overclaim what the smoke run proves; and
- it lives inside `demos/demo-2-bb72/`, matching the demo-local documentation convention.

## 8. Out of Scope

- Editing the demo runtime or wrapper scripts.
- Reworking `demos/demo-2-bb72/README.md` into a presentation guide.
- Adding browser mockups or a UI companion for this demo.
- Reframing the project-wide README around this single demo.
