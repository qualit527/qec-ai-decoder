# AutoQEC Hackathon Overview and Curated References

Last updated: 2026-04-21

## 1. What We Are Building

The long-term ambition of this project is not merely to add another QEC decoder. The intended system is an automated decoding engine for quantum error correction, `AutoQEC`, that integrates LLM-based auto-research into the physical and algorithmic workflow of QEC.

In the original project vision, the system should:

- take hardware-specific noise maps and inference-latency constraints as input,
- use agent workflows to explore decoder designs under Tanner-graph structure constraints,
- generate large-scale noisy syndrome datasets automatically, ideally with Stim-based pipelines,
- run architecture ideation, training, threshold evaluation, and iteration in a closed loop,
- combine learned and classical decoding through a `Neural-BP + MWPM/OSD` style hybrid stack,
- evolve from synthetic-data pretraining toward small-sample adaptation on real-chip data,
- eventually provide Pareto-optimal pretrained models and inference operators for surface codes and representative qLDPC codes.

That is the research direction. It is not the scope of a one-week hackathon deliverable.

## 2. What the Current Hackathon Actually Needs

The current repository has already narrowed the scope to a Rust-only MVP that can be built and demonstrated within the hackathon window.

The current target is:

- surface code only,
- a reusable Rust workspace plus one unified CLI,
- a four-stage pipeline: `generate -> train -> eval -> report`,
- two noise settings:
  - iid baseline noise,
  - one simple correlated or crosstalk-inspired noise,
- three decoder variants:
  - MWPM baseline,
  - lightweight learned decoder,
  - hybrid learned + MWPM decoder,
- a demo-quality artifact showing the accuracy-latency Pareto tradeoff.

So the current task is not "build the full AutoQEC vision." The current task is to build the first engineering-grade prototype that is runnable, benchmarkable, extensible, and demo-ready.

## 3. Why the MVP Boundary Matters

The MVP boundary is essential for four reasons:

- a hackathon usually fails because the scope explodes, not because the team cannot write enough code,
- a convincing demo is a complete and reproducible end-to-end workflow, not a large collection of ideas,
- without a unified CLI, config layer, tests, and reports, later model improvements will be difficult to reproduce or compare,
- QEC decoders must be evaluated jointly on logical error rate, latency, and throughput, so the benchmarking substrate has to exist early.

Accordingly, the current design explicitly defers:

- qLDPC support,
- OSD unless integration is nearly free,
- Tanner-graph-constrained NAS,
- full pretrain-then-adapt workflows,
- autonomous scientific self-iteration,
- broad noise-family coverage,
- heavier Neural-BP implementations.

## 4. Recommended Division of Responsibilities

This project should not rely on free-form prompting alone. Different types of work should live in different layers:

- Rust crates and CLI commands should own deterministic, reproducible, benchmarkable behavior.
- design docs, plans, and skills should own reasoning, prioritization, workflow guidance, and interpretation.
- experiment behavior should be config-driven instead of hard-coded into scripts.
- tests and reporting should be part of the main system, not afterthoughts.

This matches the principle already adopted in the design documents:

- deterministic and reproducible work belongs in Rust modules or CLI commands,
- reasoning, prioritization, experiment planning, and result interpretation belong in reusable skills or documented workflows.

## 5. Expected Deliverables for This Hackathon

The most important deliverables are:

1. a Rust workspace with focused modules such as `core`, `data`, `decoders`, `train`, `bench`, and `cli`,
2. one unified CLI with at least `generate`, `train`, `eval`, `report`, and `run`,
3. reproducible experiment configs covering a compact benchmark matrix,
4. benchmark and report artifacts that expose logical error rate, latency, throughput, and a Pareto-style comparison,
5. a reusable harness around the codebase, including design docs, implementation plans, test planning, and a repeatable workflow for future contributors.

## 6. Engineering Lessons from the References

### 6.1 OpenAI: Harness Engineering

The key idea behind harness engineering is not "find a smarter model." It is to build constraints, checks, and feedback loops that make agent behavior reliable.

Applied to this repository, that means:

- do not stop at describing the task; encode success criteria into a runnable harness,
- define evaluation and acceptance criteria before asking agents to implement,
- move repeated actions into CLI commands, scripts, tests, and stable datasets,
- convert "the model seems capable" into "the system passes verification consistently."

This directly aligns with the AutoQEC MVP: the immediate need is to turn `generate/train/eval/report` into a repeatable research pipeline with inspectable outputs.

### 6.2 arXiv 2604.11535: Problem Reductions at Scale

The main value of this paper for us is not the domain of problem reductions. It is the demonstration that a well-designed harness can let agents build large, verification-heavy software systems with sustained momentum.

The relevant takeaways are:

- provide a low-friction contribution path for domain experts,
- rely on layered verification rather than one final correctness check,
- structure implementation, review, and integration as a repeatable pipeline,
- introduce new capability through stable interfaces so the system composes cleanly.

For AutoQEC, this implies:

- new decoders, noise models, and benchmark dimensions should enter through common traits and config schemas,
- repository structure should be optimized for future extension, not just a one-off demo.

### 6.3 VibeTraining / VibeYoga

VibeTraining contributes a hackathon methodology rather than QEC-specific content:

- choose a real workflow worth automating,
- decide explicitly what belongs in skills versus scripts, CLIs, or MCP tools,
- build with verification rather than optimizing only for generation speed,
- produce something that is both demoable now and reusable later.

The most relevant steps from the guide for this repository are:

- brainstorm first, then inspect related projects,
- initialize the repo around the design file,
- create a human verification test plan early,
- prefer a subagent-driven implementation workflow when executing the plan.

### 6.4 What to Learn from `problem-reductions`

`problem-reductions` is useful as a model for how an AI-assisted Rust project can be organized:

- the separation between library and CLI is clear,
- docs, tests, CI, and examples reinforce one coherent system,
- domain experts can contribute ideas without needing to implement everything themselves,
- the repository turns external ideas into structured, trackable work.

AutoQEC should move in the same direction:

- design and implementation plans should live in the repository instead of chat history,
- the CLI should be the main interface,
- benchmark and integration tests should be first-class paths, not optional extras.

## 7. Actionable Requirements Extracted from Zulip

From the `VibeYoga-Hackathon-QEC / channel events` Zulip thread and the uploaded `QEC-week-v2.docx`, the most actionable requirements are:

- use a GitHub repository as the main execution surface,
- connect implementation work back to issues,
- study `problem-reductions` as a structural reference,
- run a reflection step on the design document before implementation grows,
- design a test plan early rather than after coding,
- encode test datasets with input and expected output, preferably as JSON,
- give agents a sufficiently non-interactive execution environment so they are not blocked by unnecessary prompts.

All of these support the same goal: the project should depend on explicit repository structure, tests, and workflows rather than on a single chat session.

## 8. Useful Prompts Mentioned in Zulip

The Zulip thread did contain useful prompts. They are worth preserving because they encode the intended working style of the hackathon.

### 8.1 Original prompts from the thread

These are the prompts or prompt-like instructions that appeared explicitly in the thread:

```text
quote https://codingthrust.github.io/VibeTraining/
```

```text
clone this repo to local machine: https://github.com/CodingThrust/problem-reductions/ , check it, see what can we learn from it?
```

```text
invoke this skill: https://github.com/CodingThrust/VibeTraining/blob/main/.claude/skills/reflect/SKILL.md on the design file.
```

```text
Brainstorm: let us design a test plan together to validate the correctness of the software. Generate test datasets as json files that records input and expected output. These tests must cover all main APIs.
```

```text
quote https://openai.com/index/harness-engineering/
```

### 8.2 Recommended adapted prompts for this repository

The raw prompts above are useful, but the following versions are more directly actionable for `qec-ai-decoder`:

```text
Read the design document at docs/superpowers/specs/2026-04-20-autoqec-rust-mvp-design.md.
Use a reflect-style critique on the design and identify:
1. scope risks,
2. missing interfaces,
3. benchmark risks,
4. what must be cut to keep the MVP hackathon-sized.
Write the result back into docs as an implementation-facing note.
```

```text
Clone https://github.com/CodingThrust/problem-reductions and study what we should borrow for this repository.
Focus on:
1. workspace structure,
2. CLI surface,
3. tests and verification,
4. issue-to-implementation workflow,
5. documentation layout.
Summarize concrete changes we should adopt for AutoQEC.
```

```text
Design a test plan for the AutoQEC MVP.
Generate small test datasets as JSON fixtures with explicit input and expected output.
The test plan must cover all main APIs and all CLI stages: generate, train, eval, report, and run.
Separate unit tests, integration tests, and benchmark-validation tests.
```

```text
Using harness-engineering principles, turn the AutoQEC MVP into a reliable agent workflow.
Define:
1. success criteria,
2. evaluation checkpoints,
3. failure modes,
4. required artifacts,
5. what should be implemented as Rust code versus skills or docs.
```

### 8.3 Why these prompts matter

These prompts are useful because they push the team toward:

- design pressure-testing before implementation,
- learning from a proven Rust + agent workflow repository,
- test-first verification habits,
- harness-level thinking instead of isolated code generation.

## 9. Working Definition of the Current Task

If the current task needs to be summarized in one sentence, it is this:

> Within one week, compress the broad vision of an AI-enhanced QEC decoder into a Rust CLI MVP that runs end-to-end, emits a Pareto-style result, and leaves behind an extensible engineering foundation.

More concretely, the immediate priorities are:

1. define stable schemas and interfaces,
2. scaffold the workspace and CLI,
3. make the minimal data-generation, training, evaluation, and reporting loop runnable,
4. produce a comparable benchmark result,
5. preserve the workflow in docs, plans, and tests.

## 10. Reference Links Worth Keeping

### Core references

- OpenAI, Harness Engineering  
  https://openai.com/index/harness-engineering/
- Pan, An, Liu. Problem Reductions at Scale: Agentic Integration of Computationally Hard Problems  
  https://arxiv.org/abs/2604.11535
- VibeTraining / VibeYoga  
  https://codingthrust.github.io/VibeTraining/
- VibeTraining step-by-step guide  
  https://codingthrust.github.io/VibeTraining/step-by-step
- VibeTraining resources  
  https://codingthrust.github.io/VibeTraining/resources
- `problem-reductions` repository  
  https://github.com/CodingThrust/problem-reductions/

### Secondary references from the Zulip thread

- brainstorming skill  
  https://github.com/obra/superpowers/blob/main/skills/brainstorming/SKILL.md
- reflect skill  
  https://github.com/CodingThrust/VibeTraining/blob/main/.claude/skills/reflect/SKILL.md
- Zulip thread  
  https://quantum-info.zulipchat.com/#narrow/channel/591576-VibeYoga-Hackathon-QEC/topic/channel.20events/with/586361588

### Links mentioned in the thread but not central to the current technical direction

- `https://luckyapi.chat/`
- `https://zenmux.ai/pricing/subscription`

These may still be operationally useful, but they do not materially shape the current AutoQEC MVP architecture.
