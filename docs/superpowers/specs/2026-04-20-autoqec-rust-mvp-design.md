# AutoQEC Rust-Only MVP Design

Date: 2026-04-20
Topic: Rust-only AutoQEC hackathon MVP
Status: Approved design for planning

## 1. Goal

Build a one-week hackathon MVP called AutoQEC: a Rust-only, end-to-end research workflow for AI-enhanced quantum error-correction decoding on surface-code data. The MVP must run from a unified Rust CLI, execute a compact experiment matrix, and produce an accuracy-latency Pareto artifact suitable for demo use.

Primary success criterion:
- one Rust command runs the full surface-code pipeline and produces benchmark metrics, latency-quality comparisons, and at least one clean Pareto plot or Pareto-style report artifact.

Secondary success criterion:
- the system is packaged as a reusable Rust framework/CLI that others can extend after the hackathon.

## 2. Scope

### In MVP
- Surface code only
- Rust-only core pipeline
- Config-driven workflow
- Surface-code syndrome data generation in Rust, or a Rust wrapper around an external generator if needed, while keeping the user-facing and orchestration layer in Rust
- Two noise models:
  - iid baseline noise
  - one simple correlated or crosstalk-inspired variant
- Three decoder variants:
  - MWPM baseline
  - lightweight learned decoder
  - hybrid learned + MWPM decoder
- Four automated pipeline stages:
  - generate
  - train
  - eval
  - report
- One unified end-to-end command
- Metrics and reporting:
  - logical error rate
  - inference latency
  - throughput
  - Pareto plot or Pareto-style summary artifact
- Five demo flows:
  - data demo
  - baseline demo
  - hybrid demo
  - Pareto demo
  - automation demo

### Explicitly out of MVP
- qLDPC support
- OSD unless integration is nearly free
- Tanner-graph-constrained NAS
- pretrain-then-adapt workflow
- autonomous scientific closed-loop iteration
- many noise families beyond the two required variants
- full Neural-BP if the lightweight decoder already satisfies the AI-enhanced story

## 3. Guiding principle

If a component is deterministic and reproducible, it should be implemented as a Rust module or CLI command. If a component involves reasoning, prioritization, experiment planning, or result interpretation, it should be implemented as a reusable AI skill.

For the MVP, the project should be treated as a scoped systems demo rather than a full deep-learning research stack.

## 4. Recommended approach

Use a Rust systems MVP with a tiny trainable message-passing decoder.

This is the recommended approach because it preserves the AI-enhanced narrative while keeping the project feasible within one week under the Rust-only constraint.

### Alternatives considered
1. Benchmark platform with no true learned training
   - safest for delivery
   - weakest AI narrative
2. Simplified Neural-BP in Rust
   - stronger research story
   - significantly higher implementation and stabilization risk

The chosen approach sits between these extremes.

## 5. Strict MVP checklist

### Must-have
1. Rust workspace with one CLI entrypoint and reproducible config-driven runs
2. Surface-code-only data generation path
3. Two noise models: iid and one simple correlated or crosstalk-inspired variant
4. One classical baseline: MWPM
5. One lightweight learned decoder with fixed structure and a small trainable parameter set
6. One hybrid decoder: learned preprocessing plus MWPM fallback
7. Four pipeline stages: generate, train, eval, report
8. One end-to-end command that runs the compact experiment matrix
9. Output artifacts:
   - metrics table
   - latency summary
   - throughput summary
   - Pareto plot or Pareto-style report
10. Demo configs and scripts for the five demos

### Should-have if time allows
- OSD as an additional fallback
- richer correlated-noise parameterization
- small ablation study over learned-decoder settings
- HTML report polish

### Must-not-expand
- qLDPC
- NAS
- adaptation loop
- autonomous self-iteration
- larger decoder family search

## 6. Pressure test and risks

### Overall viability
The overall direction is viable only if the learned component is intentionally modest and the project prioritizes engineering completeness over research breadth.

### Highest-risk components
1. Rust-native learned decoder training
   - Risk: a full trainable Neural-BP stack in Rust can consume the week
   - Mitigation: use a lightweight parameterized message-passing decoder with fixed topology and few parameters
2. Surface-code dataset generation and noise fidelity
   - Risk: realistic syndrome generation can sprawl
   - Mitigation: keep generation minimal and scoped to surface code with only two noise settings
3. Classical fallback decoder complexity
   - Risk: adding OSD can increase engineering load without being critical to the MVP
   - Mitigation: require MWPM, defer OSD
4. Fair latency benchmarking
   - Risk: unstable or incomparable latency numbers weaken the Pareto story
   - Mitigation: pin one machine, one build profile, one batch convention, and report p50 and p95 latency
5. Over-automation
   - Risk: trying to fully realize AutoQEC’s autonomous vision can dilute the demo
   - Mitigation: automate only generate, train, eval, and report

## 7. Minimum viable learned-decoder design

The learned component should be a trainable weighted message-passing decoder, not a full Neural-BP system.

### Structure
- Fixed Tanner or syndrome graph for the surface code
- Unrolled for a small number of iterations, such as 3 to 5
- Small trainable parameter set:
  - per-iteration global weight
  - per-iteration damping factor
  - optional edge-type weights by relation class, not per-edge
- Output options:
  - corrected syndrome confidence scores, or
  - ranked error-likelihood map passed to MWPM

### Why this is the MVP sweet spot
- genuinely supports the AI-enhanced decoder story
- much easier to train in Rust than a heavier neural architecture
- simple to benchmark and explain
- hybrid integration is natural and robust

### Fallback rule
If gradient-based training is unstable or too slow, reduce the learned decoder to parameterized reweighting with grid or random search while preserving the same decoder interface and benchmark flow.

## 8. Minimum viable experiment matrix

The benchmark should be intentionally compact.

### Required dimensions
- Code family: surface code only
- Code distances: 2 values
- Noise models: 2 values
  - iid
  - one correlated or crosstalk-inspired template
- Decoder variants: 3 values
  - MWPM baseline
  - learned decoder
  - hybrid learned + MWPM
- Physical error rates: 3 values
  - low
  - medium
  - high

This matrix is sufficient to produce:
- logical error comparisons
- latency comparisons
- throughput comparisons
- one Pareto-style summary

## 9. Metrics and benchmarking conventions

### Required metrics
- logical error rate
- inference latency
- throughput

### Benchmarking conventions
- use one fixed machine
- use one fixed build profile
- use one fixed batch convention
- report p50 latency
- report p95 latency
- document dataset size and number of trials for each benchmark

The MVP does not need to claim universal performance; it only needs to present consistent, reproducible measurements.

## 10. Rust repo structure

```text
autoqec-rs/
  Cargo.toml
  Cargo.lock
  README.md

  crates/
    autoqec-core/
      src/
        code/
        syndrome/
        noise/
        decoder/
        metrics/
        timing/

    autoqec-data/
      src/
        surface_code.rs
        noise_models.rs
        dataset.rs

    autoqec-decoders/
      src/
        mwpm.rs
        learned.rs
        hybrid.rs
        interfaces.rs

    autoqec-train/
      src/
        trainer.rs
        loss.rs
        checkpoint.rs

    autoqec-bench/
      src/
        eval.rs
        latency.rs
        pareto.rs
        report.rs

    autoqec-cli/
      src/
        main.rs
        commands/
          generate.rs
          train.rs
          eval.rs
          report.rs
          run.rs

  configs/
    data/
    train/
    eval/
    experiments/

  demos/
  artifacts/
    datasets/
    checkpoints/
    evals/
    reports/
  docs/
  tests/
```

### Module ownership guidance
- `autoqec-core`: shared types, interfaces, metrics, and timing utilities
- `autoqec-data`: surface-code data generation and noise instantiation
- `autoqec-decoders`: decoder implementations behind stable interfaces
- `autoqec-train`: training loop and checkpoint handling for the lightweight learned decoder
- `autoqec-bench`: evaluation, benchmarking, Pareto generation, and report assembly
- `autoqec-cli`: user-facing workflow orchestration

## 11. CLI and command-level UX

The command surface should be simple, artifact-driven, and demo-friendly.

### Core commands
- `autoqec generate -c configs/data/surface_iid.toml`
- `autoqec train -c configs/train/learned_small.toml`
- `autoqec eval -c configs/eval/mvp.toml`
- `autoqec report -c configs/experiments/mvp.toml`
- `autoqec run -c configs/experiments/mvp.toml`

### Demo commands
- `autoqec demo data`
- `autoqec demo baseline`
- `autoqec demo hybrid`
- `autoqec demo pareto`
- `autoqec demo full`

### Artifact layout
- `artifacts/datasets/...`
- `artifacts/checkpoints/...`
- `artifacts/evals/...`
- `artifacts/reports/...`

### UX rules
- every command prints input config, output paths, and key metrics
- `run` chains all stages and emits a final summary
- `report` generates both machine-readable CSV or JSON and one presentation-ready PNG or HTML summary

## 12. Division between AI skills and Rust code

### Reusable AI skills
1. Project scoping skill: refine MVP scope and cut risky features
2. Research and baseline skill: summarize prior work and recommend practical Rust-feasible baselines
3. Experiment design skill: propose experiment matrices and ablations
4. Training and tuning skill: analyze training runs and suggest next comparisons
5. Evaluation and reporting skill: analyze benchmark outputs and generate a demo-ready narrative

### Rust modules and deterministic tools
1. Syndrome generation
2. Noise instantiation
3. Decoder execution
4. Training loop
5. Benchmarking and Pareto report generation

## 13. Five MVP features

1. Surface-code syndrome data generation with configurable code distance and noise rate
2. Pluggable Rust noise models starting from iid and extending to one correlated or crosstalk-inspired variant
3. Hybrid decoding pipeline: lightweight learned preprocessing followed by MWPM fallback
4. Automated benchmarking for logical error rate, inference latency, throughput, and Pareto output
5. Unified Rust CLI or workflow runner for generate, train, evaluate, and report

## 14. Five reusable skills

1. Project scoping skill
2. Research and baseline skill
3. Experiment design skill
4. Training and tuning skill
5. Evaluation and reporting skill

These skills support planning and interpretation, not deterministic execution.

## 15. Five demos

1. Data demo: generate a surface-code syndrome dataset from a Rust config
2. Baseline demo: run MWPM on the generated data
3. Hybrid decoder demo: run the hybrid pipeline and compare it with the baseline
4. Pareto demo: visualize the quality-latency tradeoff across decoder variants
5. Automation demo: execute the full pipeline end-to-end with one command

## 16. One-week execution plan

### Day 1
- Freeze scope
- Freeze repo structure
- Freeze config format
- Freeze benchmark definition
- Implement surface-code data path
- Implement iid noise model

### Day 2
- Implement MWPM baseline
- Verify end-to-end metric computation
- Define and document latency benchmarking convention

### Day 3
- Implement the smallest feasible learned decoder in Rust
- Verify both training and inference run successfully

### Day 4
- Implement the hybrid decoder
- Add one correlated-noise variant

### Day 5
- Run the compact benchmark matrix
- Collect metrics and artifacts

### Day 6
- Finalize report generation
- Polish CLI and configs
- Finalize demo scripts and documentation

### Day 7
- Prepare the final demo
- Refine the presentation narrative
- Document stretch goals and future work

## 17. Team execution plan

This plan assumes three collaborators: Jiahan Chen, Jinguo Liu, and Tengxiang Lin.

### Jiahan Chen
Primary ownership:
- data pipeline
- noise models
- experiment configs

Deliverables:
- surface-code generation path
- iid noise model
- correlated-noise variant
- baseline experiment configs

### Jinguo Liu
Primary ownership:
- decoder implementations
- training loop
- hybrid integration

Deliverables:
- MWPM baseline implementation or integration
- lightweight learned decoder
- hybrid learned + MWPM decoder
- checkpoint and training config support

### Tengxiang Lin
Primary ownership:
- CLI orchestration
- benchmarking
- reporting
- demo packaging

Deliverables:
- `autoqec` CLI command surface
- `run` workflow orchestration
- latency and throughput benchmarking
- Pareto/report artifact generation
- demo command polish

### Shared checkpoints
- End of Day 2: baseline end-to-end path works
- End of Day 4: hybrid decoder path works
- End of Day 5: full benchmark matrix runs
- End of Day 6: demo is rehearsable from clean checkout

## 18. Fallback plans

If the learned decoder becomes unstable or too costly:
- preserve the AI-enhanced story with a lightweight trainable message-passing or parameterized reweighting decoder
- keep the same CLI, artifact layout, and evaluation flow
- do not expand the search space to compensate

If the correlated-noise variant is costly:
- keep one minimal correlated template only
- do not broaden noise coverage

If reporting polish is delayed:
- prioritize one clean summary figure and one machine-readable metrics table

## 19. Final success definition

The hackathon MVP succeeds if a clean Rust command can run the full surface-code workflow and produce:
- benchmark metrics
- latency-quality comparisons
- and at least one clean Pareto plot or Pareto-style report artifact

The MVP does not need to solve the full AutoQEC vision. It needs to demonstrate a compelling, reproducible Rust-first slice of that vision.
