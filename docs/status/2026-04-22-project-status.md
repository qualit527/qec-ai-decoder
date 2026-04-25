# AutoQEC Status — 2026-04-22 (Day-2 end)

**Audience:** project admin / advisor.
**Scope:** status of the AutoQEC research harness at the close of Day-2 of
a 3-day sprint. Written for someone who wants the picture without reading
every PR.

---

## 1. What AutoQEC is

A self-driving research harness for **quantum error correction decoders**.
It takes an environment (code × noise × compute constraints) and, through
a multi-round loop of LLM sub-agents + a training runner, discovers neural
*predecoders* that sit in front of a classical decoder (PyMatching /
BP-OSD) and improve the logical error rate (Δ LER).

One research round =
`Ideator → Coder → Runner (train + eval) → Analyst → Verifier`.
The Ideator proposes an architectural hypothesis, the Coder turns it into
a validated DSL config, the Runner trains + evaluates, the Analyst
summarises, the Verifier independently re-evaluates on a held-out seed
range. Candidates that survive verification feed a Pareto front of
(Δ LER, FLOPs, n_params).

The MVP ships two reference environments:

| Env | Code | Classical backend | Owner of env |
|---|---|---|---|
| `surface_d5_depol` | rotated d=5 surface code, circuit-level depolarising noise | MWPM (PyMatching) | Chen Jiahan |
| `bb72_depol` | bivariate-bicycle [[72, 12, 6]] qLDPC | OSD (via ldpc) | Xie Jingu |

Authoritative documents (frozen):

- `docs/superpowers/specs/2026-04-20-autoqec-design.md` — v2.2 spec, source of truth.
- `docs/contracts/interfaces.md` — 6 frozen interfaces (EnvSpec, RunnerConfig + RoundMetrics, VerifyReport, PredecoderModule I/O, subagent JSON shapes, skill CLI).
- `docs/contracts/round_dir_layout.md` — who writes what inside `runs/<run_id>/round_<N>/`.

---

## 2. Development flow

### 2.1 Spec → plan → TDD → review → merge

Every substantive change goes through the same cycle:

1. **Spec-level decision** lives in the frozen v2.2 design doc.
2. **Plan** — a per-owner plan file (`docs/superpowers/plans/…`) breaks
   the spec into bite-sized tasks (A1.4, A2.1, …). Each task is one
   commit group.
3. **Red/Green TDD** — failing unit test first, then implementation,
   then green test. Contracts (pydantic schemas, JSON shapes) are
   encoded as tests so drift fails CI rather than silently.
4. **Cross-model review** — `/codex-review` ships the diff to GPT-5.4
   via Codex MCP for an independent second opinion. Findings are
   addressed in a follow-up commit on the same branch before merge.
5. **Merge** — `feat/<owner>-<topic>` branch → PR → main. PRs are kept
   small (one phase at a time) so review is tractable.

### 2.2 What "frozen" actually means

The Phase-0 contract freeze is not a doc promise — it is enforced in
Python. `autoqec/envs/schema.py`, `autoqec/runner/schema.py`,
`autoqec/eval/schema.py`, and `autoqec/agents/schemas.py` are pydantic
models. `parse_response()` validates every subagent JSON payload against
the per-role model before it can reach `history.jsonl`. Contract drift
is a test failure, not a PR-review judgement call.

### 2.3 Cross-model review = real signal, not theatre

Both audits of Chen Jiahan's work produced actionable findings, not
rubber stamps:

- Day-1 Codex review caught 1 HIGH + 3 MED + 1 LOW + 1 open question
  (most notable: the Coder prompt told the sub-agent to emit a bare
  string for Tier-2 custom functions, but the runtime schema required
  an object — every Tier-2 round would have crashed at compile time).
- Day-2 Codex review caught 4 MED + 2 open questions (script
  brittleness outside the repo root, an unguarded CUDA probe, a doc
  that overclaimed its own test coverage).

All findings were closed in follow-up commits before merge.

---

## 3. Collaboration shape

### 3.1 Three owners, disjoint subtrees, frozen contracts between them

| Owner | AI agent | Primary subtree | Secondary (delivery) |
|---|---|---|---|
| Chen Jiahan | Claude Code (Opus 4.7, 1M ctx) | `autoqec/envs/`, `autoqec/orchestration/`, `autoqec/agents/`, `.claude/agents/` | `/autoqec-run`, `/add-env`, Demo 1 |
| Lin Tengxiang | Codex CLI (GPT-5.4) | `autoqec/decoders/`, `autoqec/runner/`, `cli/` | Demo 2 (bb72 templates), Makefile + CLI polish |
| Xie Jingu | (TBD — currently using Codex) | `autoqec/eval/`, `circuits/bb72_*`, `autoqec/decoders/baselines/bposd*` | `/verify-decoder`, `/review-log`, `/diagnose-failure`, Demo 4, Demo 5 |

The subtrees are disjoint, so merges rarely collide. When one owner
needs a torch-free constant the other owns, the pattern is: extract
constants to a lightweight module (`autoqec/decoders/custom_fn_rules.py`),
have the heavy module re-export them — preserves the original API, lets
the consumer import without pulling torch. No owner has unilaterally
modified another owner's code.

### 3.2 Execution principle — no one does only glue

From the spec's §12.1: each owner has to touch real QEC artifacts as
well as one delivery-facing surface. This prevents any one role
becoming the "prompts and docs" person. Example: Chen Jiahan is the
orchestration owner but personally ran the 1M-shot PyMatching baseline
(`LER = 0.01394` at p = 5e-3, seed 42) that the whole team benchmarks Δ
LER against.

### 3.3 Synchronous coordination points

- **Phase-0 sync** (before Day-1 coding) — walk through
  `interfaces.md`, edit in the meeting, merge before anyone scaffolds.
- **Pre-integration handshake** (Day-2) — the A2.1 e2e_handshake
  script is the shared sandbox proving orchestration ↔ Runner works
  with a hand-written Tier-1 config, *before* any LLM is wired in.
- **Cross-review** — each owner reviews at least one other owner's QEC
  artifact, not just prompts or docs.

### 3.4 What's explicitly **not** in the collaboration model

- No shared IDE sessions. Every owner drives their own agent.
- No "one agent to rule them all". The agents above know their subtree
  deeply and coordinate through the frozen contracts.
- No informal interface changes. Any edit to `interfaces.md` or the
  pydantic schemas requires 3-of-3 owner sign-off on the PR.

---

## 4. Harness system state (as of 2026-04-22, end of Day-2)

### 4.1 Merged to `main`

| PR | Content | Merge date |
|---|---|---|
| #2 | v2.2 spec + per-owner plans + Day-1 task briefs | 2026-04-21 |
| #3 | Romanise team member names (Chinese → pinyin) for CI | 2026-04-21 |
| #4 | Chen's Day-1 — benchmark, sub-agent prompts, orchestration skeleton, pydantic §2.5 | 2026-04-22 |
| #5 | Chen's Day-2 — e2e_handshake, machine_state, run_single_round, round-dir layout | 2026-04-22 |
| (earlier) | Lin's Runner slice + DSL compiler + GNN/Neural-BP templates | (before PR numbering) |

### 4.2 Capabilities available today

- **End-to-end baseline**: `python scripts/benchmark_surface_baseline.py` runs 1M-shot PyMatching on `surface_d5` and emits a JSON reference (`demos/demo-1-surface-d5/expected_output/baseline_benchmark.json`, LER = 0.01394 at p = 5e-3).
- **No-LLM handshake**: `python scripts/e2e_handshake.py --round-dir <dir>` runs the full Runner on a hand-written Tier-1 GNN config. Proves the orchestration ↔ Runner interface end-to-end.
- **Round planner**: `python scripts/run_single_round.py --env-yaml … --run-dir … --round-idx N` assembles the Ideator's L3 context (env + Pareto + history + machine-state + knowledge excerpts) and prints the prompt as JSON.
- **Machine-state probe**: `machine_state(run_dir, total_wallclock_s_budget=…)` returns GPU free-VRAM + history-derived round timings + budget accounting. Gracefully returns `{}` on any CUDA/driver failure.
- **Runner (Lin)**: `python -m cli.autoqec run <env.yaml> --rounds N --profile dev --no-llm` picks a random seed template from `autoqec/example_db/`, trains it, writes `metrics.json` + `checkpoint.pt` + `train.log`.
- **Schemas enforced**: six frozen contracts live as pydantic models, validated on every round.

### 4.3 Not yet built

| Gap | Owner | Planned |
|---|---|---|
| Ideator/Coder/Analyst wired to real LLM calls through `Agent` tool | Chen | Day-3 |
| `/autoqec-run` + `/add-env` SKILL.md | Chen | Day-3 |
| Demo 1 README + run.sh + walkthrough (surface_d5 full demo) | Chen | Day-3 |
| `independent_eval` + `/verify-decoder` | Xie | Day-3 |
| Demo 2 (bb72 3-round dev-profile run) | Lin | Day-3 |
| `/review-log` + `/diagnose-failure` | Xie | Day-3 |
| Runner-side UTF-8 file opens + absolute-path `.resolve()` in `cli/autoqec.py::run` | Lin | [TODO-fill-in] flagged in `docs/contracts/round_dir_layout.md` |

### 4.4 Test posture

- **33 unit tests** on the CPU-only sweep for Chen's subtree (orchestration + agents + tools + scripts) — all green.
- **1 integration test** for the full `e2e_handshake` training round — currently `@pytest.mark.integration`-gated; needs a torch-enabled CI runner to light up.
- Lin's `tests/test_runner_smoke.py` is also `integration`-gated; runs the full training path on a minimal GNN.

### 4.5 Risks the admin should know about

- **No GPU integration test is being exercised yet.** The orchestration side is green in a torch-free env; the Runner side has an integration-marked smoke test but no one has flipped it on yet. Day-3's first deliverable is to run it.
- **Day-3 scope is large.** Four skills + one full Demo + Verifier slice, across three owners. Buffer is thin.
- **`cli/autoqec.py::run` passes a relative `round_dir`.** Every round's `metrics.json` then records `checkpoint_path` as relative, which breaks the Verifier's assumption it can be loaded from any cwd. One-line fix on Lin's side; flagged in the layout doc but not yet in.

---

## 5. Glossary for non-QEC readers

- **QEC** — quantum error correction. The logical qubit is encoded across many physical ones; errors are detected through syndromes (indirect measurement).
- **Decoder** — given a syndrome, guesses the most likely error and emits a correction.
- **Predecoder** — a learned front-end that rewrites the syndrome or its priors before the classical decoder runs. Our research target.
- **Surface code** — the canonical topological QEC code. `d=5` means distance 5 (error-correcting "thickness"). Industry-standard baseline.
- **qLDPC / bb72** — quantum LDPC code, specifically bivariate-bicycle [[72, 12, 6]]. Harder than surface; tests that our harness isn't surface-specific.
- **Stim** — the de-facto Clifford simulator for QEC experiments. We use it for circuit generation, sampling, and `DetectorErrorModel` extraction.
- **PyMatching** — the standard MWPM decoder. Our surface-code baseline.
- **LER** — logical error rate. `Δ LER` is the improvement a predecoder produces over the pure-classical baseline; it is the headline metric.
- **Pareto** — the non-dominated front in (Δ LER, FLOPs, n_params). A "candidate" is a point on or near the front; "VERIFIED" means it survived an independent holdout evaluation.
