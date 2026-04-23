# Open Questions Log — Test Plan Resolution

Resolution log for the six Open Questions raised in
[`human-verification-test-plan.md`](./human-verification-test-plan.md#open-questions).
Each entry captures the decision, the evidence or reasoning behind it,
and the commit or PR that closed it. Owners should update this file
alongside the PR that removes their row from the plan's open-questions
section.

---

## O-1 — CI runner: CPU vs self-hosted GPU

**Status:** Resolved (Chen, 2026-04-23)
**Decision:** CI stays on the GitHub-hosted Ubuntu CPU runner. Integration
tests (`--run-integration --model-path …`) remain opt-in and are not run
in CI.

**Reasoning:**
- The `"not integration"` marker covers 226 tests and reaches 95.73 % line
  coverage on CPU alone, which already exceeds the project's enforced
  95 % floor.
- Live-GPU and live-LLM paths are Phase 4 concerns gated on the
  Ideator → Coder → Analyst DAG landing. CPU-runner CI is the wrong
  vehicle for those — they need separate infra (advisor-replay
  machines, model-path fixtures) that does not belong in every PR gate.
- A self-hosted GPU runner would raise maintenance cost (token budgets,
  driver drift, concurrency caps) without unblocking any checkbox the
  CPU runner does not already handle.

**Follow-up:** if Lin's live DAG lands before demo day, add a manually-triggered
`workflow_dispatch` job that runs integration tests on-demand; do not
wire it into the default push/PR event set.

---

## O-2 — Phase 4 wall-clock + token budgets

**Status:** Deferred until live DAG lands (Chen, 2026-04-23)
**Decision:** Budgets stay provisional (75 min surface_d5, 120 min
bb72; 500 K input / 80 K output per 3 rounds). Any budget-enforcement
check in Phase 4 emits a warning only — not a failure — until a
calibration dry-run produces real numbers.

**Reasoning:**
- No calibration data exists today. Phase 4 is blocked on upstream
  work (Lin's LLM dispatch); Chen cannot record credible budgets on
  zero live rounds.
- The test plan explicitly uses "暂时保持" / "provisional" language for
  these numbers so Phase 4 is not falsely failed on arbitrary thresholds.

**Follow-up:** once the live DAG is runnable, do a 1-round dry-run on
`surface_d5_depol` with each of the two backend pairings
(Ideator=codex, Analyst=claude) and (both claude). Record wallclock +
input / output tokens into a new `docs/verification/phase-4-budget-calibration.md`.
Promote the numbers to hard thresholds only after two independent
dry-runs agree within ±20 %.

---

## O-4 — Advisor walkthrough: live re-run vs recorded replay

**Status:** Resolved (Chen, 2026-04-23)
**Decision:** Recorded replay is the primary demo path. A live re-run
on the advisor's request remains a stretch goal conditional on live DAG
availability on the day.

**Reasoning:**
- Phase 5.5.3 and 5.5.4 already require offline replay to succeed with
  all `AUTOQEC_*_BACKEND` env vars unset. A recorded `runs/<id>.tar.gz`
  plus the repo SHA is therefore guaranteed to reproduce the numbers
  without any network dependency.
- A recorded replay is deterministic (same inputs → same outputs) and
  gives the advisor an artifact they can re-run themselves; a live demo
  rolls the dice on network, rate-limit, and backend availability at a
  single fixed moment.
- The walkthrough value (confirming the audit trail, Pareto math, and
  safety containment) is served equally well by replay; a live
  tie-breaker can be offered only if everything else works.

**Follow-up:** Chen prepares the tarball ahead of Day 3 from the most
recent green Phase 3 run. README walk-through (Phase 4 section) links
the tarball under `demos/demo-1-surface-d5/expected_output/` once a
verified run exists.

---

## Other Open Questions

O-3 (reward-hacking fixtures), O-5 (`diagnose.md` vs JSON contract),
O-6 (`cleanup-worktree` CLI wrapper) belong to Xie and Lin respectively;
their resolution lives in the issues they own (`#31` for Xie, `#30` for
Lin) and will be logged here as they close.
