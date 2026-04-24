"""Research-loop driver skeleton.

In inline (Claude Code chat) mode the orchestrator is the main session and
Python just hands it prompts via `run_round_plan`. A second and third
prompt for the Coder and Analyst are assembled lazily by
`build_coder_prompt` and `build_analyst_prompt` after the Ideator
response has arrived.

This file intentionally avoids any LLM call itself — it only produces the
prompt strings + bookkeeping paths. The caller injects the model.
"""
from __future__ import annotations

import json
from pathlib import Path

from autoqec.agents.dispatch import build_prompt
from autoqec.envs.schema import EnvSpec
from autoqec.orchestration.memory import RunMemory


def run_round_plan(
    env_spec: EnvSpec,
    run_dir: Path | str,
    round_idx: int,
    machine_state: dict,
    kb_excerpt: str,
    dsl_schema_md: str,
    fork_from: str | list[str] = "baseline",
) -> dict:
    """Return everything needed to drive one round of the research loop.

    The caller fills in the Ideator/Coder/Analyst responses between stages
    (inline or via subprocess). The Coder/Analyst prompts are assembled
    lazily after their upstream step completes, so only the Ideator prompt
    is materialised up front here.

    `fork_from` names the parent branch the Coder will fork its worktree
    from (§15.4). Passed through verbatim to the worktree-creation step.
    """
    run_dir = Path(run_dir)
    mem = RunMemory(run_dir)
    round_dir = run_dir / f"round_{round_idx}"

    ideator_ctx = mem.l3_for_ideator(
        env_spec=env_spec.model_dump(),
        kb_excerpt=kb_excerpt,
        machine_state=machine_state,
        run_id=run_dir.name,
    )

    return {
        "round_idx": round_idx,
        "round_dir": str(round_dir),
        "ideator_prompt": build_prompt("ideator", ideator_ctx),
        "dsl_schema_md": dsl_schema_md,  # forwarded to the Coder when its turn arrives
        "fork_from": fork_from,  # passed forward to worktree creation
    }


def _cold_start_best_so_far(mem: RunMemory, top_k: int = 3) -> list[dict]:
    """Cold-start fallback when ``pareto.json`` is empty.

    The Pareto only admits rounds with ``verify_verdict == "VERIFIED"``,
    which requires the Analyst to say ``candidate`` first, which in turn
    requires ``delta_ler + 0.5*(ci_high-ci_low) > 0``. Until a round
    clears that bar the Pareto is ``[]`` and the Coder previously saw an
    empty ``best_so_far`` — no reference configs to mutate from. We
    instead surface the top-``top_k`` ``status=ok`` rows by ``delta_ler``
    so the Coder has *something* to steer off of, each entry flagged
    ``cold_start_fallback=True`` so the Coder knows it is NOT VERIFIED.
    """
    if not mem.history_path.exists():
        return []
    rows: list[dict] = []
    with mem.history_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    ok = [r for r in rows if r.get("status") == "ok" and r.get("delta_ler") is not None]
    # Stable descending sort by delta_ler — None handled by the filter above.
    ok.sort(key=lambda r: float(r["delta_ler"]), reverse=True)
    out: list[dict] = []
    for r in ok[:top_k]:
        out.append(
            {
                "round": r.get("round"),
                "delta_ler": r.get("delta_ler"),
                "flops_per_syndrome": r.get("flops_per_syndrome"),
                "n_params": r.get("n_params"),
                "train_loss_final": r.get("train_loss_final"),
                "checkpoint_path": r.get("checkpoint_path"),
                "cold_start_fallback": True,
            }
        )
    return out


def build_coder_prompt(
    hypothesis: dict,
    mem: RunMemory,
    dsl_schema_md: str,
    best_so_far: list[dict] | None = None,
    worktree_dir: str | None = None,
) -> str:
    """Build the Coder prompt after the Ideator returns a hypothesis.

    `best_so_far` defaults to the current Pareto (top 3); when the Pareto
    is empty (cold-start, no VERIFIED round yet) it falls back to the top
    3 ``status=ok`` history rows by ``delta_ler`` so the Coder always
    has a reference config to steer off of.

    `worktree_dir`, when supplied, is threaded into the Coder ctx so the
    subagent knows where to make edits + commit (§15.4).
    """
    if best_so_far is None:
        pareto = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
        best_so_far = pareto[:3]
        if not best_so_far:
            best_so_far = _cold_start_best_so_far(mem)
    ctx = mem.l3_for_coder(
        hypothesis=hypothesis,
        schema_md=dsl_schema_md,
        best_so_far=best_so_far,
    )
    if worktree_dir:
        ctx["worktree_dir"] = worktree_dir
    return build_prompt("coder", ctx)


def build_analyst_prompt(
    mem: RunMemory,
    round_dir: Path | str,
    prev_summary: str,
) -> str:
    """Build the Analyst prompt once the Runner has written metrics.json.

    Reads the current Pareto from disk so the Analyst can classify the
    round against it without the caller threading state through.
    """
    pareto = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
    ctx = mem.l3_for_analyst(
        round_dir=round_dir,
        prev_summary=prev_summary,
        pareto=pareto,
    )
    return build_prompt("analyst", ctx)
