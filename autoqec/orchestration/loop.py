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
) -> dict:
    """Return everything needed to drive one round of the research loop.

    The caller fills in the Ideator/Coder/Analyst responses between stages
    (inline or via subprocess). The Coder/Analyst prompts are assembled
    lazily after their upstream step completes, so only the Ideator prompt
    is materialised up front here.
    """
    run_dir = Path(run_dir)
    mem = RunMemory(run_dir)
    round_dir = run_dir / f"round_{round_idx}"

    ideator_ctx = mem.l3_for_ideator(
        env_spec=env_spec.model_dump(),
        kb_excerpt=kb_excerpt,
        machine_state=machine_state,
    )

    return {
        "round_idx": round_idx,
        "round_dir": str(round_dir),
        "ideator_prompt": build_prompt("ideator", ideator_ctx),
        "dsl_schema_md": dsl_schema_md,  # forwarded to the Coder when its turn arrives
    }


def build_coder_prompt(
    hypothesis: dict,
    mem: RunMemory,
    dsl_schema_md: str,
    best_so_far: list[dict] | None = None,
) -> str:
    """Build the Coder prompt after the Ideator returns a hypothesis.

    `best_so_far` defaults to the current Pareto (top 3). Callers can
    override when they want a tighter "dominant configs only" slice.
    """
    if best_so_far is None:
        pareto = json.loads(mem.pareto_path.read_text(encoding="utf-8") or "[]")
        best_so_far = pareto[:3]
    ctx = mem.l3_for_coder(
        hypothesis=hypothesis,
        schema_md=dsl_schema_md,
        best_so_far=best_so_far,
    )
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
