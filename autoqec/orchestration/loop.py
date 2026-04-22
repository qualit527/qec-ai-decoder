"""Research-loop driver skeleton.

In inline (Claude Code chat) mode the orchestrator is the main session and
Python just hands it prompts via `run_round_plan`. In a future background
mode (Day-2/3) the same return dict drives a subprocess dispatcher.

This file intentionally avoids any LLM call itself — it only produces the
prompt strings + bookkeeping paths. The caller injects the model.
"""
from __future__ import annotations

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
