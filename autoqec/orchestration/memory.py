"""Three-layer memory for the AutoQEC research loop.

Layer map
---------

- **L1 (disk)**  — `history.jsonl`, `log.md`, `pareto.json` inside `run_dir`.
  Append-only, crash-safe. The single source of truth across rounds.
- **L2 (orchestrator context)** — a compact snapshot the orchestrator can
  carry round-to-round without ballooning token usage. Always rebuilt from
  L1, never stored separately.
- **L3 (per-subagent context)** — role-specific views assembled on demand
  when dispatching `autoqec-{ideator, coder, analyst}`.

This module owns L1 writes and the L2/L3 views. Subagents consume L3
payloads through `autoqec.agents.dispatch.build_prompt`.
"""
from __future__ import annotations

import json
from pathlib import Path


def tier2_validator_rules() -> dict:
    """Summarise `custom_fn_validator` constraints so the Coder subagent
    can honour its contract (see .claude/agents/autoqec-coder.md).

    Synthesised from `custom_fn_rules` (torch-free) so this works in lean
    environments that don't have pytorch installed.
    """
    from autoqec.decoders.custom_fn_rules import (
        ALLOWED_FROM_IMPORTS,
        ALLOWED_TOP_IMPORTS,
        FORBIDDEN_NAMES,
        SLOT_SIGNATURES,
    )

    return {
        "slot_signatures": dict(SLOT_SIGNATURES),
        "allowed_top_imports": sorted(ALLOWED_TOP_IMPORTS),
        "allowed_from_imports": sorted(ALLOWED_FROM_IMPORTS),
        "forbidden_names": sorted(FORBIDDEN_NAMES),
        "output_shape": {
            "type": "custom",
            "code": "<python source defining exactly one function with the slot signature>",
            "params_declared": {"<name>": "<type-hint str>"},
        },
    }


class RunMemory:
    """L1/L2 bridge. L3 is assembled on the fly when dispatching."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.jsonl"
        self.log_path = self.run_dir / "log.md"
        self.pareto_path = self.run_dir / "pareto.json"
        if not self.pareto_path.exists():
            self.pareto_path.write_text("[]", encoding="utf-8")

    # ─── L1 writes ───────────────────────────────────────────────────

    def append_round(self, record: dict) -> None:
        """Append one round record to history.jsonl (one JSON object per line).

        The record is validated through :class:`RoundMetrics` before the
        disk write so the §15.2 mutual-exclusion invariant
        (``round_attempt_id`` XOR ``reconcile_id``), the
        ``branch ⇒ commit_sha`` implication, and the
        ``compose_conflict ⇒ branch=None`` rule are enforced at ingress
        instead of silently polluting the log. Extra keys (``round``,
        ``hypothesis``, free-form ``note``) are preserved unchanged — only
        the schema-owned fields are re-validated.
        """
        # Lazy import to keep memory.py importable in torch-free contexts.
        from autoqec.runner.schema import RoundMetrics

        # Pydantic ignores unknown keys by default; validate a projection
        # and keep the original dict's shape when writing so callers'
        # extra metadata (e.g. ``hypothesis``, ``note``) survives.
        RoundMetrics.model_validate(record)
        with self.history_path.open("a", encoding="utf-8") as f:
            # ensure_ascii=False so Chinese / Δ / other non-ASCII survives
            f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")

    def append_log(self, md: str) -> None:
        """Append a markdown snippet to the run's narrative log."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(md + "\n")

    def update_pareto(self, pareto: list[dict]) -> None:
        """Replace the pareto.json file with the given front."""
        self.pareto_path.write_text(
            json.dumps(pareto, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ─── L2 summary (rebuilt each round from L1) ─────────────────────

    def _load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        with self.history_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def l2_snapshot(self, k_last: int = 3) -> dict:
        history = self._load_history()
        pareto = json.loads(self.pareto_path.read_text(encoding="utf-8") or "[]")
        return {
            "rounds_so_far": len(history),
            "pareto": pareto[:5],
            "last_rounds": history[-k_last:],
        }

    # ─── L3 views per subagent role ──────────────────────────────────

    def l3_for_ideator(
        self,
        env_spec: dict,
        kb_excerpt: str,
        machine_state: dict,
        run_id: str | None = None,
    ) -> dict:
        """Assemble the Ideator context including the full fork_graph (§15.4).

        The fork_graph replaces the previous `last_5_hypotheses` + `pareto_front`
        shape. It gives the Ideator a branch-aware view of the run so it can
        steer forks deliberately (pick a parent, see what worked on each
        branch, avoid dead ends).
        """
        from autoqec.orchestration.fork_graph import build_fork_graph

        history = self._load_history()
        pareto = json.loads(self.pareto_path.read_text(encoding="utf-8") or "[]")
        fork_graph = build_fork_graph(
            history=history, pareto=pareto, run_id=run_id or ""
        )
        return {
            "env_spec": env_spec,
            "fork_graph": fork_graph,
            "knowledge_excerpts": kb_excerpt,
            "machine_state_hint": machine_state,
        }

    def l3_for_coder(
        self,
        hypothesis: dict,
        schema_md: str,
        best_so_far: list[dict],
    ) -> dict:
        return {
            "hypothesis": hypothesis,
            "dsl_schema": schema_md,
            "best_so_far": best_so_far,
            "tier2_validator_rules": tier2_validator_rules(),
        }

    def l3_for_analyst(
        self,
        round_dir: Path | str,
        prev_summary: str,
        pareto: list[dict],
    ) -> dict:
        # analyst.md declares metrics_path as absolute; resolve() guarantees it
        # even when the caller passed a relative round_dir.
        metrics_abs = (Path(round_dir) / "metrics.json").resolve()
        return {
            "metrics_path": str(metrics_abs),
            "previous_summary": prev_summary,
            "pareto_front": pareto,
        }
