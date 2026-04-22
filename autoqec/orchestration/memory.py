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


class RunMemory:
    """L1/L2 bridge. L3 is assembled on the fly when dispatching."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.run_dir / "history.jsonl"
        self.log_path = self.run_dir / "log.md"
        self.pareto_path = self.run_dir / "pareto.json"
        if not self.pareto_path.exists():
            self.pareto_path.write_text("[]")

    # ─── L1 writes ───────────────────────────────────────────────────

    def append_round(self, record: dict) -> None:
        """Append one round record to history.jsonl (one JSON object per line)."""
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def append_log(self, md: str) -> None:
        """Append a markdown snippet to the run's narrative log."""
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(md + "\n")

    def update_pareto(self, pareto: list[dict]) -> None:
        """Replace the pareto.json file with the given front."""
        self.pareto_path.write_text(json.dumps(pareto, indent=2))

    # ─── L2 summary (rebuilt each round from L1) ─────────────────────

    def _load_history(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        with self.history_path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def l2_snapshot(self, k_last: int = 3) -> dict:
        history = self._load_history()
        pareto = json.loads(self.pareto_path.read_text() or "[]")
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
    ) -> dict:
        """Assemble the Ideator context.

        `last_5_hypotheses` carries status + delta_ler so the Ideator can
        avoid re-proposing killed hypotheses without explicit justification.
        """
        history = self._load_history()
        pareto = json.loads(self.pareto_path.read_text() or "[]")
        last_5 = [
            {
                "hypothesis": r.get("hypothesis"),
                "status": r.get("status"),
                "delta_ler": r.get("delta_ler"),
            }
            for r in history[-5:]
        ]
        return {
            "env_spec": env_spec,
            "pareto_front": pareto[:5],
            "last_5_hypotheses": last_5,
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
        }

    def l3_for_analyst(
        self,
        round_dir: Path | str,
        prev_summary: str,
        pareto: list[dict],
    ) -> dict:
        return {
            "metrics_path": str(Path(round_dir) / "metrics.json"),
            "previous_summary": prev_summary,
            "pareto_front": pareto,
        }
