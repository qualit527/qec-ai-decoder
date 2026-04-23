"""Single-round driver (Task A2.2).

Assembles the L3 context for one research round and emits the Ideator
prompt + bookkeeping paths as JSON. The caller (inline `Agent` tool or a
future subprocess router) fills in the Ideator/Coder/Analyst responses.

This script does NOT invoke any LLM itself. Day-3 will add the wrapper
that chains responses through the Runner.

Use
---

    python scripts/run_single_round.py \\
        --env-yaml autoqec/envs/builtin/surface_d5_depol.yaml \\
        --run-dir runs/demo-1 --round-idx 1 [--budget-s 3600]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when the script is run directly
# (e.g. `python scripts/run_single_round.py ...`) without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autoqec.envs.schema import load_env_yaml  # noqa: E402
from autoqec.orchestration.loop import run_round_plan  # noqa: E402
from autoqec.tools.machine_state import machine_state  # noqa: E402

# Anchor knowledge excerpts to the repo root so running from a foreign
# cwd doesn't silently drop them from the Ideator prompt (Codex review).
KB_PATH = _REPO_ROOT / "knowledge/DECODER_ROADMAP.md"
SPEC_PATH = _REPO_ROOT / "docs/specs/2026-04-20-autoqec-design.md"
KB_EXCERPT_MAX_CHARS = 3000


def _read_text_best_effort(path: Path, max_chars: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return text[:max_chars] if max_chars else text


def plan_for_round(
    env_yaml: str | Path,
    run_dir: str | Path,
    round_idx: int,
    budget_s: float | None = None,
) -> dict:
    env = load_env_yaml(env_yaml)
    run_dir = Path(run_dir)
    ms = machine_state(run_dir, total_wallclock_s_budget=budget_s)
    kb = _read_text_best_effort(KB_PATH, KB_EXCERPT_MAX_CHARS)
    dsl_md = _read_text_best_effort(SPEC_PATH)
    return run_round_plan(
        env_spec=env,
        run_dir=run_dir,
        round_idx=round_idx,
        machine_state=ms,
        kb_excerpt=kb,
        dsl_schema_md=dsl_md,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-yaml", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--round-idx", type=int, required=True)
    p.add_argument("--budget-s", type=float, default=None, help="outer wall-clock budget in seconds")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = plan_for_round(
        env_yaml=args.env_yaml,
        run_dir=args.run_dir,
        round_idx=args.round_idx,
        budget_s=args.budget_s,
    )
    sys.stdout.write(json.dumps(plan, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
