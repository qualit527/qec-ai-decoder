"""Cross-platform equivalent of demos/demo-1-surface-d5/run_quick.sh.

Runs the no-LLM smoke loop on surface_d5_depol (or a user-provided env)
and prints a short summary. Works on Windows PowerShell, macOS, and
Linux without relying on bash, `ls | head`, `wc`, or `/tmp`.

Usage::

    python scripts/run_quick.py                       # defaults
    python scripts/run_quick.py --rounds 5 --profile prod
    python scripts/run_quick.py --env-yaml envs/my.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_YAML = _REPO_ROOT / "autoqec/envs/builtin/surface_d5_depol.yaml"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-yaml", default=str(DEFAULT_ENV_YAML))
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--profile", choices=("dev", "prod"), default="dev")
    return p.parse_args()


def _latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.is_dir():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _validated_env_yaml(raw: str) -> str:
    """Accept `raw` only if it's an existing file path.

    subprocess.run below uses list-form (no shell), so shell metacharacters
    are never interpreted — command injection in the classic sense is not
    possible. We still validate here because:
      (a) it fails fast with a clear error if the user mistyped the path,
          rather than the Runner crashing three frames deeper, and
      (b) it closes the "hostile filename" hole for Sourcery/Bandit-style
          static scanners without touching the subprocess call shape.
    """
    resolved = Path(raw).resolve()
    if not resolved.is_file():
        raise SystemExit(f"env_yaml not found or not a file: {resolved}")
    return str(resolved)


def main() -> int:
    args = _parse_args()
    env_yaml_checked = _validated_env_yaml(args.env_yaml)
    # argparse already restricts args.profile to ('dev','prod') and args.rounds
    # to int, so every element below is either a literal or a validated value.
    subprocess.run(  # noqa: S603 — list-form, no shell; inputs validated above
        [
            sys.executable,
            "-m",
            "cli.autoqec",
            "run",
            env_yaml_checked,
            "--rounds",
            str(args.rounds),
            "--profile",
            args.profile,
            "--no-llm",
        ],
        check=True,
        cwd=_REPO_ROOT,
        shell=False,
    )

    run_dir = _latest_run_dir(_REPO_ROOT / "runs")
    print()
    print("=== Demo 1 (no-LLM) complete ===")
    if run_dir is None:
        print("No runs/ directory found — check the CLI output above.")
        return 1
    print(f"Run dir: {run_dir.relative_to(_REPO_ROOT)}")

    hist_path = run_dir / "history.jsonl"
    if hist_path.exists():
        rounds = sum(1 for line in hist_path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"History: {rounds} rounds")

    candidate_pareto_path = run_dir / "candidate_pareto.json"
    if candidate_pareto_path.exists():
        print(
            "Candidate Pareto:",
            json.loads(candidate_pareto_path.read_text(encoding="utf-8")),
        )
    else:
        print("Candidate Pareto:  (none yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
