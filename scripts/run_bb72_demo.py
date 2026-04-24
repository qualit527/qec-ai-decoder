"""Cross-platform driver for the BB72 qLDPC demo.

Runs the no-LLM smoke loop on ``bb72_depol`` and prints a short summary.
The helper is intentionally testable so the bash wrapper can stay thin
while still working from nested git worktrees that share a top-level
``.venv``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_YAML = _REPO_ROOT / "autoqec/envs/builtin/bb72_depol.yaml"


@dataclass(frozen=True)
class ModeConfig:
    rounds: int
    profile: str


_MODE_DEFAULTS = {
    "fast": ModeConfig(rounds=1, profile="dev"),
    "dev": ModeConfig(rounds=3, profile="dev"),
    "prod": ModeConfig(rounds=10, profile="prod"),
}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-yaml", default=os.environ.get("ENV_YAML", str(DEFAULT_ENV_YAML)))
    p.add_argument("--mode", choices=tuple(_MODE_DEFAULTS), default=os.environ.get("MODE", "fast"))
    p.add_argument("--rounds", type=int, default=_env_int("ROUNDS"))
    # Allow profile overrides to include benchmark while leaving mode
    # defaults untouched for the existing fast/dev/prod presets.
    p.add_argument("--profile", choices=("dev", "prod", "benchmark"), default=os.environ.get("PROFILE"))
    p.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN"))
    return p.parse_args(argv)


def _mode_defaults(mode: str) -> ModeConfig:
    return _MODE_DEFAULTS[mode]


def _discover_python_bin(start_dir: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    for root in (start_dir, *start_dir.parents):
        for rel in (Path(".venv/bin/python"), Path(".venv/Scripts/python.exe")):
            candidate = root / rel
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def _latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.is_dir():
        return None
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _validated_env_yaml(raw: str) -> str:
    resolved = Path(raw).resolve()
    if not resolved.is_file():
        raise SystemExit(f"env_yaml not found or not a file: {resolved}")
    return str(resolved)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    env_yaml_checked = _validated_env_yaml(args.env_yaml)
    mode_cfg = _mode_defaults(args.mode)
    rounds = args.rounds if args.rounds is not None else mode_cfg.rounds
    profile = args.profile if args.profile is not None else mode_cfg.profile
    python_bin = _discover_python_bin(_REPO_ROOT, explicit=args.python_bin)

    subprocess.run(  # noqa: S603 - list-form, no shell; arguments validated above
        [
            python_bin,
            "-m",
            "cli.autoqec",
            "run",
            env_yaml_checked,
            "--rounds",
            str(rounds),
            "--profile",
            profile,
            "--no-llm",
        ],
        check=True,
        cwd=_REPO_ROOT,
        shell=False,
    )

    run_dir = _latest_run_dir(_REPO_ROOT / "runs")
    print()
    print("=== Demo 2 (bb72 qLDPC) complete ===")
    print(f"Mode: {args.mode} ({profile}, {rounds} round{'s' if rounds != 1 else ''})")
    if run_dir is None:
        print("No runs/ directory found - check the CLI output above.")
        return 1
    print(f"Run dir: {run_dir.relative_to(_REPO_ROOT)}")
    print("Routing: autoqec/envs/builtin/bb72_depol.yaml -> classical_backend: osd")

    hist_path = run_dir / "history.jsonl"
    if hist_path.exists():
        completed_rounds = sum(1 for line in hist_path.read_text(encoding="utf-8").splitlines() if line.strip())
        print(f"History: {completed_rounds} rounds")

    candidate_pareto_path = run_dir / "candidate_pareto.json"
    if candidate_pareto_path.exists():
        print(
            "Candidate Pareto:",
            json.loads(candidate_pareto_path.read_text(encoding="utf-8")),
        )
    else:
        print("Candidate Pareto: (none yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
