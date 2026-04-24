"""Internal tool for demo-3 maintainers.

Refreshes ``expected_output/`` from the latest ``runs/demo38-sample/`` and a
user-provided walkthrough stdout. Strips host-specific absolute paths so the
committed snapshot is portable.

Usage::

    bash demos/demo-3-worktree-provenance/run.sh --run-id demo38-sample > walk.txt
    python demos/demo-3-worktree-provenance/_refresh_expected.py walk.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO = REPO_ROOT / "demos" / "demo-3-worktree-provenance" / "expected_output"
RUN = REPO_ROOT / "runs" / "demo38-sample"

WORKTREE_WIN = str(REPO_ROOT)
WORKTREE_FWD = WORKTREE_WIN.replace("\\", "/")
# Attempt to locate the main repo's venv; fall back to ignoring it.
_GCDIR = subprocess.check_output(
    ["git", "-C", str(REPO_ROOT), "rev-parse", "--git-common-dir"],
    text=True,
    encoding="utf-8",
).strip()
_MAIN_REPO = Path(_GCDIR).resolve().parent
VENV_WIN = str(_MAIN_REPO / ".venv")
VENV_FWD = VENV_WIN.replace("\\", "/")


def sanitize(text: str) -> str:
    return (
        text.replace(WORKTREE_WIN, "<REPO_ROOT>")
            .replace(WORKTREE_FWD, "<REPO_ROOT>")
            .replace(VENV_WIN, "<VENV>")
            .replace(VENV_FWD, "<VENV>")
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("walkthrough", type=Path, help="stdout captured from run.sh")
    p.add_argument(
        "--branch",
        default="exp/demo38-sample/01-idea-a",
        help="branch to read round_1_pointer.json from",
    )
    args = p.parse_args()

    DEMO.mkdir(parents=True, exist_ok=True)

    # Refresh per-round metrics.json
    for round_idx in (1, 2, 3):
        src = RUN / f"round_{round_idx}" / "metrics.json"
        if not src.exists():
            print(f"skip round_{round_idx}: {src} missing")
            continue
        metrics = json.loads(src.read_text(encoding="utf-8"))
        for key in ("checkpoint_path", "training_log_path"):
            if isinstance(metrics.get(key), str):
                metrics[key] = sanitize(metrics[key])
        dst = DEMO / f"round_{round_idx}_metrics.json"
        dst.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"wrote {dst.name}")

    # Alias for backwards compatibility: metrics.json == round_1
    if (DEMO / "round_1_metrics.json").exists():
        (DEMO / "metrics.json").write_text(
            (DEMO / "round_1_metrics.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print("wrote metrics.json (alias)")

    # round_1_pointer.json via git show
    try:
        pointer = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "show",
                f"{args.branch}:round_1/round_1_pointer.json",
            ],
            text=True,
            encoding="utf-8",
        )
        (DEMO / "round_1_pointer.json").write_text(pointer, encoding="utf-8")
        print("wrote round_1_pointer.json")
    except subprocess.CalledProcessError:
        print(f"warning: could not read {args.branch}:round_1/round_1_pointer.json")

    # Walkthrough stdout
    text = args.walkthrough.read_text(encoding="utf-8", errors="replace")
    (DEMO / "run_demo.stdout.txt").write_text(sanitize(text), encoding="utf-8")
    print("wrote run_demo.stdout.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
