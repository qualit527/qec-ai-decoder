"""End-to-end demo of the §15 worktree experiment model.

Runs one research round inside an isolated git worktree / branch, then
inspects the provenance left behind:

    Ideator fork decision (simulated)
      -> create_round_worktree(run_id, round_idx=1, slug)
      -> cli.autoqec run-round --code-cwd ... --branch ... --fork-from ...
      -> Runner subprocess writes metrics.json
      -> subprocess_runner commits round_1/round_1_pointer.json on the branch
      -> cleanup_round_worktree removes the checkout, keeps the branch
      -> reconcile_at_startup would auto-heal from the pointer after a crash

No LLM, no Ideator dispatch — the predecoder config comes from a YAML fixture
in this directory so the demo is deterministic and offline.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = Path(__file__).resolve().parent
ENV_YAML = REPO_ROOT / "autoqec" / "envs" / "builtin" / "surface_d5_depol.yaml"
FIXTURE_CONFIG = DEMO_DIR / "fixture_config.yaml"

# Make the demo runnable from any venv that has the deps: import the
# worktree's own autoqec package instead of whatever happens to be
# pip-installed (which may point at a different checkout).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _hr(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args], text=True
    ).rstrip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=time.strftime("demo38-%Y%m%d-%H%M%S"),
        help="Run identifier; drives branch name exp/<run_id>/01-<slug>",
    )
    parser.add_argument(
        "--slug", default="smoke", help="Slug appended to the branch name"
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter that will invoke cli.autoqec run-round",
    )
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Skip cleanup_round_worktree so the checkout stays on disk",
    )
    parser.add_argument(
        "--keep-branch",
        action="store_true",
        help="Skip the final git branch -D suggestion line",
    )
    args = parser.parse_args()

    from autoqec.orchestration.worktree import (
        cleanup_round_worktree,
        create_round_worktree,
    )

    round_attempt_id = str(uuid.uuid4())
    run_id = args.run_id
    round_idx = 1
    fork_from = _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")

    _hr("1. Simulated Ideator decision")
    print(f"run_id             = {run_id}")
    print(f"round_idx          = {round_idx}")
    print(f"slug               = {args.slug}")
    print(f"fork_from          = {fork_from}")
    print(f"round_attempt_id   = {round_attempt_id}")

    _hr("2. create_round_worktree")
    plan = create_round_worktree(
        repo_root=REPO_ROOT,
        run_id=run_id,
        round_idx=round_idx,
        slug=args.slug,
        fork_from=fork_from,
    )
    print(f"branch             = {plan['branch']}")
    print(f"worktree_dir       = {plan['worktree_dir']}")

    round_dir = REPO_ROOT / "runs" / run_id / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    try:
        _hr("3. cli.autoqec run-round (subprocess dispatch -> worktree)")
        cmd = [
            args.python,
            "-m",
            "cli.autoqec",
            "run-round",
            str(ENV_YAML),
            str(FIXTURE_CONFIG),
            str(round_dir),
            "--profile",
            "dev",
            "--code-cwd",
            plan["worktree_dir"],
            "--branch",
            plan["branch"],
            "--fork-from",
            fork_from,
            "--round-attempt-id",
            round_attempt_id,
        ]
        print("$ " + " ".join(cmd))
        sys.stdout.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
        if proc.stdout:
            # The Runner contract is "child prints RoundMetrics JSON to stdout"
            # — show it verbatim so reviewers can see the contract in action.
            print("\n-- child stdout (RoundMetrics JSON) --")
            print(proc.stdout.rstrip())
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            print(f"run-round exited {proc.returncode}", file=sys.stderr)
            return proc.returncode

        _hr("4. metrics.json (branch + commit_sha populated by Runner)")
        metrics_path = round_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key in (
            "status",
            "branch",
            "commit_sha",
            "round_attempt_id",
            "fork_from",
            "delta_ler",
        ):
            print(f"  {key:17s}= {metrics.get(key)!r}")

        assert metrics["status"] == "ok", f"status={metrics['status']}"
        assert metrics["branch"] == plan["branch"]
        assert metrics["commit_sha"] is not None
        assert metrics["round_attempt_id"] == round_attempt_id

        _hr("5. round_1_pointer.json round-trips via git show <branch>:...")
        pointer_blob = _git(
            REPO_ROOT,
            "show",
            f"{plan['branch']}:round_{round_idx}/round_{round_idx}_pointer.json",
        )
        pointer = json.loads(pointer_blob)
        for key in ("round_attempt_id", "round_idx", "branch", "written_at_utc"):
            print(f"  {key:18s}= {pointer[key]!r}")
        assert pointer["round_attempt_id"] == round_attempt_id
        assert pointer["branch"] == plan["branch"]

        _hr("6. git log on the experiment branch")
        print(_git(REPO_ROOT, "log", plan["branch"], "--oneline", "-n", "3"))

        if args.keep_worktree:
            _hr("7. cleanup_round_worktree SKIPPED (--keep-worktree)")
        else:
            _hr("7. cleanup_round_worktree (removes checkout, keeps branch)")
            cleanup_round_worktree(REPO_ROOT, plan["worktree_dir"])
            print("checkout removed.")
            branch_exists = (
                _git(REPO_ROOT, "branch", "--list", plan["branch"]) != ""
            )
            worktree_list = _git(REPO_ROOT, "worktree", "list")
            worktree_still_present = plan["worktree_dir"].replace("\\", "/") in (
                worktree_list.replace("\\", "/")
            )
            print(f"branch retained    = {branch_exists}")
            print(f"worktree present   = {worktree_still_present}")
            assert branch_exists
            assert not worktree_still_present

        _hr("8. reconcile_at_startup (demonstration pointer)")
        print(
            "The same pointer blob we just read would let\n"
            "  autoqec.orchestration.reconcile.reconcile_at_startup(repo, run_id)\n"
            "auto-heal this branch into an 'orphaned_branch' synthetic row if the\n"
            "runner had crashed before history.jsonl was appended. See\n"
            "tests/test_reconcile.py::test_orphaned_branch_with_pointer_autoheals\n"
            "for the full auto-heal contract."
        )

        _hr("Demo complete")
        if not args.keep_branch:
            print(
                "To drop the experiment branch after the presentation, run:\n"
                f"  git branch -D {plan['branch']}\n"
                "Or use demos/demo-3-worktree-provenance/cleanup.sh"
            )
        return 0

    except Exception:
        cleanup_round_worktree(REPO_ROOT, plan["worktree_dir"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
