"""End-to-end demo of the §15 worktree + compose-merge experiment model.

Walks through four scenarios in one run, with worktree contents, git
log --graph, and compose_conflict made visible at every step:

  1. Round 1 — Idea A forks from baseline, trains, commits pointer,
     worktree is inspected before and after run-round, then released.
  2. Round 2 — Idea B forks from baseline independently.
  3. Round 3 — create_compose_worktree(parents=[A, B]) merges the two
     ideas, the merge commit is shown in git log --graph, and the
     compose round trains on top of the merged checkout.
  4. Compose-conflict probe — two deliberately-conflicting branches
     are merged; the demo catches status="compose_conflict" and the
     reported conflicting_files.

Finally the full fork graph across exp/<run_id>/* is printed so you
can see all branches as a single tree.

No live LLM: the predecoder config comes from fixture_config.yaml, so
the same path runs reproducibly offline.
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

# Force UTF-8 on stdout so the §/— characters in the walkthrough
# survive redirection on Windows consoles whose locale is cp936/GBK.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _hr(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _git(cwd: Path | str, *args: str) -> str:
    # encoding="utf-8" + errors="replace" to avoid Windows cp936/GBK decode
    # crashes when a commit message or author name carries non-ASCII bytes.
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args],
        encoding="utf-8",
        errors="replace",
    ).rstrip()


def _git_checked(cwd: Path | str, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )


def _show_worktree_contents(worktree_dir: str, label: str) -> None:
    """Prove the worktree is a real independent checkout: list the top-level
    tree entries, drill into round_*/ subdirs, and tail 3 commits.
    """
    wt = Path(worktree_dir)
    print(f"  [{label}]")
    print(f"  $ ls {wt.name}/")
    for entry in sorted(wt.iterdir()):
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        marker = (
            "    <-- §15 round provenance"
            if entry.is_dir() and entry.name.startswith("round_")
            else ""
        )
        suffix = "/" if entry.is_dir() else ""
        print(f"    {entry.name}{suffix}{marker}")

    for round_dir in sorted(wt.glob("round_*/")):
        print(f"  $ ls {wt.name}/{round_dir.name}/")
        for f in sorted(round_dir.iterdir()):
            print(f"    {f.name}")

    print(f"  $ git -C {wt.name} log --oneline -n 3")
    try:
        for line in _git(wt, "log", "--oneline", "-n", "3").splitlines():
            print(f"    {line}")
    except subprocess.CalledProcessError:
        print("    (no commits)")


def _run_round(
    python_exe: str,
    plan: dict,
    run_id: str,
    round_idx: int,
    round_attempt_id: str,
    fork_from_arg: str | list[str],
    compose_mode: str | None = None,
) -> dict:
    """Invoke ``python -m cli.autoqec run-round`` and return parsed metrics."""
    round_dir = REPO_ROOT / "runs" / run_id / f"round_{round_idx}"
    round_dir.mkdir(parents=True, exist_ok=True)

    fork_from_str = (
        json.dumps(fork_from_arg)
        if isinstance(fork_from_arg, list)
        else fork_from_arg
    )
    cmd = [
        python_exe, "-m", "cli.autoqec", "run-round",
        str(ENV_YAML), str(FIXTURE_CONFIG), str(round_dir),
        "--profile", "dev",
        "--code-cwd", plan["worktree_dir"],
        "--branch", plan["branch"],
        "--fork-from", fork_from_str,
        "--round-attempt-id", round_attempt_id,
    ]
    if compose_mode is not None:
        cmd += ["--compose-mode", compose_mode]

    summary_cmd = (
        f"python -m cli.autoqec run-round ... "
        f"--code-cwd .worktrees/{Path(plan['worktree_dir']).name} "
        f"--branch {plan['branch']} --fork-from {fork_from_str}"
    )
    print(f"  $ {summary_cmd}")
    sys.stdout.flush()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"run-round exited {proc.returncode}")

    metrics = json.loads((round_dir / "metrics.json").read_text(encoding="utf-8"))
    print(
        f"    status={metrics['status']}  "
        f"branch={metrics['branch']}  "
        f"commit={metrics['commit_sha'][:12]}"
    )
    assert metrics["status"] == "ok"
    assert metrics["branch"] == plan["branch"]
    assert metrics["commit_sha"] is not None
    assert metrics["round_attempt_id"] == round_attempt_id
    return metrics


def _setup_conflict_pair(
    repo_root: Path, run_id: str, fork_from: str
) -> tuple[dict, dict]:
    """Create two branches that each commit CONFLICT_MARKER.tmp with a
    different payload, guaranteeing create_compose_worktree's git merge
    will hit conflict and return status=compose_conflict."""
    from autoqec.orchestration.worktree import (
        cleanup_round_worktree,
        create_round_worktree,
    )

    plans: list[dict] = []
    for idx, variant in [(90, "left"), (91, "right")]:
        plan = create_round_worktree(
            repo_root=repo_root,
            run_id=run_id,
            round_idx=idx,
            slug=f"conflict-{variant}",
            fork_from=fork_from,
        )
        wt = Path(plan["worktree_dir"])
        (wt / "CONFLICT_MARKER.tmp").write_text(
            f"variant={variant}\n", encoding="utf-8"
        )
        _git_checked(wt, "add", "CONFLICT_MARKER.tmp")
        _git_checked(
            wt, "commit", "-q", "-m", f"demo: conflict variant {variant}"
        )
        cleanup_round_worktree(repo_root, plan["worktree_dir"])
        plans.append(plan)
    return plans[0], plans[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=time.strftime("demo38-%Y%m%d-%H%M%S"),
        help="Run identifier; drives branch name exp/<run_id>/NN-<slug>",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for cli.autoqec run-round",
    )
    parser.add_argument(
        "--skip-conflict-demo",
        action="store_true",
        help="Skip Step 5 (compose_conflict probe)",
    )
    args = parser.parse_args()

    from autoqec.orchestration.worktree import (
        cleanup_round_worktree,
        create_compose_worktree,
        create_round_worktree,
    )

    run_id = args.run_id
    fork_from = _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    attempt_ids = {i: str(uuid.uuid4()) for i in (1, 2, 3)}
    live_worktrees: list[str] = []

    _hr("Step 0 — Setup")
    print(f"run_id    = {run_id}")
    print(f"fork_from = {fork_from}  (this is 'baseline' in Ideator terminology)")
    print(f"round_attempt_ids = {attempt_ids}")

    try:
        # ── Round 1: Idea A, fork from baseline ────────────────────────
        _hr("Step 1 — Round 1: Idea A (create_round_worktree, fork baseline)")
        plan_a = create_round_worktree(
            repo_root=REPO_ROOT, run_id=run_id, round_idx=1,
            slug="idea-a", fork_from=fork_from,
        )
        live_worktrees.append(plan_a["worktree_dir"])
        print(f"  branch       = {plan_a['branch']}")
        print(f"  worktree_dir = {plan_a['worktree_dir']}")
        print()
        print("  The worktree is a real, separate checkout (pre-run):")
        _show_worktree_contents(plan_a["worktree_dir"], "before run-round")

        print("\n  Now run the actual Runner inside that worktree:")
        _run_round(
            args.python, plan_a, run_id, round_idx=1,
            round_attempt_id=attempt_ids[1], fork_from_arg=fork_from,
        )

        print("\n  Post-run contents — notice the new round_1/ + pointer commit:")
        _show_worktree_contents(plan_a["worktree_dir"], "after run-round")

        cleanup_round_worktree(REPO_ROOT, plan_a["worktree_dir"])
        live_worktrees.remove(plan_a["worktree_dir"])
        branch_row = _git(
            REPO_ROOT, "branch", "--list", plan_a["branch"]
        ).strip()
        print(
            f"\n  cleanup_round_worktree(...) done — checkout gone, "
            f"branch kept: {branch_row!r}"
        )

        # ── Round 2: Idea B, independent fork ──────────────────────────
        _hr("Step 2 — Round 2: Idea B (independent fork from baseline)")
        plan_b = create_round_worktree(
            repo_root=REPO_ROOT, run_id=run_id, round_idx=2,
            slug="idea-b", fork_from=fork_from,
        )
        live_worktrees.append(plan_b["worktree_dir"])
        print(f"  branch = {plan_b['branch']}")
        _run_round(
            args.python, plan_b, run_id, round_idx=2,
            round_attempt_id=attempt_ids[2], fork_from_arg=fork_from,
        )
        cleanup_round_worktree(REPO_ROOT, plan_b["worktree_dir"])
        live_worktrees.remove(plan_b["worktree_dir"])

        _hr("Step 3 — Two independent experiment branches now in git")
        for line in _git(
            REPO_ROOT, "branch", "--list", f"exp/{run_id}/*"
        ).splitlines():
            print(f"  {line.strip()}")

        # ── Round 3: Compose merge ─────────────────────────────────────
        _hr("Step 4 — Round 3: Compose merge of Idea A ⊕ Idea B")
        print("  create_compose_worktree(parents=[A.branch, B.branch])")
        plan_c = create_compose_worktree(
            repo_root=REPO_ROOT, run_id=run_id, round_idx=3,
            slug="compose-ab",
            parents=[plan_a["branch"], plan_b["branch"]],
        )
        if plan_c.get("status") == "compose_conflict":
            raise AssertionError(
                "happy-path compose unexpectedly conflicted: "
                f"{plan_c['conflicting_files']}"
            )
        live_worktrees.append(plan_c["worktree_dir"])
        print(f"  status             = {plan_c['status']}  (merge OK)")
        print(f"  branch             = {plan_c['branch']}")
        print(f"  parents (ordered)  = {plan_c['fork_from_ordered']}")
        print(f"  canonical key      = {plan_c['fork_from_canonical']!r}")
        print()
        print(
            "  git log --graph --oneline (merge commit visible as a diamond):"
        )
        for line in _git(
            plan_c["worktree_dir"], "log", "--graph", "--oneline",
            "--decorate", "-n", "8",
            plan_c["branch"], plan_a["branch"], plan_b["branch"],
        ).splitlines():
            print(f"    {line}")
        print()
        print(
            "  Post-merge worktree contents — both round_1/ and round_2/ "
            "pointer files are present, proving the merge really combined "
            "both parents:"
        )
        _show_worktree_contents(plan_c["worktree_dir"], "after compose merge")

        print("\n  Now train a compose-round on the merged checkout:")
        _run_round(
            args.python, plan_c, run_id, round_idx=3,
            round_attempt_id=attempt_ids[3],
            fork_from_arg=[plan_a["branch"], plan_b["branch"]],
            compose_mode="pure",
        )
        cleanup_round_worktree(REPO_ROOT, plan_c["worktree_dir"])
        live_worktrees.remove(plan_c["worktree_dir"])

        # ── Step 5: deliberate compose_conflict ────────────────────────
        if not args.skip_conflict_demo:
            _hr("Step 5 — Compose-conflict probe (§15.6.3)")
            print(
                "  Build two branches that both write CONFLICT_MARKER.tmp "
                "with different payloads, then attempt to merge them:"
            )
            conflict_a, conflict_b = _setup_conflict_pair(
                REPO_ROOT, run_id, fork_from
            )
            print(f"    {conflict_a['branch']}  <- variant=left")
            print(f"    {conflict_b['branch']}  <- variant=right")
            plan_x = create_compose_worktree(
                repo_root=REPO_ROOT, run_id=run_id, round_idx=99,
                slug="compose-conflict",
                parents=[conflict_a["branch"], conflict_b["branch"]],
            )
            print(f"  status            = {plan_x['status']!r}")
            print(f"  conflicting_files = {plan_x.get('conflicting_files')}")
            print(
                "  — per §15.6.3 the orchestrator writes a row with\n"
                "    status='compose_conflict', branch=None, commit_sha=None\n"
                "    and does NOT retry. The worktree + branch are both "
                "removed by create_compose_worktree on conflict."
            )
            assert plan_x["status"] == "compose_conflict"
            assert plan_x.get("conflicting_files"), "should list conflicts"

        # ── Step 6: Final fork graph ───────────────────────────────────
        _hr(f"Step 6 — Final fork graph across exp/{run_id}/*")
        branches = [
            line.strip()
            for line in _git(
                REPO_ROOT, "branch", "--list", f"exp/{run_id}/*"
            ).splitlines()
            if line.strip()
        ]
        print(f"  {len(branches)} branch(es) alive:")
        for b in branches:
            print(f"    {b}")
        print("\n  git log --graph across all of them:")
        graph = _git(
            REPO_ROOT, "log", "--graph", "--oneline", "--decorate",
            "-n", "30", *branches,
        )
        for line in graph.splitlines():
            print(f"    {line}")

        _hr("Step 7 — reconcile_at_startup contract")
        print(
            "  For each exp/<run_id>/* branch, reconcile would:\n"
            "    git show <branch>:round_<NN>/round_<NN>_pointer.json\n"
            "      -> recoverable round_attempt_id  => auto-heal orphan row\n"
            "      -> missing/malformed             => pause for human review\n"
            "  See tests/test_reconcile.py::test_orphaned_branch_with_pointer_autoheals"
        )

        _hr("Demo complete — cleanup")
        print(
            "  bash demos/demo-3-worktree-provenance/cleanup.sh\n"
            f"  # removes .worktrees/exp-{run_id}-* and deletes exp/{run_id}/* branches"
        )
        return 0

    except Exception:
        for wt in list(live_worktrees):
            cleanup_round_worktree(REPO_ROOT, wt)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
