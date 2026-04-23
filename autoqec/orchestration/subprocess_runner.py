"""Shell-out Runner dispatch for §15.8 worktree mode.

Python's import cache can't hot-reload edited ``modules/*.py`` files, so we
launch a fresh interpreter with ``cwd=cfg.code_cwd`` and ``PYTHONPATH``
pinned to the worktree. The child invokes ``python -m cli.autoqec run-round``
and prints a ``metrics.json``-shaped JSON payload; we parse it here.

After a successful non-``compose_conflict`` round the parent also writes
and commits ``round_<N>/round_<N>_pointer.json`` on the branch — this is
the producer side of the §15.10 reconcile contract. Without it reconcile
cannot auto-heal an orphaned branch after a crash and always falls through
to ``pause``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from autoqec.envs.schema import EnvSpec
from autoqec.runner.schema import RoundMetrics, RunnerConfig

log = logging.getLogger(__name__)


class RunnerSubprocessError(RuntimeError):
    """Raised when the child process returns a non-zero exit code."""


_ROUND_DIR_RE = re.compile(r"^round_(?P<idx>\d+)$")


def _extract_round_idx(round_dir: str) -> int | None:
    """Pull ``N`` from a ``round_<N>`` directory name. Return None on mismatch."""
    match = _ROUND_DIR_RE.match(Path(round_dir).name)
    return int(match.group("idx")) if match else None


def _write_and_commit_pointer(
    code_cwd: str,
    round_idx: int,
    round_attempt_id: str | None,
    branch: str,
) -> str | None:
    """Write ``round_<N>/round_<N>_pointer.json`` into the worktree, commit it,
    and return the new HEAD sha.

    This is the §15.10 auto-heal producer. The pointer lives on the branch so
    ``git show <branch>:round_<N>/round_<N>_pointer.json`` resolves it even
    after a crash / kill that wiped the in-memory history append.
    Returns None on any git failure so the caller can still produce metrics
    instead of masking a training-side success with a pointer-side error.
    """
    pointer_dir = Path(code_cwd) / f"round_{round_idx}"
    pointer_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = pointer_dir / f"round_{round_idx}_pointer.json"
    pointer_path.write_text(
        json.dumps(
            {
                "round_attempt_id": round_attempt_id,
                "round_idx": round_idx,
                "branch": branch,
                "written_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    rel_pointer = f"round_{round_idx}/round_{round_idx}_pointer.json"
    try:
        subprocess.run(
            ["git", "-C", code_cwd, "add", rel_pointer],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                code_cwd,
                "commit",
                "-q",
                "-m",
                f"round {round_idx}: pointer for attempt {round_attempt_id or 'unknown'}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        head = subprocess.run(
            ["git", "-C", code_cwd, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return head.stdout.strip()
    except subprocess.CalledProcessError as exc:
        log.warning(
            "pointer commit failed for round %s (branch=%s): %s",
            round_idx,
            branch,
            (exc.stderr or "").strip(),
        )
        return None


def run_round_in_subprocess(
    cfg: RunnerConfig,
    env: EnvSpec,
    round_attempt_id: str | None = None,
    timeout_s: int = 3000,
) -> RoundMetrics:
    """Run one round in a subprocess with cwd=cfg.code_cwd and PYTHONPATH pinned.

    Returns a :class:`RoundMetrics` parsed from the child's stdout. Raises
    :class:`RunnerSubprocessError` if the child exits non-zero and
    :class:`ValueError` if ``cfg.code_cwd`` is unset.
    """
    if cfg.code_cwd is None:
        raise ValueError("run_round_in_subprocess requires cfg.code_cwd")

    # Persist predecoder config to a temp YAML the subprocess can read.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg.predecoder_config, f)
        config_path = f.name

    # Env YAML on disk: the subprocess re-loads it, so the in-memory EnvSpec
    # is not enough. Prefer the builtin path if the env name matches one.
    env_file = Path("autoqec/envs/builtin") / f"{env.name}.yaml"
    if not env_file.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(env.model_dump(), f)
            env_file = Path(f.name)

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = cfg.code_cwd + os.pathsep + child_env.get("PYTHONPATH", "")

    argv: list[str] = [
        sys.executable,
        "-m",
        "cli.autoqec",
        "run-round",
        str(env_file),
        str(config_path),
        cfg.round_dir,
        "--profile",
        cfg.training_profile,
        "--code-cwd",
        cfg.code_cwd,
        "--branch",
        cfg.branch or "",
    ]
    if cfg.fork_from is not None:
        fork_arg = (
            json.dumps(cfg.fork_from)
            if isinstance(cfg.fork_from, list)
            else cfg.fork_from
        )
        argv += ["--fork-from", fork_arg]
    if cfg.compose_mode is not None:
        argv += ["--compose-mode", cfg.compose_mode]
    if round_attempt_id is not None:
        argv += ["--round-attempt-id", round_attempt_id]
    # Recursion guard: the child must NOT re-dispatch through subprocess_runner.
    # See cli/autoqec.py:run_round_cmd — when this flag is set, the child runs
    # the in-process Runner even though --code-cwd is present.
    argv += ["--_internal-execute-locally"]

    proc = subprocess.run(
        argv,
        cwd=cfg.code_cwd,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RunnerSubprocessError(
            f"subprocess runner failed: rc={proc.returncode}\n"
            f"stdout={proc.stdout}\nstderr={proc.stderr}"
        )

    metrics_data = json.loads(proc.stdout)
    metrics_data.setdefault("round_attempt_id", round_attempt_id)

    # compose_conflict rows must have branch=None / commit_sha=None per §15.6.3.
    # For every other status attach the branch so downstream joins work, and
    # write + commit the §15.10 pointer so reconcile can auto-heal later.
    if metrics_data.get("status") != "compose_conflict":
        metrics_data.setdefault("branch", cfg.branch)
        if cfg.branch is not None:
            round_idx = _extract_round_idx(cfg.round_dir)
            if round_idx is not None:
                commit_sha = _write_and_commit_pointer(
                    cfg.code_cwd,
                    round_idx,
                    round_attempt_id or metrics_data.get("round_attempt_id"),
                    cfg.branch,
                )
                if commit_sha is not None:
                    # Pointer commit is the authoritative round provenance.
                    metrics_data["commit_sha"] = commit_sha
            else:
                log.warning(
                    "cfg.round_dir=%r does not match round_<N>; skipping pointer",
                    cfg.round_dir,
                )

    return RoundMetrics(**metrics_data)
