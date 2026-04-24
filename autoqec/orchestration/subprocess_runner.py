"""Shell-out Runner dispatch for §15.8 worktree mode.

Python's import cache can't hot-reload edited ``modules/*.py`` files, so we
launch a fresh interpreter with ``cwd=cfg.code_cwd`` and ``PYTHONPATH``
pinned to the worktree. The child invokes ``python -m cli.autoqec
run-round-internal`` (static argv) and reads its payload from
``AUTOQEC_CHILD_*`` env vars so no dynamic user value ever lands on the
child's argv.

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
import shutil
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

from autoqec.envs.schema import EnvSpec
from autoqec.runner.artifact_manifest import write_artifact_manifest
from autoqec.runner.schema import RoundMetrics, RunnerConfig

log = logging.getLogger(__name__)


class RunnerSubprocessError(RuntimeError):
    """Raised when the child process returns a non-zero exit code."""


AUTOQEC_CHILD_ENV_YAML = "AUTOQEC_CHILD_ENV_YAML"
AUTOQEC_CHILD_CONFIG_YAML = "AUTOQEC_CHILD_CONFIG_YAML"
AUTOQEC_CHILD_ROUND_DIR = "AUTOQEC_CHILD_ROUND_DIR"
AUTOQEC_CHILD_PROFILE = "AUTOQEC_CHILD_PROFILE"
AUTOQEC_CHILD_CODE_CWD = "AUTOQEC_CHILD_CODE_CWD"
AUTOQEC_CHILD_BRANCH = "AUTOQEC_CHILD_BRANCH"
AUTOQEC_CHILD_FORK_FROM = "AUTOQEC_CHILD_FORK_FROM"
AUTOQEC_CHILD_COMPOSE_MODE = "AUTOQEC_CHILD_COMPOSE_MODE"
AUTOQEC_CHILD_ROUND_ATTEMPT_ID = "AUTOQEC_CHILD_ROUND_ATTEMPT_ID"

_SAFE_GIT_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,254}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_ROUND_DIR_RE = re.compile(r"^round_(?P<idx>\d+)$")


def _resolve_existing_dir(path_str: str, *, field: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise ValueError(f"{field} must point to an existing directory: {path_str!r}")
    return str(path)


def _resolve_path_arg(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def _validate_git_ref(value: str, *, field: str) -> str:
    if not _SAFE_GIT_REF_RE.fullmatch(value):
        raise ValueError(f"{field} contains unsafe characters: {value!r}")
    if value.endswith("/") or "//" in value or ".." in value or "@{" in value:
        raise ValueError(f"{field} is not a safe git ref: {value!r}")
    return value


def _validate_optional_token(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if not _SAFE_TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field} contains unsafe characters: {value!r}")
    return value


def _extract_round_idx(round_dir: str) -> int | None:
    """Pull ``N`` from a ``round_<N>`` directory name. Return None on mismatch."""
    match = _ROUND_DIR_RE.match(Path(round_dir).name)
    return int(match.group("idx")) if match else None


def _write_round_metrics(round_dir: str, metrics: RoundMetrics) -> RoundMetrics:
    metrics_path = Path(round_dir).resolve() / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return metrics


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
                f"chore(pointer): record round {round_idx} pointer for attempt {round_attempt_id or 'unknown'}",
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
    code_cwd = _resolve_existing_dir(cfg.code_cwd, field="code_cwd")
    branch = _validate_git_ref(cfg.branch or "", field="branch")
    round_dir = _resolve_path_arg(cfg.round_dir)
    safe_round_attempt_id = _validate_optional_token(
        round_attempt_id, field="round_attempt_id"
    )
    safe_fork_from = cfg.fork_from
    if isinstance(safe_fork_from, list):
        safe_fork_from = [
            _validate_git_ref(parent, field="fork_from") for parent in safe_fork_from
        ]
    elif isinstance(safe_fork_from, str):
        safe_fork_from = _validate_git_ref(safe_fork_from, field="fork_from")

    cleanup_files: list[Path] = []
    cleanup_dirs: list[Path] = []

    # Persist predecoder config to a temp YAML the subprocess can read.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg.predecoder_config, f)
        config_path = Path(f.name).resolve()
        cleanup_files.append(config_path)

    # Env YAML on disk: the subprocess re-loads it, so the in-memory EnvSpec
    # is not enough. Prefer the builtin path if the env name matches one.
    env_file = (Path("autoqec/envs/builtin") / f"{env.name}.yaml").resolve()
    if not env_file.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(env.model_dump(), f)
            env_file = Path(f.name).resolve()
            cleanup_files.append(env_file)

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = code_cwd + os.pathsep + child_env.get("PYTHONPATH", "")
    if "MPLCONFIGDIR" not in child_env:
        mpl_config_dir = Path(tempfile.mkdtemp(prefix="autoqec-mpl-")).resolve()
        cleanup_dirs.append(mpl_config_dir)
        child_env["MPLCONFIGDIR"] = str(mpl_config_dir)
    child_env[AUTOQEC_CHILD_ENV_YAML] = str(env_file)
    child_env[AUTOQEC_CHILD_CONFIG_YAML] = str(config_path)
    child_env[AUTOQEC_CHILD_ROUND_DIR] = round_dir
    child_env[AUTOQEC_CHILD_PROFILE] = cfg.training_profile
    child_env[AUTOQEC_CHILD_CODE_CWD] = code_cwd
    child_env[AUTOQEC_CHILD_BRANCH] = branch

    if safe_fork_from is not None:
        child_env[AUTOQEC_CHILD_FORK_FROM] = (
            json.dumps(safe_fork_from)
            if isinstance(safe_fork_from, list)
            else safe_fork_from
        )
    if cfg.compose_mode is not None:
        child_env[AUTOQEC_CHILD_COMPOSE_MODE] = cfg.compose_mode
    if safe_round_attempt_id is not None:
        child_env[AUTOQEC_CHILD_ROUND_ATTEMPT_ID] = safe_round_attempt_id

    child_argv = [sys.executable, "-m", "cli.autoqec", "run-round-internal"]
    try:
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        # Static child command only; all dynamic values go through validated env vars.
        proc = subprocess.run(
            child_argv,
            cwd=code_cwd,
            env=child_env,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            Path(round_dir).mkdir(parents=True, exist_ok=True)
            try:
                write_artifact_manifest(
                    round_dir=Path(round_dir),
                    config=cfg.model_copy(
                        update={
                            "env_yaml_path": str(env_file),
                            "invocation_argv": child_argv,
                        }
                    ),
                    checkpoint_path=Path(round_dir) / "checkpoint.pt",
                    metrics_path=Path(round_dir) / "metrics.json",
                    train_log_path=Path(round_dir) / "train.log",
                )
            except Exception as exc:  # noqa: BLE001 - manifest failures must not mask a round.
                round_path = Path(round_dir)
                round_path.mkdir(parents=True, exist_ok=True)
                (round_path / "manifest_error.txt").write_text(
                    f"{type(exc).__name__}: {exc}",
                    encoding="utf-8",
                )
    finally:
        for path in cleanup_files:
            path.unlink(missing_ok=True)
        for path in cleanup_dirs:
            shutil.rmtree(path, ignore_errors=True)
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
                    code_cwd,
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

    final_metrics = RoundMetrics(**metrics_data)
    return _write_round_metrics(round_dir, final_metrics)
