"""Shell-out Runner dispatch for §15.8 worktree mode.

Python's import cache can't hot-reload edited ``modules/*.py`` files, so we
launch a fresh interpreter with ``cwd=cfg.code_cwd`` and ``PYTHONPATH``
pinned to the worktree. The child invokes ``python -m cli.autoqec run-round``
and prints a ``metrics.json``-shaped JSON payload; we parse it here.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from autoqec.envs.schema import EnvSpec
from autoqec.runner.schema import RoundMetrics, RunnerConfig


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

    # Persist predecoder config to a temp YAML the subprocess can read.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg.predecoder_config, f)
        config_path = Path(f.name).resolve()

    # Env YAML on disk: the subprocess re-loads it, so the in-memory EnvSpec
    # is not enough. Prefer the builtin path if the env name matches one.
    env_file = (Path("autoqec/envs/builtin") / f"{env.name}.yaml").resolve()
    if not env_file.exists():
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(env.model_dump(), f)
            env_file = Path(f.name).resolve()

    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = code_cwd + os.pathsep + child_env.get("PYTHONPATH", "")
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

    # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
    # Static child command only; all dynamic values go through validated env vars.
    proc = subprocess.run(
        ["python", "-m", "cli.autoqec", "run-round-internal"],
        executable=str(Path(sys.executable).resolve()),
        cwd=code_cwd,
        env=child_env,
        shell=False,
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
    metrics_data.setdefault("branch", cfg.branch)
    return RoundMetrics(**metrics_data)
