"""Shell-out Runner dispatch for §15.8 worktree mode.

Python's import cache can't hot-reload edited ``modules/*.py`` files, so we
launch a fresh interpreter with ``cwd=cfg.code_cwd`` and ``PYTHONPATH``
pinned to the worktree. The child invokes ``python -m cli.autoqec run-round``
and prints a ``metrics.json``-shaped JSON payload; we parse it here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from autoqec.envs.schema import EnvSpec
from autoqec.runner.schema import RoundMetrics, RunnerConfig


class RunnerSubprocessError(RuntimeError):
    """Raised when the child process returns a non-zero exit code."""


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
    metrics_data.setdefault("branch", cfg.branch)
    return RoundMetrics(**metrics_data)
