"""Subprocess-based runner for worktree-path rounds (§15.7 stub).

Full implementation lands in Task 9. This stub exists so the CLI can import
the symbol and `run-round --help` works without the torch-heavy runner.
"""
from __future__ import annotations

from autoqec.envs.schema import EnvSpec
from autoqec.runner.schema import RoundMetrics, RunnerConfig


def run_round_in_subprocess(
    cfg: RunnerConfig,
    env: EnvSpec,
    round_attempt_id: str | None = None,
) -> RoundMetrics:
    """Launch the Runner in a child process that cwd's into ``cfg.code_cwd``.

    Stub: implemented in Task 9.
    """
    raise NotImplementedError(
        "implemented in Task 9; stub exists for import resolution"
    )
