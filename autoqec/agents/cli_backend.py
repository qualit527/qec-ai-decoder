"""Background subagent dispatch — subprocess adapter for codex-cli / claude-cli.

Mirrors the inline Agent-tool contract used by `/autoqec-run` SKILL: one
fenced ```json block per response. Env-var matrix per role:
  AUTOQEC_{role}_BACKEND ∈ {"codex-cli", "claude-cli"}
  AUTOQEC_{role}_MODEL   = backend-specific model id

Used by `autoqec.orchestration.llm_loop`; the SKILL path stays untouched.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Literal

Role = Literal["ideator", "coder", "analyst"]


class InvalidSubagentResponseError(RuntimeError):
    pass


_BACKEND_ARGV = {
    "codex-cli": lambda model: ["codex", "exec", "--model", model, "-"],
    "claude-cli": lambda model: ["claude", "-p", "--model", model],
}


def _build_cli_argv(role: Role) -> list[str]:
    backend = os.environ.get(f"AUTOQEC_{role.upper()}_BACKEND", "codex-cli")
    model = os.environ.get(
        f"AUTOQEC_{role.upper()}_MODEL",
        "gpt-5.4" if backend == "codex-cli" else "claude-haiku-4-5",
    )
    if backend not in _BACKEND_ARGV:
        raise InvalidSubagentResponseError(
            f"unknown backend {backend!r} for role {role!r}"
        )
    return _BACKEND_ARGV[backend](model)


_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)\n```", re.DOTALL)


def _parse_fenced_json(stdout: str) -> dict:
    m = _FENCE.search(stdout)
    if m is None:
        raise InvalidSubagentResponseError(
            f"no fenced json block in response (got {stdout[:120]!r})"
        )
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise InvalidSubagentResponseError(f"malformed json: {exc}") from exc


def invoke_subagent(role: Role, prompt: str, timeout: float = 300.0) -> dict:
    argv = _build_cli_argv(role)
    result = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise InvalidSubagentResponseError(
            f"{argv[0]} exit {result.returncode}: {result.stderr[:200]}"
        )
    return _parse_fenced_json(result.stdout)
