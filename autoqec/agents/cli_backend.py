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
from pathlib import Path
from typing import Any
from typing import Literal

Role = Literal["ideator", "coder", "analyst"]
_VALID_BACKENDS = frozenset({"codex-cli", "claude-cli"})


class InvalidSubagentResponseError(RuntimeError):
    pass


def _backend_and_model(role: Role) -> tuple[str, str]:
    backend = os.environ.get(f"AUTOQEC_{role.upper()}_BACKEND", "codex-cli")
    model = os.environ.get(
        f"AUTOQEC_{role.upper()}_MODEL",
        "gpt-5.4" if backend == "codex-cli" else "claude-haiku-4-5",
    )
    if backend not in _VALID_BACKENDS:
        raise InvalidSubagentResponseError(
            f"unknown backend {backend!r} for role {role!r}"
        )
    return backend, model


def _build_cli_argv(role: Role, *, structured_output: bool = False) -> list[str]:
    backend, model = _backend_and_model(role)
    if backend == "codex-cli":
        argv = ["codex", "exec", "--model", model]
        if structured_output:
            argv.append("--json")
        argv.append("-")
        return argv
    argv = ["claude", "-p", "--model", model]
    if structured_output:
        argv.extend(["--output-format", "json"])
    return argv


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


def _parse_claude_json_result(stdout: str) -> tuple[dict, dict[str, Any]]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise InvalidSubagentResponseError(f"malformed claude json output: {exc}") from exc
    if not isinstance(payload, dict) or "result" not in payload:
        raise InvalidSubagentResponseError(
            "claude json output missing result field"
        )
    return _parse_fenced_json(str(payload["result"])), {
        "usage": payload.get("usage"),
        "duration_ms": payload.get("duration_ms"),
        "total_cost_usd": payload.get("total_cost_usd"),
        "model_usage": payload.get("modelUsage"),
        "session_id": payload.get("session_id"),
        "uuid": payload.get("uuid"),
    }


def _resolve_repo_root(cwd: str | None) -> Path:
    return Path(cwd).resolve() if cwd is not None else Path(__file__).resolve().parents[2]


def _prepend_agent_spec(role: Role, prompt: str, cwd: str | None) -> str:
    spec_path = _resolve_repo_root(cwd) / ".claude" / "agents" / f"autoqec-{role}.md"
    if not spec_path.exists():
        return prompt
    spec_text = spec_path.read_text(encoding="utf-8").strip()
    return (
        f"# AUTOQEC {role.upper()} SPEC\n\n"
        f"{spec_text}\n\n"
        "# TASK INPUT\n\n"
        f"{prompt}"
    )


def invoke_subagent_with_metadata(
    role: Role,
    prompt: str,
    timeout: float = 300.0,
    *,
    cwd: str | None = None,
) -> tuple[dict, dict[str, Any]]:
    backend, model = _backend_and_model(role)
    argv = _build_cli_argv(role, structured_output=(backend == "claude-cli"))
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    effective_prompt = _prepend_agent_spec(role, prompt, cwd)
    result = subprocess.run(
        argv,
        input=effective_prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        timeout=timeout,
        cwd=cwd,
    )
    if result.returncode != 0:
        raise InvalidSubagentResponseError(
            f"{argv[0]} exit {result.returncode}: {result.stderr[:200]}"
        )
    if backend == "claude-cli":
        parsed, meta = _parse_claude_json_result(result.stdout)
    else:
        parsed, meta = _parse_fenced_json(result.stdout), {}
    meta.setdefault("usage", None)
    meta.setdefault("duration_ms", None)
    meta.setdefault("total_cost_usd", None)
    meta["backend"] = backend
    meta["model"] = model
    return parsed, meta


def invoke_subagent(role: Role, prompt: str, timeout: float = 300.0) -> dict:
    parsed, _ = invoke_subagent_with_metadata(role, prompt, timeout)
    return parsed
