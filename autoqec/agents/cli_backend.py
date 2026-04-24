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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

Role = Literal["ideator", "coder", "analyst"]


class InvalidSubagentResponseError(RuntimeError):
    pass


_BACKEND_ARGV = {
    "codex-cli": lambda model: ["codex", "exec", "--model", model, "-"],
    "claude-cli": lambda model: ["claude", "-p", "--model", model],
}


def _backend_name_and_model(role: Role) -> tuple[str, str]:
    backend = os.environ.get(f"AUTOQEC_{role.upper()}_BACKEND", "codex-cli")
    model = os.environ.get(
        f"AUTOQEC_{role.upper()}_MODEL",
        "gpt-5.4" if backend == "codex-cli" else "claude-haiku-4-5",
    )
    return backend, model


def _build_cli_argv(role: Role) -> list[str]:
    backend, model = _backend_name_and_model(role)
    if backend not in _BACKEND_ARGV:
        raise InvalidSubagentResponseError(
            f"backend {backend!r} for role {role!r} is not a subprocess CLI backend"
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


def _github_auth_token() -> str:
    for env_key in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(env_key, "").strip()
        if token:
            return token
    try:
        token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise InvalidSubagentResponseError(
            "GitHub authentication is required for the github-models backend; "
            "set GH_TOKEN/GITHUB_TOKEN or run `gh auth login`."
        ) from exc
    if not token:
        raise InvalidSubagentResponseError(
            "GitHub authentication is required for the github-models backend; "
            "received an empty token from `gh auth token`."
        )
    return token


def _role_system_prompt(role: Role) -> str:
    agent_path = (
        Path(__file__).resolve().parents[2]
        / ".claude"
        / "agents"
        / f"autoqec-{role}.md"
    )
    try:
        return agent_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise InvalidSubagentResponseError(
            f"missing agent spec for role {role!r}: {agent_path}"
        ) from exc


def _invoke_github_models(role: Role, model: str, prompt: str, timeout: float) -> dict:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _role_system_prompt(role)},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://models.github.ai/inference/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_github_auth_token()}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise InvalidSubagentResponseError(
            f"github-models HTTP {exc.code}: {error_body[:200]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise InvalidSubagentResponseError(f"github-models request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise InvalidSubagentResponseError(f"github-models request timed out after {timeout}s") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidSubagentResponseError(
            f"github-models response missing assistant content: {body!r}"
        ) from exc
    return _parse_fenced_json(content)


def invoke_subagent(role: Role, prompt: str, timeout: float = 300.0) -> dict:
    backend, model = _backend_name_and_model(role)
    if backend == "github-models":
        return _invoke_github_models(role, model, prompt, timeout)

    if backend not in _BACKEND_ARGV:
        raise InvalidSubagentResponseError(
            f"unknown backend {backend!r} for role {role!r}"
        )

    argv = _BACKEND_ARGV[backend](model)
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        argv,
        input=prompt,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise InvalidSubagentResponseError(
            f"{argv[0]} exit {result.returncode}: {result.stderr[:200]}"
        )
    return _parse_fenced_json(result.stdout)
