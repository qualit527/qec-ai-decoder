"""Subagent dispatch adapter.

Two invocation modes share this prompt/response contract:

- **Inline mode** — the orchestrator (running inside Claude Code) calls the
  `Agent` tool with the `autoqec-{role}` subagent, passing the prompt we
  produce here and parsing the fenced JSON block back.
- **Background mode** — `cli/autoqec.py run` (or a future router under
  `autoqec/llm/`) shells out to `claude -p` / `codex exec` with the same
  prompt and reads the same JSON.

Keeping both modes behind `build_prompt` / `parse_response` means we only
have one place to tweak the wire format.
"""
from __future__ import annotations

import json
import re
from typing import Literal

Role = Literal["ideator", "coder", "analyst"]


def build_prompt(role: Role, context: dict) -> str:
    """Assemble the subagent prompt.

    The header is deliberately plain so inline (`Agent` tool) and background
    (`claude -p`) invocations produce identical inputs.
    """
    header = f"# {role.upper()} CONTEXT\n\n"
    body = json.dumps(context, indent=2, default=str)
    footer = (
        "\n\nRespond with exactly one fenced ```json block matching your "
        f"`autoqec-{role}` agent spec. No prose outside the block."
    )
    return header + body + footer


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def parse_response(role: Role, text: str) -> dict:
    """Extract the first fenced ```json block from the subagent's response."""
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        snippet = text[:500].replace("\n", " ⏎ ")
        raise ValueError(f"No JSON block in {role} response: {snippet!r}")
    return json.loads(match.group(1))
