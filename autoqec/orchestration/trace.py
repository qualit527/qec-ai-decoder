"""Orchestrator-side narrative trace for ``/autoqec-run``.

The Claude Code session JSONL (``~/.claude/projects/.../<uuid>.jsonl``)
already captures every tool call + result the orchestrator made, but it
lives outside ``runs/<run_id>/`` and is keyed on session UUID. This
module renders the chat-level story — subagent prompts / responses,
Runner metrics, Verifier reports, free-form orchestrator notes — into a
stable markdown file next to ``history.jsonl``, so the narrative
survives outside CC and can be diffed, grepped, or shown on a PR.

Append-only. Never truncates. Resumes are marked with a divider line so
multiple sessions against the same ``run_id`` don't silently overwrite
each other.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_FILENAME = "orchestrator_trace.md"


def trace_path(run_dir: Path | str) -> Path:
    """Canonical location of the trace file for ``run_dir``."""
    return Path(run_dir) / _FILENAME


def init_trace(
    run_dir: Path | str,
    *,
    env_yaml: str,
    rounds: int,
    profile: str,
) -> Path:
    """Create or resume the trace file. Idempotent.

    First call writes a YAML-like header with run metadata. Subsequent
    calls on the same ``run_dir`` append a ``_resumed at ..._`` divider
    instead of truncating — the existing header stays intact.
    """
    path = trace_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _append(path, f"\n---\n\n_resumed at {_iso()}_\n")
        return path
    header = (
        f"# Orchestrator trace — {Path(run_dir).name}\n\n"
        f"- env: `{env_yaml}`\n"
        f"- rounds: {rounds}\n"
        f"- profile: {profile}\n"
        f"- started: {_iso()}\n"
    )
    path.write_text(header, encoding="utf-8")
    return path


def append_section(
    run_dir: Path | str,
    round_idx: int | None,
    kind: str,
    body: Any,
    *,
    fence: str = "",
) -> None:
    """Append one labeled, timestamped section to the trace file.

    Parameters
    ----------
    run_dir:
        The run directory. Must have been through :func:`init_trace`.
    round_idx:
        1-based round index. A ``## Round <N>`` header is emitted the
        first time a new round is seen; reused on subsequent appends for
        the same round. Pass ``None`` for run-wide sections (e.g. the
        final "run complete" block).
    kind:
        Short human label — ``"ideator raw"``, ``"runner metrics"``,
        ``"verifier report"``. Used as the ``###`` heading.
    body:
        The section content. ``dict``/``list`` → JSON-fenced;
        ``None`` → ``_(empty)_``; anything else → ``str(body)`` inside a
        fenced block tagged with ``fence``.
    fence:
        Info-string for the code fence when ``body`` is a string
        (e.g. ``"yaml"``, ``"python"``, ``""`` for plain).
    """
    path = trace_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"trace not initialised: {path}")

    if round_idx is not None:
        existing = path.read_text(encoding="utf-8")
        round_header = f"\n## Round {round_idx}\n"
        if round_header not in existing:
            _append(path, round_header)

    ts = _hms()
    if isinstance(body, (dict, list)):
        rendered = (
            "```json\n"
            + json.dumps(body, indent=2, ensure_ascii=False, default=str)
            + "\n```"
        )
    elif body is None:
        rendered = "_(empty)_"
    else:
        rendered = f"```{fence}\n{str(body).rstrip()}\n```"
    _append(path, f"\n### [{ts}] {kind}\n\n{rendered}\n")


def append_note(
    run_dir: Path | str,
    round_idx: int | None,
    text: str,
) -> None:
    """Append a free-form orchestrator note — the ``●`` chat line equivalent."""
    path = trace_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"trace not initialised: {path}")
    if round_idx is not None:
        existing = path.read_text(encoding="utf-8")
        round_header = f"\n## Round {round_idx}\n"
        if round_header not in existing:
            _append(path, round_header)
    _append(path, f"\n_{_hms()}_ {text}\n")


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _hms() -> str:
    return time.strftime("%H:%M:%S")


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)
