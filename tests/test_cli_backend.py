from unittest.mock import patch

import pytest

from autoqec.agents.cli_backend import (
    InvalidSubagentResponseError,
    _build_cli_argv,
    _parse_fenced_json,
    invoke_subagent,
)


def test_build_argv_codex_cli(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "gpt-5.4")
    assert _build_cli_argv("ideator") == ["codex", "exec", "--model", "gpt-5.4", "-"]


def test_build_argv_claude_cli(monkeypatch):
    monkeypatch.setenv("AUTOQEC_CODER_BACKEND", "claude-cli")
    monkeypatch.setenv("AUTOQEC_CODER_MODEL", "claude-haiku-4-5")
    assert _build_cli_argv("coder") == ["claude", "-p", "--model", "claude-haiku-4-5"]


def test_parse_fenced_json_strict():
    text = 'hello\n\n```json\n{"k": "v"}\n```\ntrailing'
    assert _parse_fenced_json(text) == {"k": "v"}


def test_parse_fenced_json_no_block_raises():
    with pytest.raises(InvalidSubagentResponseError, match="no fenced json"):
        _parse_fenced_json("plain prose with no json")


def test_parse_fenced_json_malformed_raises():
    with pytest.raises(InvalidSubagentResponseError, match="malformed json"):
        _parse_fenced_json("```json\n{not valid\n```")


def test_invoke_subagent_returns_parsed(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "gpt-5.4")

    class FakeCompleted:
        stdout = '```json\n{"hypothesis": "try GNN", "fork_from": "baseline"}\n```'
        stderr = ""
        returncode = 0

    with patch("subprocess.run", return_value=FakeCompleted()):
        out = invoke_subagent("ideator", "prompt here")
    assert out["hypothesis"] == "try GNN"


def test_invoke_subagent_propagates_nonzero(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "codex-cli")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "gpt-5.4")

    class FakeCompleted:
        stdout = ""
        stderr = "model refused"
        returncode = 2

    with patch("subprocess.run", return_value=FakeCompleted()):
        with pytest.raises(InvalidSubagentResponseError, match="exit 2"):
            invoke_subagent("ideator", "prompt")
