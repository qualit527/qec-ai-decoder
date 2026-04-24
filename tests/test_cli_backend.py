from unittest.mock import patch

import pytest

from autoqec.agents.cli_backend import (
    InvalidSubagentResponseError,
    _build_cli_argv,
    _parse_fenced_json,
    invoke_subagent,
    invoke_subagent_with_metadata,
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
    captured = {}

    class FakeCompleted:
        stdout = '```json\n{"hypothesis": "try GNN", "fork_from": "baseline"}\n```'
        stderr = ""
        returncode = 0

    def fake_run(*args, **kwargs):
        captured["kwargs"] = kwargs
        return FakeCompleted()

    with patch("subprocess.run", side_effect=fake_run):
        out = invoke_subagent("ideator", "prompt here")
    assert out["hypothesis"] == "try GNN"
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"
    assert captured["kwargs"]["env"]["PYTHONIOENCODING"] == "utf-8"
    assert captured["kwargs"]["env"]["PYTHONUTF8"] == "1"


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


def test_invoke_subagent_with_metadata_parses_claude_json_result(monkeypatch):
    monkeypatch.setenv("AUTOQEC_ANALYST_BACKEND", "claude-cli")
    monkeypatch.setenv("AUTOQEC_ANALYST_MODEL", "claude-haiku-4-5")

    class FakeCompleted:
        stdout = (
            '{"result":"```json\\n{\\"summary_1line\\": \\"ok\\", \\"verdict\\": \\"candidate\\", '
            '\\"next_hypothesis_seed\\": \\"try bigger\\"}\\n```",'
            '"usage":{"input_tokens":123,"output_tokens":45},'
            '"duration_ms":789,"total_cost_usd":0.00123}'
        )
        stderr = ""
        returncode = 0

    with patch("subprocess.run", return_value=FakeCompleted()):
        payload, meta = invoke_subagent_with_metadata("analyst", "prompt")

    assert payload["verdict"] == "candidate"
    assert meta["usage"]["input_tokens"] == 123
    assert meta["usage"]["output_tokens"] == 45
    assert meta["duration_ms"] == 789
    assert meta["total_cost_usd"] == 0.00123
