from unittest.mock import patch
import io
import json

import pytest

from autoqec.agents.cli_backend import (
    InvalidSubagentResponseError,
    _backend_name_and_model,
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


def test_backend_name_and_model_github_models(monkeypatch):
    monkeypatch.setenv("AUTOQEC_ANALYST_BACKEND", "github-models")
    monkeypatch.setenv("AUTOQEC_ANALYST_MODEL", "openai/gpt-4.1-mini")
    assert _backend_name_and_model("analyst") == ("github-models", "openai/gpt-4.1-mini")


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


def test_invoke_subagent_github_models_uses_gh_token_and_parses(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "github-models")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "openai/gpt-4.1-mini")

    captured = {}

    class FakeResponse(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_check_output(argv, text):
        assert argv == ["gh", "auth", "token"]
        assert text is True
        return "gho_test_token\n"

    def fake_urlopen(req, timeout):
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        payload = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"hypothesis": "try github models", "fork_from": "baseline"}\n```'
                    }
                }
            ]
        }
        return FakeResponse(json.dumps(payload).encode("utf-8"))

    with patch("subprocess.check_output", side_effect=fake_check_output), patch(
        "urllib.request.urlopen", side_effect=fake_urlopen
    ):
        out = invoke_subagent("ideator", "prompt here", timeout=12.5)

    assert out["hypothesis"] == "try github models"
    assert captured["timeout"] == 12.5
    assert captured["headers"]["Authorization"] == "Bearer gho_test_token"
    assert captured["body"]["model"] == "openai/gpt-4.1-mini"
    assert captured["body"]["messages"][0]["role"] == "system"
    assert "You are the **Ideator** in AutoQEC" in captured["body"]["messages"][0]["content"]
    assert captured["body"]["messages"][1]["content"] == "prompt here"


def test_invoke_subagent_github_models_requires_auth(monkeypatch):
    monkeypatch.setenv("AUTOQEC_IDEATOR_BACKEND", "github-models")
    monkeypatch.setenv("AUTOQEC_IDEATOR_MODEL", "openai/gpt-4.1-mini")

    with patch("subprocess.check_output", side_effect=FileNotFoundError("gh missing")):
        with pytest.raises(InvalidSubagentResponseError, match="GitHub authentication"):
            invoke_subagent("ideator", "prompt here")
