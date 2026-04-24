"""Tests for autoqec.orchestration.trace — chat-level narrative log."""
from __future__ import annotations

import pytest

from autoqec.orchestration.trace import (
    append_note,
    append_section,
    init_trace,
    trace_path,
)


def test_init_creates_header(tmp_path):
    path = init_trace(tmp_path, env_yaml="envs/x.yaml", rounds=3, profile="dev")
    assert path == trace_path(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Orchestrator trace — ")
    assert "env: `envs/x.yaml`" in text
    assert "rounds: 3" in text
    assert "profile: dev" in text
    assert "started:" in text


def test_init_is_idempotent_and_marks_resume(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert text.count("# Orchestrator trace") == 1
    assert "_resumed at" in text


def test_append_section_with_dict_renders_json(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_section(
        tmp_path,
        1,
        "ideator raw",
        {"hypothesis": "test", "fork_from": "baseline"},
    )
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert "## Round 1" in text
    assert "ideator raw" in text
    assert '"hypothesis": "test"' in text
    assert "```json" in text


def test_append_section_with_str_uses_fence(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_section(tmp_path, 1, "config", "type: gnn\nhidden: 32", fence="yaml")
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert "```yaml" in text
    assert "type: gnn" in text
    assert "hidden: 32" in text


def test_same_round_reuses_header(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_section(tmp_path, 1, "a", "x")
    append_section(tmp_path, 1, "b", "y")
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert text.count("## Round 1") == 1


def test_new_round_adds_header(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=2, profile="dev")
    append_section(tmp_path, 1, "a", "x")
    append_section(tmp_path, 2, "b", "y")
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert text.count("## Round 1") == 1
    assert text.count("## Round 2") == 1


def test_append_section_before_init_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        append_section(tmp_path, 1, "x", "y")


def test_append_note_writes_line_with_timestamp(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_note(tmp_path, 1, "dispatching autoqec-ideator")
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert "## Round 1" in text
    assert "dispatching autoqec-ideator" in text


def test_append_note_without_round(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_note(tmp_path, None, "run complete")
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert "run complete" in text
    # No round header should be added for a None round_idx
    assert "## Round" not in text


def test_append_none_body_renders_empty_marker(tmp_path):
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_section(tmp_path, 1, "empty", None)
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert "_(empty)_" in text


def test_non_ascii_survives_roundtrip(tmp_path):
    """Chinese hypotheses / Δ / non-ASCII must not get mojibaked."""
    init_trace(tmp_path, env_yaml="x", rounds=1, profile="dev")
    append_section(tmp_path, 1, "hypothesis", {"summary": "Δ_LER 改善 3e-3"})
    text = trace_path(tmp_path).read_text(encoding="utf-8")
    assert "Δ_LER 改善 3e-3" in text
