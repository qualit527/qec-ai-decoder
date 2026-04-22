"""Unit tests for the round recorder (B2 + §15.2 non-dominated Pareto).

The recorder now takes a single `round_metrics` dict (flattened row shape
matching `RoundMetrics` plus the superset fields from §15.7) and a
separate `verify_verdict` gate — "VERIFIED" is the only value that admits
the row into the Pareto archive.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round


def _admit(mem: RunMemory, **overrides) -> None:
    """Helper: simulate a VERIFIED candidate admission."""
    base = {
        "status": "ok",
        "hypothesis": "h",
        "verdict": "candidate",
        "delta_ler": 1e-4,
        "flops_per_syndrome": 100_000,
        "n_params": 50_000,
        "checkpoint_path": "runs/foo/round_1/checkpoint.pt",
        "round": 1,
        "branch": None,
        "commit_sha": None,
        "round_attempt_id": None,
    }
    base.update(overrides)
    record_round(mem, round_metrics=base, verify_verdict="VERIFIED")


# ─── §15.2 non-dominated Pareto tests ────────────────────────────────────


def test_pareto_keeps_dominated_points_no_more(tmp_path: Path) -> None:
    # Three points: a and b are non-dominated, c is dominated by a.
    mem = RunMemory(tmp_path)
    _admit(mem, round=1, delta_ler=4e-4, flops_per_syndrome=200_000, n_params=40_000,
           branch="exp/t/01-a", commit_sha="a1", round_attempt_id="u1")
    _admit(mem, round=2, delta_ler=2e-4, flops_per_syndrome=50_000, n_params=20_000,
           branch="exp/t/02-b", commit_sha="b1", round_attempt_id="u2")
    _admit(mem, round=3, delta_ler=1e-4, flops_per_syndrome=300_000, n_params=60_000,
           branch="exp/t/03-c", commit_sha="c1", round_attempt_id="u3")
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    branches = {row["branch"] for row in pareto}
    assert branches == {"exp/t/01-a", "exp/t/02-b"}  # c is dominated


def test_pareto_has_no_size_cap(tmp_path: Path) -> None:
    # Admit 7 distinct non-dominated points (vary the axes to avoid dominance).
    mem = RunMemory(tmp_path)
    for i in range(7):
        _admit(
            mem,
            round=i + 1,
            delta_ler=1e-4 + i * 1e-5,           # higher delta = better
            flops_per_syndrome=400_000 - i * 50_000,  # lower flops = better
            n_params=10_000 + i * 10_000,        # lower params = better (tradeoff)
            branch=f"exp/t/{i+1:02d}-x",
            commit_sha=f"sha{i}",
            round_attempt_id=f"u{i}",
        )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert len(pareto) == 7  # no truncation to 5


def test_pareto_preview_is_top_5_by_delta(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    for i in range(7):
        _admit(
            mem,
            round=i + 1,
            delta_ler=1e-4 + i * 1e-5,
            flops_per_syndrome=400_000 - i * 50_000,
            n_params=10_000 + i * 10_000,
            branch=f"exp/t/{i+1:02d}-x",
            commit_sha=f"sha{i}",
            round_attempt_id=f"u{i}",
        )
    preview = json.loads((tmp_path / "pareto_preview.json").read_text())
    assert len(preview) == 5
    deltas = [row["delta_ler"] for row in preview]
    assert deltas == sorted(deltas, reverse=True)


# ─── Regression tests for history/log/verdict-gating ────────────────────


def test_record_round_writes_history_row(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1,
            "status": "ok",
            "hypothesis": "gated_mlp + 3 layers",
            "delta_ler": 8e-5,
            "flops_per_syndrome": 10_000,
            "n_params": 5000,
            "checkpoint_path": "/tmp/round_1/checkpoint.pt",
        },
        verify_verdict="VERIFIED",
    )
    history = [
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 1
    assert history[0]["round"] == 1
    assert history[0]["hypothesis"] == "gated_mlp + 3 layers"
    assert history[0]["delta_ler"] == 8e-5


def test_record_round_appends_log(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={"round": 3, "status": "ok", "hypothesis": "try attention"},
        verify_verdict="VERIFIED",
    )
    log_text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "round 3" in log_text


def test_non_verified_rounds_skip_pareto(tmp_path: Path) -> None:
    """Only verify_verdict=='VERIFIED' admits into the Pareto archive."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1,
            "status": "ok",
            "hypothesis": "useless",
            "delta_ler": -2e-5,
            "flops_per_syndrome": 10_000,
            "n_params": 5000,
        },
        verify_verdict="FAILED",
    )
    history_rows = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    assert pareto == []


def test_non_verified_rounds_skip_pareto_when_verdict_missing(tmp_path: Path) -> None:
    """A round without a verdict (e.g., compile_error) must not enter Pareto."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 7,
            "status": "compile_error",
            "status_reason": "pydantic ValidationError on gnn.hidden_dim < 4",
            "hypothesis": "h7",
        },
        verify_verdict=None,
    )
    history_rows = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    assert pareto == []


def test_pareto_rows_include_worktree_fields(tmp_path: Path) -> None:
    """§15.7 superset fields flow through into pareto.json rows."""
    mem = RunMemory(tmp_path)
    _admit(
        mem,
        round=1,
        branch="exp/t/01-a",
        commit_sha="a1",
        round_attempt_id="u1",
        fork_from="baseline",
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    assert len(pareto) == 1
    row = pareto[0]
    assert row["branch"] == "exp/t/01-a"
    assert row["commit_sha"] == "a1"
    assert row["round_attempt_id"] == "u1"
    assert row["fork_from"] == "baseline"
