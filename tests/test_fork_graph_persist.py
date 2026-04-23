"""Regression for issue #50 — `fork_graph.json` persists to `run_dir/`.

Before issue #50, `fork_graph` only lived inside the Ideator's L3 context
and was never serialised; the test plan's Phase 3.4 / 4.1 invariants
(e.g. `round_1.parent == "baseline"`) had no artifact to assert against.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round


def _round_metrics(round_idx: int, branch: str, fork_from: str) -> dict:
    return {
        "round": round_idx,
        "status": "ok",
        "delta_ler": 0.01,
        "flops_per_syndrome": 1000,
        "n_params": 100,
        "train_wallclock_s": 1.0,
        "eval_wallclock_s": 0.1,
        "vram_peak_gb": 0.0,
        "branch": branch,
        "commit_sha": "0" * 40,
        "round_attempt_id": f"aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaa{round_idx}",
        "fork_from": fork_from,
    }


def test_update_fork_graph_writes_atomically(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    graph = {"run_id": "r", "nodes": [{"branch": "baseline"}], "pareto_front": []}

    mem.update_fork_graph(graph)

    fg_path = tmp_path / "fork_graph.json"
    assert fg_path.exists()
    assert json.loads(fg_path.read_text(encoding="utf-8")) == graph
    # Atomic writer uses `os.replace` on a sibling tmp file.
    assert not (tmp_path / "fork_graph.json.tmp").exists()


def test_update_fork_graph_crash_leaves_previous_intact(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    good = {"run_id": "r", "nodes": [{"branch": "baseline"}], "pareto_front": []}
    mem.update_fork_graph(good)

    with mock.patch("os.replace", side_effect=RuntimeError("simulated crash")):
        with pytest.raises(RuntimeError):
            mem.update_fork_graph({"run_id": "r", "nodes": [{"branch": "X"}]})

    # Previous good file remains readable.
    assert json.loads((tmp_path / "fork_graph.json").read_text(encoding="utf-8")) == good


def test_record_round_persists_fork_graph_after_round_one(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        _round_metrics(round_idx=1, branch="exp/run/01-first", fork_from="baseline"),
    )

    graph = json.loads((tmp_path / "fork_graph.json").read_text(encoding="utf-8"))
    branches = [n.get("branch") for n in graph["nodes"]]
    assert "baseline" in branches
    assert "exp/run/01-first" in branches
    round_one = next(n for n in graph["nodes"] if n.get("branch") == "exp/run/01-first")
    assert round_one["parent"] == "baseline"


def test_record_round_fork_graph_tracks_fork_lineage(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        _round_metrics(round_idx=1, branch="exp/run/01-a", fork_from="baseline"),
    )
    record_round(
        mem,
        _round_metrics(round_idx=2, branch="exp/run/02-b", fork_from="exp/run/01-a"),
    )

    graph = json.loads((tmp_path / "fork_graph.json").read_text(encoding="utf-8"))
    parents = {n.get("branch"): n.get("parent") for n in graph["nodes"] if n.get("branch")}
    assert parents["exp/run/01-a"] == "baseline"
    assert parents["exp/run/02-b"] == "exp/run/01-a"
