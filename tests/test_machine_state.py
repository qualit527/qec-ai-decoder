"""Unit tests for autoqec.tools.machine_state (Task A2.2).

GPU probing is covered by a separate integration test; these tests
exercise the history-derived branches without torch.
"""
from __future__ import annotations

import json
from pathlib import Path


def _write_history(run_dir: Path, rounds: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "history.jsonl").open("w", encoding="utf-8") as f:
        for record in rounds:
            f.write(json.dumps(record) + "\n")


def test_machine_state_empty_history(tmp_path: Path) -> None:
    from autoqec.tools.machine_state import machine_state

    state = machine_state(tmp_path / "empty")
    assert state["history_timings"]["rounds_so_far"] == 0
    assert state["history_timings"]["wall_clock_mean_s"] == 0
    assert state["history_timings"]["wall_clock_p95_s"] == 0
    assert state["history_timings"]["killed_by_safety_count"] == 0
    assert state["budget"]["total_wallclock_s_spent"] == 0


def test_machine_state_aggregates_timings_and_killed_count(tmp_path: Path) -> None:
    from autoqec.tools.machine_state import machine_state

    _write_history(
        tmp_path / "run",
        rounds=[
            {"round": 1, "status": "ok", "train_wallclock_s": 10.0, "eval_wallclock_s": 2.0, "n_params": 5000},
            {"round": 2, "status": "killed_by_safety", "train_wallclock_s": 3.0, "eval_wallclock_s": 0.0, "n_params": 50000},
            {"round": 3, "status": "ok", "train_wallclock_s": 20.0, "eval_wallclock_s": 5.0, "n_params": 8000},
        ],
    )
    state = machine_state(tmp_path / "run")
    timings = state["history_timings"]
    assert timings["rounds_so_far"] == 3
    assert timings["killed_by_safety_count"] == 1
    # (10+2) + (3+0) + (20+5) = 40; mean = 40/3
    assert timings["wall_clock_mean_s"] == 40 / 3
    # p95 index: int(0.95 * 3) == 2 → max of sorted = 25
    assert timings["wall_clock_p95_s"] == 25.0
    assert timings["params_vs_time"] == [(5000, 12.0), (50000, 3.0), (8000, 25.0)]
    assert state["budget"]["total_wallclock_s_spent"] == 40.0
    assert state["budget"]["total_wallclock_s_remaining"] is None


def test_machine_state_accepts_budget_remaining_override(tmp_path: Path) -> None:
    """Callers (the orchestrator) plug in the outer time budget."""
    from autoqec.tools.machine_state import machine_state

    state = machine_state(tmp_path / "new", total_wallclock_s_budget=3600)
    assert state["budget"]["total_wallclock_s_remaining"] == 3600


def test_machine_state_gpu_section_is_present_even_without_cuda(tmp_path: Path) -> None:
    """When torch is missing or CUDA is unavailable, gpu is {} (not absent)."""
    from autoqec.tools.machine_state import machine_state

    state = machine_state(tmp_path / "gpuless")
    assert "gpu" in state
    assert isinstance(state["gpu"], dict)


def test_gpu_snapshot_swallows_driver_errors_from_is_available(monkeypatch) -> None:
    """Codex review (medium): the docstring promises *any* failure returns {};
    the original code only guarded `import torch`, so `is_available()` raising
    (e.g. driver init error) would crash plan assembly."""
    import sys
    import types

    fake_torch = types.ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("simulated driver failure")

    fake_torch.cuda = _FakeCuda()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from autoqec.tools.machine_state import _gpu_snapshot

    assert _gpu_snapshot() == {}
