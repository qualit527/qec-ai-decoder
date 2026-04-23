"""Phase 5.5 — pareto.json must be written via tmp-file + atomic rename.

5.5.1  memory.py uses ``os.replace`` (grep-level assertion).
5.5.2  a writer killed between tmp-write and rename leaves the original
       pareto.json intact — never truncated, never empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_memory_module_uses_atomic_rename() -> None:
    """5.5.1 — sentinel check: memory.py must call ``os.replace`` on the
    Pareto write path. If this fails, the atomic-rename guarantee lapsed."""
    from autoqec.orchestration import memory

    src = Path(memory.__file__).read_text(encoding="utf-8")
    assert "os.replace" in src, (
        "autoqec/orchestration/memory.py must use os.replace for pareto.json writes "
        "(§5.5 atomic-rename contract)"
    )


def test_update_pareto_is_atomic_on_writer_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5.5.2 — simulate a kill between the tmp-file write and the final
    rename. The pre-existing ``pareto.json`` must survive unchanged: no
    truncation, no partial JSON, no empty file."""
    from autoqec.orchestration.memory import RunMemory

    mem = RunMemory(tmp_path / "run")
    baseline = [{"branch": "exp/t/01-ok", "delta_ler": 1e-4}]
    mem.update_pareto(baseline)
    original_bytes = mem.pareto_path.read_bytes()

    # Patch os.replace on the *memory* module (that's the binding the method
    # actually calls) to simulate a crash AFTER the tmp write but BEFORE the
    # atomic swap — the exact point where the old file must survive.
    import autoqec.orchestration.memory as memory_module

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated mid-write kill")

    monkeypatch.setattr(memory_module.os, "replace", _boom)

    with pytest.raises(RuntimeError, match="simulated mid-write kill"):
        mem.update_pareto([{"branch": "exp/t/02-corrupt-target", "delta_ler": 9e9}])

    # Old file content is byte-identical — no truncation, no partial write.
    assert mem.pareto_path.read_bytes() == original_bytes
    # And the parsed content is still the pre-crash front.
    assert json.loads(mem.pareto_path.read_text(encoding="utf-8")) == baseline
