# tests/test_pareto_atomic.py
import json
from pathlib import Path
from unittest.mock import patch

from autoqec.orchestration.memory import RunMemory


def test_update_pareto_uses_os_replace(tmp_path, monkeypatch):
    mem = RunMemory(tmp_path)
    calls = {"write_text": 0, "replace": 0}
    original_replace = __import__("os").replace

    def spy_replace(src, dst):
        calls["replace"] += 1
        return original_replace(src, dst)

    monkeypatch.setattr("os.replace", spy_replace)
    mem.update_pareto([{"round": 1, "delta_ler": 0.01}])
    assert calls["replace"] == 1


def test_update_pareto_leaves_old_on_crash(tmp_path, monkeypatch):
    """If write_text to tmp fails, pareto.json is unchanged."""
    mem = RunMemory(tmp_path)
    mem.update_pareto([{"round": 1, "delta_ler": 0.01}])
    before = Path(mem.pareto_path).read_text()

    def boom(*a, **kw):
        raise OSError("simulated disk-full")

    monkeypatch.setattr("pathlib.Path.write_text", boom)
    try:
        mem.update_pareto([{"round": 2, "delta_ler": 0.99}])
    except OSError:
        pass
    after = Path(mem.pareto_path).read_text()
    assert before == after
