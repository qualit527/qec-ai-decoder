"""Task 5: run-round CLI accepts worktree flags (§15.7)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="legacy runner path needs torch")
def test_run_round_legacy_positional_still_works(tmp_path):
    """Lin's positional form must keep working."""
    env = Path("autoqec/envs/builtin/surface_d5_depol.yaml").absolute()
    cfg = Path("autoqec/example_db/gnn_small.yaml").absolute()
    out = tmp_path / "round_1"
    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "run-round",
         str(env), str(cfg), str(out), "--profile", "dev"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    metrics = json.loads(result.stdout)
    assert metrics["status"] in ("ok", "killed_by_safety")


def test_run_round_accepts_worktree_flags_without_running():
    """The --help page lists the new flags so callers can discover them."""
    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "run-round", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    for flag in ("--code-cwd", "--branch", "--fork-from", "--compose-mode", "--round-attempt-id"):
        assert flag in result.stdout, f"missing flag {flag}"
