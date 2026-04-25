"""Tests for demos/_lib/python_bin.sh.

The helper resolves the project venv python so Demo 1 and Demo 4 launchers
pick up the shared ``.venv`` even when the bash shell's ``PATH`` points at
a system Python that does not have ``torch`` installed (which is what
broke the advisor walkthrough on 2026-04-24).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER = REPO_ROOT / "demos" / "_lib" / "python_bin.sh"


def _requires_bash() -> None:
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this host")


def _run_discover(start_dir: Path, env: dict[str, str] | None = None) -> str:
    _requires_bash()
    bash_bin = shutil.which("bash")
    assert bash_bin is not None
    cmd = [
        bash_bin,
        "-c",
        f'set -euo pipefail; source "{HELPER.as_posix()}"; discover_demo_python "{start_dir.as_posix()}"',
    ]
    proc = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return proc.stdout.strip()


def test_helper_file_exists() -> None:
    assert HELPER.is_file(), "demos/_lib/python_bin.sh is required by Demo 1 + Demo 4"


def test_prefers_windows_venv_from_worktree_subdir(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    worktree = repo / ".worktrees" / "exp-1"
    venv = repo / ".venv" / "Scripts" / "python.exe"
    venv.parent.mkdir(parents=True)
    venv.write_text("", encoding="utf-8")
    venv.chmod(0o755)
    worktree.mkdir(parents=True)

    assert _run_discover(worktree) == str(venv.as_posix())


def test_prefers_unix_venv_when_no_windows_layout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    venv = repo / ".venv" / "bin" / "python"
    venv.parent.mkdir(parents=True)
    venv.write_text("", encoding="utf-8")
    venv.chmod(0o755)
    nested = repo / "demos" / "demo-X"
    nested.mkdir(parents=True)

    assert _run_discover(nested) == str(venv.as_posix())


def test_falls_back_when_no_venv_found(tmp_path: Path) -> None:
    # Unrelated tmp subdir with no .venv anywhere above it. On every host
    # we run tests on, either python3 or python resolves via PATH — we just
    # check the helper drops through to that branch rather than exploding.
    target = tmp_path / "elsewhere"
    target.mkdir()
    result = _run_discover(target)
    assert result in {"python3", "python"}, f"unexpected fallback: {result!r}"


def test_demo1_launcher_uses_helper() -> None:
    """Demo 1's run_quick.sh must source the shared helper so it picks up
    the project venv, not the system python that broke the 2026-04-24 walkthrough."""
    launcher = REPO_ROOT / "demos" / "demo-1-surface-d5" / "run_quick.sh"
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert "demos/_lib/python_bin.sh" in text
    assert "discover_demo_python" in text


def test_demo4_launcher_uses_helper() -> None:
    launcher = REPO_ROOT / "demos" / "demo-4-reward-hacking" / "run.sh"
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert "demos/_lib/python_bin.sh" in text
    assert "discover_demo_python" in text
