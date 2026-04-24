from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_pytest_project(tmp_path: Path) -> None:
    (tmp_path / "conftest.py").write_text(
        (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        textwrap.dedent(
            """
        [tool.pytest.ini_options]
        markers = [
          "slow: marks wall-clock-heavy tests that require explicit opt-in",
        ]
        """
        ),
        encoding="utf-8",
    )
    (tmp_path / "test_sample.py").write_text(
        textwrap.dedent(
            """
        import pytest

        @pytest.mark.slow
        def test_slow():
            assert True

        def test_fast():
            assert True
        """
        ),
        encoding="utf-8",
    )


def test_slow_tests_are_skipped_even_with_explicit_non_integration_marker(tmp_path: Path) -> None:
    _write_minimal_pytest_project(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-m", "not integration"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert re.search(r"1 passed, 1 skipped", result.stdout), result.stdout
    assert "need --run-slow option to run" in result.stdout


def test_run_slow_opt_in_restores_slow_tests(tmp_path: Path) -> None:
    _write_minimal_pytest_project(tmp_path)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--run-slow"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert re.search(r"2 passed", result.stdout), result.stdout
