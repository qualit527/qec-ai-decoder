from __future__ import annotations

from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE_TEXT = (ROOT / "Makefile").read_text()
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
CI_WORKFLOW_TEXT = (ROOT / ".github/workflows/ci.yml").read_text()


def _target_body(target: str) -> str:
    match = re.search(rf"^{target}:\n((?:\t.*\n)+)", MAKEFILE_TEXT, re.MULTILINE)
    assert match is not None, f"missing make target: {target}"
    return match.group(1)


def test_make_test_target_stays_fast_by_default() -> None:
    body = _target_body("test")
    assert "$(PYTEST)" in body
    assert "--cov" not in body


def test_make_coverage_target_opt_in_collects_coverage() -> None:
    body = _target_body("coverage")
    assert "$(PYTEST)" in body
    assert "--cov" in body


def test_pytest_default_selection_is_configured_in_pyproject() -> None:
    pytest_config = PYPROJECT["tool"]["pytest"]["ini_options"]
    assert pytest_config["testpaths"] == ["tests"]
    assert pytest_config["addopts"] == '-m "not integration" -v'


def test_coverage_settings_are_centralized_in_pyproject() -> None:
    coverage_config = PYPROJECT["tool"]["coverage"]
    assert coverage_config["run"]["source"] == ["autoqec", "cli"]
    assert coverage_config["report"]["show_missing"] is True
    assert coverage_config["report"]["skip_covered"] is True


def test_ci_runs_explicit_coverage_target() -> None:
    assert "run: make coverage" in CI_WORKFLOW_TEXT
