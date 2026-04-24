from __future__ import annotations

import re
from pathlib import Path


def test_makefile_exposes_manual_integration_target_without_changing_default_gate() -> None:
    text = Path("Makefile").read_text(encoding="utf-8")
    assert re.search(r"^test:\n\t\$\(PYTEST\) tests/ -m \"not integration\" -v", text, re.MULTILINE)
    assert re.search(
        r"^test-integration:\n\t\$\(PYTEST\) tests/ -m \"integration\" -v --run-integration$",
        text,
        re.MULTILINE,
    )


def test_integration_entry_is_documented_for_contributors_and_agents() -> None:
    plan = Path("docs/test-plan.md").read_text(encoding="utf-8")
    assert "make test-integration" in plan
    assert '--run-integration' in plan
    assert "make lint" in plan
    assert "make test" in plan

    readme = Path("README.md").read_text(encoding="utf-8")
    assert "docs/test-plan.md" in readme
    assert "make test-integration" in readme

    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "docs/test-plan.md" in agents
    assert "make test-integration" in agents


def test_slow_test_opt_in_is_documented_for_contributors_and_agents() -> None:
    plan = Path("docs/test-plan.md").read_text(encoding="utf-8")
    assert "--run-slow" in plan

    claude = Path("CLAUDE.md").read_text(encoding="utf-8")
    assert "--run-slow" in claude

    agents = Path("AGENTS.md").read_text(encoding="utf-8")
    assert "--run-slow" in agents
