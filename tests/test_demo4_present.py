"""Structural tests for the Demo 4 narrated presentation runner.

These run without GPU/torch-heavy work by exercising just the pure-helper
functions and then smoke-running the full script in -m "integration" mode
when the user explicitly opts in via --run-integration. The structural
checks are cheap: they assert the five phase headers, the argparse
surface, and the skill wiring.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENT_PY = REPO_ROOT / "demos" / "demo-4-reward-hacking" / "present.py"
PRESENT_SH = REPO_ROOT / "demos" / "demo-4-reward-hacking" / "present.sh"


def test_present_py_exists_and_is_executable_from_python() -> None:
    assert PRESENT_PY.is_file()
    # --help must succeed and mention every required flag.
    proc = subprocess.run(
        [sys.executable, str(PRESENT_PY), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    for flag in ("--env-yaml", "--run-dir", "--n-shots", "--n-seeds", "--n-probes", "--no-png"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_present_py_declares_five_phases() -> None:
    """The skill doc promises five labeled phases; enforce that the source
    still actually prints PHASE 1..5 headers so the skill narration stays
    in sync with the implementation."""
    text = PRESENT_PY.read_text(encoding="utf-8")
    for i in range(1, 6):
        assert f"PHASE {i} " in text, f"PHASE {i} header missing in present.py"


def test_present_sh_sources_shared_venv_helper() -> None:
    """The launcher must source demos/_lib/python_bin.sh so it keeps
    picking up the project .venv (same root-cause as the 2026-04-24 fix)."""
    assert PRESENT_SH.is_file()
    text = PRESENT_SH.read_text(encoding="utf-8")
    assert "demos/_lib/python_bin.sh" in text
    assert "discover_demo_python" in text
    assert "present.py" in text


def test_skill_demo_order_routes_demo4_through_present_sh() -> None:
    """If someone reverts Demo 4 to run.sh in the skill doc, we lose the
    narrated-mode walkthrough silently. Guard against that."""
    skill = REPO_ROOT / ".claude" / "skills" / "demo-presenter" / "SKILL.md"
    text = skill.read_text(encoding="utf-8")
    assert "bash demos/demo-4-reward-hacking/present.sh" in text, (
        "SKILL.md must advertise present.sh as Demo 4's live-walkthrough command"
    )
    # And the five-phase narration guidance must still be present.
    for i in range(1, 6):
        assert f"PHASE {i} " in text, f"SKILL.md lost PHASE {i} narration guidance"


def test_live_present_demo_wires_demo4_to_present_sh() -> None:
    """The live presenter must run present.sh (not run.sh) for Demo 4."""
    live = REPO_ROOT / ".claude" / "skills" / "demo-presenter" / "scripts" / "live_present_demo.py"
    text = live.read_text(encoding="utf-8")
    assert "demos/demo-4-reward-hacking/present.sh" in text


@pytest.mark.integration
def test_present_py_emits_summary_json_and_ascii_sections(tmp_path: Path) -> None:
    """End-to-end smoke: run present.py, assert summary JSON + the five
    phase markers actually appeared in stdout. Gated on --run-integration
    because it does real stim sampling and calls into independent_verify."""
    run_dir = tmp_path / "demo-4"
    proc = subprocess.run(
        [
            sys.executable,
            str(PRESENT_PY),
            "--run-dir",
            str(run_dir),
            "--n-shots",
            "1000",
            "--n-seeds",
            "3",
            "--n-probes",
            "500",
            "--memorize-shots",
            "2000",
            "--no-png",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"present.py failed:\n{proc.stdout}\n{proc.stderr}"
    for i in range(1, 6):
        assert f"PHASE {i} " in proc.stdout, f"PHASE {i} header missing in stdout"

    summary_path = run_dir / "present_summary.json"
    assert summary_path.is_file(), "present_summary.json was not written"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"] in {"FAILED", "SUSPICIOUS"}, (
        f"memorizer must be rejected; got {summary['verdict']}"
    )
    assert summary["phase2"]["seen_hit_rate"] > 0.9, (
        "sanity: memorized-shots hit rate should be ~100%"
    )
    assert len(summary["phase4_guards"]) == 3
