"""Tests for the Demo 4 HTML visualization generator.

``present_html.py`` renders a self-contained HTML page from the verifier's
JSON report so that ``run.sh`` / ``present.sh`` can open a browser window
at the end of the demo. These tests exercise the pure-rendering path on a
synthetic report fixture so they stay CPU-only and never touch the real
``stim`` sampler.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENT_HTML = REPO_ROOT / "demos" / "demo-4-reward-hacking" / "present_html.py"
RUN_SH = REPO_ROOT / "demos" / "demo-4-reward-hacking" / "run.sh"
PRESENT_SH = REPO_ROOT / "demos" / "demo-4-reward-hacking" / "present.sh"


@pytest.fixture
def demo4_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "demo-4" / "round_0"
    run_dir.mkdir(parents=True)
    (run_dir / "verification_report.json").write_text(
        json.dumps(
            {
                "verdict": "FAILED",
                "ler_holdout": 0.2292,
                "ler_holdout_ci": [0.2178, 0.2406],
                "delta_ler_holdout": -0.2166,
                "ler_shuffled": 0.2292,
                "ablation_sanity_ok": True,
                "holdout_seeds_used": [9000, 9001, 9002],
                "seed_leakage_check_ok": True,
                "notes": "n_shots=5000, plain_ler=0.0126, pred_ler=0.2292, shuffled_ler=0.2292",
                "paired_eval_bundle_id": "4d89644f-f0e7-5442-965f-4ffc11047fbe",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def _run(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "AUTOQEC_NO_OPEN": "1"}
    return subprocess.run(
        [sys.executable, str(PRESENT_HTML), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


def test_present_html_exists_and_advertises_flags() -> None:
    assert PRESENT_HTML.is_file()
    proc = _run("--help")
    assert proc.returncode == 0, proc.stderr
    for flag in ("--run-dir", "--out", "--no-open"):
        assert flag in proc.stdout, f"{flag} missing from --help"


def test_present_html_writes_index_with_verdict_banner(demo4_run_dir: Path) -> None:
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0, f"stderr={proc.stderr}\nstdout={proc.stdout}"
    out = demo4_run_dir / "visualizations" / "index.html"
    assert out.is_file()
    body = out.read_text(encoding="utf-8")
    assert "<!doctype html>" in body.lower()
    assert "verdict: FAILED" in body
    assert "cheat correctly rejected" in body
    for guard in ("seed-leakage hygiene", "paired bootstrap CI", "ablation sanity"):
        assert guard in body, f"{guard} card missing from HTML"


def test_present_html_renders_ler_numbers(demo4_run_dir: Path) -> None:
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0
    body = (demo4_run_dir / "visualizations" / "index.html").read_text(encoding="utf-8")
    assert "0.0126" in body, "plain MWPM LER (from notes) missing"
    assert "0.2292" in body, "memorizer holdout LER missing"
    assert "-0.2166" in body, "Δ_LER missing"
    assert "4d89644f-f0e7-5442-965f-4ffc11047fbe" in body, "paired eval bundle id missing"


def test_present_html_falls_back_when_summary_absent(demo4_run_dir: Path) -> None:
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0
    body = (demo4_run_dir / "visualizations" / "index.html").read_text(encoding="utf-8")
    # Phase 2 hit-rate card should show the "not computed" placeholder.
    assert "not computed" in body
    assert "present.sh" in body, "placeholder must point users at the narrated mode"


def test_present_html_uses_summary_when_available(demo4_run_dir: Path) -> None:
    (demo4_run_dir / "present_summary.json").write_text(
        json.dumps(
            {
                "verdict": "FAILED",
                "phase2": {
                    "seen_hit_rate": 1.0,
                    "fresh_train_hit_rate": 0.002,
                    "holdout_hit_rate": 0.0015,
                    "n_probes": 2000,
                },
            }
        ),
        encoding="utf-8",
    )
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0
    body = (demo4_run_dir / "visualizations" / "index.html").read_text(encoding="utf-8")
    assert "100.0%" in body, "memorized-shots hit rate missing"
    assert "0.2%" in body, "fresh-train hit rate missing"
    assert "not computed" not in body, "placeholder still shown despite present_summary.json"


def test_present_html_handles_suspicious_verdict(demo4_run_dir: Path) -> None:
    report_path = demo4_run_dir / "verification_report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["verdict"] = "SUSPICIOUS"
    data["delta_ler_holdout"] = 0.0001  # inside CI
    report_path.write_text(json.dumps(data), encoding="utf-8")
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0
    body = (demo4_run_dir / "visualizations" / "index.html").read_text(encoding="utf-8")
    assert "verdict: SUSPICIOUS" in body
    assert "kept off the Pareto front" in body


def test_present_html_no_broken_template_placeholders(demo4_run_dir: Path) -> None:
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0
    body = (demo4_run_dir / "visualizations" / "index.html").read_text(encoding="utf-8")
    # Catch f-string leftovers / unformatted placeholders.
    for bad in ("{report", "{html.escape", "{summary", "{float(", "{verdict}"):
        assert bad not in body, f"unreplaced template fragment: {bad}"


def test_run_sh_invokes_html_generator() -> None:
    """run.sh must call present_html.py after a successful verdict so the
    user gets an HTML visualization at the end of the demo."""
    text = RUN_SH.read_text(encoding="utf-8")
    assert "present_html.py" in text
    assert "rendering HTML visualization" in text


def test_present_sh_invokes_html_generator() -> None:
    """present.sh (narrated mode) must also generate the HTML, which picks
    up the richer present_summary.json hit-rate data."""
    text = PRESENT_SH.read_text(encoding="utf-8")
    assert "present_html.py" in text
    assert "rendering HTML visualization" in text
    # Must still propagate present.py's exit code.
    assert "exit $rc" in text or "exit ${rc}" in text


def test_failed_verdict_uses_green_banner(demo4_run_dir: Path) -> None:
    """For demo-4 the 'good' outcome is that the cheat is rejected. A red
    FAILED banner would mislead users into thinking the verifier broke, so
    the banner color must track DEMO OUTCOME: FAILED→green, VERIFIED→red."""
    proc = _run("--run-dir", str(demo4_run_dir), "--no-open")
    assert proc.returncode == 0
    body = (demo4_run_dir / "visualizations" / "index.html").read_text(encoding="utf-8")
    # Green (rgba(34,197,94,…)) must be wired to .banner.FAILED, not .VERIFIED.
    assert ".banner.FAILED{background:rgba(34,197,94" in body, (
        "FAILED banner must use the green --ok color (cheat rejected = good)"
    )
    assert ".banner.VERIFIED{background:rgba(185,28,28" in body, (
        "VERIFIED banner must use the red --bad color (cheat admitted = broken)"
    )


def test_present_html_respects_autoqec_no_open(demo4_run_dir: Path) -> None:
    """When AUTOQEC_NO_OPEN=1 is set, the generator must not print the
    'opened in default browser' line (used as a proxy for the real call)."""
    env = {**os.environ, "AUTOQEC_NO_OPEN": "1"}
    proc = subprocess.run(
        [sys.executable, str(PRESENT_HTML), "--run-dir", str(demo4_run_dir)],
        check=False, capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )
    assert proc.returncode == 0
    assert "opened in default browser" not in proc.stdout
