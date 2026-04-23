"""Issue #40: demo-5 failure diagnosis and recovery walkthrough acceptance tests."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path("demos/demo-5-failure-recovery")
HAS_BASH = shutil.which("bash") is not None
TEST_PYTHON = "/home/jinguxie/qec-ai-decoder/.venv/bin/python"


def _load_generate_module():
    spec = importlib.util.spec_from_file_location(
        "demo5_generate_diagnosis",
        DEMO_DIR / "generate_diagnosis.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def demo5_run():
    """Run demo-5 and return the run directory."""
    result = subprocess.run(
        ["bash", str(DEMO_DIR / "run.sh")],
        env={**dict(os.environ), "PYTHON_BIN": TEST_PYTHON},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Demo exited {result.returncode}\nstderr: {result.stderr}\nstdout: {result.stdout}"
    return Path("runs/demo-5")


@pytest.mark.skipif(not HAS_BASH, reason="bash is required for demo-5 script tests")
def test_demo5_exits_0():
    """Demo run.sh exits 0."""
    result = subprocess.run(
        ["bash", str(DEMO_DIR / "run.sh")],
        env={**dict(os.environ), "PYTHON_BIN": TEST_PYTHON},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "/diagnose-failure" in result.stdout


def test_diagnose_direct_round_dir(tmp_path: Path):
    """CLI diagnose handles direct round-dir input."""
    round_dir = tmp_path / "round_42"
    round_dir.mkdir()
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "metrics.json").write_text(
        json.dumps({
            "status": "killed_by_safety",
            "status_reason": "NaN rate 0.500",
            "train_wallclock_s": 1.0,
        }),
        encoding="utf-8",
    )
    (round_dir / "train.log").write_text("0\tnan\n1\tnan\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "diagnose", str(round_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out["root_cause"] == "nan_loss"
    assert out["has_metrics.json"] is True
    assert out["read_only"] is True


def test_diagnose_run_dir_picks_latest(tmp_path: Path):
    """CLI diagnose identifies the latest round when passed a run directory."""
    for i in (1, 2, 3):
        rd = tmp_path / f"round_{i}"
        rd.mkdir()
        (rd / "metrics.json").write_text(
            json.dumps({"status": "ok", "delta_ler": 0.01 * i}),
            encoding="utf-8",
        )

    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "diagnose", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert str(tmp_path / "round_3") in out["path"]


def test_diagnose_run_dir_picks_latest_numeric_round(tmp_path: Path):
    """Round selection must be numeric, not lexicographic."""
    for i in (2, 9, 10):
        rd = tmp_path / f"round_{i}"
        rd.mkdir()
        (rd / "metrics.json").write_text(
            json.dumps({"status": "ok", "delta_ler": 0.01 * i}),
            encoding="utf-8",
        )

    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "diagnose", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert str(tmp_path / "round_10") in out["path"]


def test_diagnose_compile_error(tmp_path: Path):
    """Diagnose identifies compile_error from synthetic fixture."""
    fixture_dir = Path("tests/fixtures/diagnose") / "compile_error"
    round_dir = tmp_path / "compile_error"
    shutil.copytree(fixture_dir, round_dir)
    (round_dir / "train.log").write_text("ValueError: hidden_dim must be >= 4\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "diagnose", str(round_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out["root_cause"] == "compile_error"
    assert "validation" in "\n".join(out["signals"]).lower()
    assert out["has_config.yaml"] is True
    assert out["has_metrics.json"] is True


def test_diagnose_pydantic_word_alone_is_not_compile_error(tmp_path: Path):
    """Loose 'pydantic' matches should not force compile_error."""
    round_dir = tmp_path / "round_43"
    round_dir.mkdir()
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "train_error",
                "status_reason": "pydantic imported successfully before runtime failure",
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "train.log").write_text("warning: pydantic cache warmed\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "cli.autoqec", "diagnose", str(round_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    out = json.loads(result.stdout)
    assert out["root_cause"] != "compile_error"


def test_generate_diagnosis_does_not_match_banana_or_boom(tmp_path: Path):
    """Token matching should avoid substring false positives."""
    module = _load_generate_module()
    round_dir = tmp_path / "round_7"
    round_dir.mkdir()
    (round_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "train_error",
                "status_reason": "banana boom pydantic chatter only",
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "train.log").write_text("banana\nboom\n", encoding="utf-8")
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")

    module.generate(round_dir)

    diagnosis = (round_dir / "diagnosis.md").read_text(encoding="utf-8").lower()
    assert "nan_loss" not in diagnosis
    assert "\noom:" not in diagnosis


def test_demo5_run_script_uses_overridable_python_bin() -> None:
    text = (DEMO_DIR / "run.sh").read_text(encoding="utf-8")
    assert 'PYTHON_BIN="${PYTHON_BIN:-python}"' in text


@pytest.mark.skipif(not HAS_BASH, reason="bash is required for demo-5 script tests")
def test_demo5_generates_diagnosis_md(demo5_run: Path):
    """Each failed round gets a diagnosis.md with root cause, evidence, and fix."""
    assert (demo5_run / "round_0").exists()
    for round_dir in sorted(demo5_run.glob("round_*")):
        diag = round_dir / "diagnosis.md"
        assert diag.exists(), f"Missing {round_dir.name}/diagnosis.md"
        text = diag.read_text(encoding="utf-8")
        assert "/diagnose-failure" in text
        assert "Root Cause" in text
        assert "Evidence" in text
        assert "Recommended Fix" in text
        assert "```yaml" in text
        assert "metrics.json:" in text
        assert "not apply" in text.lower() or "not applied" in text.lower()
