from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SHOWCASE_DIR = Path("demos/demo-5-failure-recovery/showcase")
DEMO_README = Path("demos/demo-5-failure-recovery/README.md")


def _load_build_report_module():
    spec = importlib.util.spec_from_file_location(
        "demo5_visual_showcase_build_report",
        SHOWCASE_DIR / "build_report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_round(root: Path, name: str, status: str, reason: str, log_text: str) -> None:
    round_dir = root / "runs" / "demo-5" / name
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "metrics.json").write_text(
        json.dumps({"status": status, "status_reason": reason, "train_wallclock_s": 0.5}),
        encoding="utf-8",
    )
    (round_dir / "train.log").write_text(log_text, encoding="utf-8")
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "diagnosis.md").write_text(
        f"# Diagnosis\n\n## Root Cause\n{reason}\n\n## Evidence\nmetrics.json: {status}\n\n## Recommended Fix\nDo not apply automatically.\n",
        encoding="utf-8",
    )


def _write_round_with_real_demo_heading(root: Path) -> None:
    round_dir = root / "runs" / "demo-5" / "round_0"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "metrics.json").write_text(
        json.dumps({"status": "compile_error", "status_reason": "hidden_dim validation failed: -1 < 4"}),
        encoding="utf-8",
    )
    (round_dir / "train.log").write_text("ValueError: hidden_dim must be >= 4", encoding="utf-8")
    (round_dir / "config.yaml").write_text("gnn:\n  hidden_dim: -1\n", encoding="utf-8")
    (round_dir / "diagnosis.md").write_text(
        "# Diagnosis: round_0\n\n"
        "## Root Cause\ncompile_error: hidden_dim validation failed: -1 < 4\n\n"
        "## Evidence\n- config.yaml:2: hidden_dim: -1\n"
        "- metrics.json: status_reason=hidden_dim validation failed: -1 < 4\n\n"
        "## Recommended Fix (not applied)\n```yaml\ngnn:\n  hidden_dim: 32   # was -1\n```\n\n"
        "**Note:** The system does not apply fixes automatically. Apply the suggested patch manually.\n",
        encoding="utf-8",
    )
def test_demo5_visual_report_explains_expected_failures_and_evidence(tmp_path: Path) -> None:
    build_report = _load_build_report_module()
    _write_round(tmp_path, "round_0", "compile_error", "hidden_dim validation failed: -1 < 4", "ValueError: hidden_dim must be >= 4")
    _write_round(tmp_path, "round_1", "killed_by_safety", "NaN rate 0.500", "0\tnan\n1\tnan")
    _write_round(tmp_path, "round_2", "train_error", "RuntimeError: CUDA out of memory", "CUDA out of memory")

    output_dir = tmp_path / "runs" / "demo-5-showcase"
    phases = [{"name": "demo5", "cmd": "bash demos/demo-5-failure-recovery/run.sh", "exit_code": 0, "elapsed_s": 0.7}]
    summary = build_report.build_summary(tmp_path, output_dir, phases)
    markdown_path, html_path, summary_path = build_report.write_reports(summary, output_dir)

    html = html_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert "Demo 5 Failure Diagnosis Showcase" in html
    assert "Loaded 3 diagnosis.md artifacts" in html
    assert "Broken Round → Root Cause → Safe Fix" in html
    assert "How to view this report" in html
    assert "Claim → Evidence → Why it matters" in html
    assert "compile_error" in html
    assert "nan_loss" in html
    assert "oom" in html
    assert "runs/demo-5/round_0/diagnosis.md" in html
    assert "demo5 showcase healthy" in markdown


def test_demo5_visual_report_uses_diagnosis_md_content_instead_of_generic_note(tmp_path: Path) -> None:
    build_report = _load_build_report_module()
    _write_round(
        tmp_path,
        "round_0",
        "compile_error",
        "hidden_dim validation failed: -1 < 4",
        "ValueError: hidden_dim must be >= 4",
    )

    output_dir = tmp_path / "runs" / "demo-5-showcase"
    summary = build_report.build_summary(
        tmp_path,
        output_dir,
        [{"name": "demo5", "cmd": "bash demos/demo-5-failure-recovery/run.sh", "exit_code": 0, "elapsed_s": 0.7}],
    )
    _, html_path, _ = build_report.write_reports(summary, output_dir)
    html = html_path.read_text(encoding="utf-8")

    assert "Agent Diagnosis" in html
    assert "metrics.json: compile_error" in html
    assert "Do not apply automatically." in html
    assert "The round is intentionally broken; the success criterion" not in html


def test_demo5_visual_report_parses_real_recommended_fix_heading(tmp_path: Path) -> None:
    build_report = _load_build_report_module()
    _write_round_with_real_demo_heading(tmp_path)

    output_dir = tmp_path / "runs" / "demo-5-showcase"
    summary = build_report.build_summary(
        tmp_path,
        output_dir,
        [{"name": "demo5", "cmd": "bash demos/demo-5-failure-recovery/run.sh", "exit_code": 0, "elapsed_s": 0.7}],
    )
    _, html_path, _ = build_report.write_reports(summary, output_dir)
    html = html_path.read_text(encoding="utf-8")

    assert "hidden_dim: 32" in html
    assert "# was -1" in html
    assert "No recommended fix text found" not in html
    assert "No evidence text found" not in html
    assert "No root-cause text found" not in html


def test_demo5_visual_report_keeps_recommended_fix_to_patch_only(tmp_path: Path) -> None:
    build_report = _load_build_report_module()
    _write_round_with_real_demo_heading(tmp_path)

    output_dir = tmp_path / "runs" / "demo-5-showcase"
    summary = build_report.build_summary(
        tmp_path,
        output_dir,
        [{"name": "demo5", "cmd": "bash demos/demo-5-failure-recovery/run.sh", "exit_code": 0, "elapsed_s": 0.7}],
    )
    _, html_path, _ = build_report.write_reports(summary, output_dir)
    html = html_path.read_text(encoding="utf-8")

    assert "Recommended fix" in html
    assert "hidden_dim: 32" in html
    assert "The system does not apply fixes automatically" not in html
    assert "Apply the suggested patch manually" not in html
    assert "Expected failures become useful evidence" not in html
    assert "Expected failure, successful diagnosis" not in html


def test_demo5_showcase_run_script_is_standalone_and_portable() -> None:
    script = (SHOWCASE_DIR / "run.sh").read_text(encoding="utf-8")
    assert 'if [[ -z "${PYTHON_BIN:-}" ]]; then' in script
    assert "PYTHONPATH" in script
    assert "demos/demo-5-failure-recovery/run.sh" in script
    assert "build_report.py" in script
    assert "demo-6" not in script
    assert "node" not in script.lower()


def test_demo5_readme_contains_copy_paste_agent_prompt() -> None:
    readme = DEMO_README.read_text(encoding="utf-8")
    assert "Copy-paste Agent Prompt" in readme
    assert "demos/demo-5-failure-recovery/showcase/run.sh" in readme
    assert "report.html" in readme
    assert "file://" in readme
    assert "http://127.0.0.1" in readme
