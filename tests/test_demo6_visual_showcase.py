from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SHOWCASE_DIR = Path("demos/demo-6-advisor-replay/showcase")
DEMO_README = Path("demos/demo-6-advisor-replay/README.md")


def _load_build_report_module():
    spec = importlib.util.spec_from_file_location(
        "demo6_visual_showcase_build_report",
        SHOWCASE_DIR / "build_report.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _materialize_replay(root: Path) -> None:
    round_dir = root / "runs" / "demo-6-replay" / "demo-run" / "round_1"
    round_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        round_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "repo": {"commit_sha": "abc123", "branch": "main", "dirty": False},
            "environment": {"env_yaml_path": "autoqec/envs/builtin/surface_d5_depol.yaml", "env_yaml_sha256": "deadbeef"},
            "round": {"command_line": ["python", "-m", "cli.autoqec", "run", "--no-llm"]},
            "artifacts": {"checkpoint": "checkpoint.pt", "metrics": "metrics.json"},
        },
    )
    original = {
        "verdict": "VERIFIED",
        "holdout_seeds_used": [9000, 9001],
        "paired_eval_bundle_id": "bundle-1",
        "ler_holdout": 0.1,
        "delta_ler_holdout": 0.02,
        "ler_shuffled": 0.11,
        "ler_holdout_ci": [0.08, 0.12],
    }
    replay = {**original, "delta_ler_holdout": 0.0200001}
    _write_json(round_dir / "verification_report.original.json", original)
    _write_json(round_dir / "verification_report.json", replay)
    (root / "runs").mkdir(exist_ok=True)
    (root / "runs" / "demo-run.tar.gz").write_text("tarball", encoding="utf-8")


def test_demo6_collect_replay_scopes_artifacts_to_demo6_replay(tmp_path: Path) -> None:
    build_report = _load_build_report_module()
    _materialize_replay(tmp_path)

    stale_round = tmp_path / "runs" / "other-demo" / "round_1"
    stale_round.mkdir(parents=True)
    _write_json(stale_round / "verification_report.original.json", {"verdict": "FAILED"})
    _write_json(stale_round / "verification_report.json", {"verdict": "FAILED"})
    stale_package = tmp_path / "runs" / "zz-stale.tar.gz"
    stale_package.write_text("stale", encoding="utf-8")

    summary = build_report.build_summary(tmp_path, tmp_path / "runs" / "demo-6-showcase", [])

    assert summary["round_dir"] == "runs/demo-6-replay/demo-run/round_1"
    assert summary["package"] == "runs/demo-run.tar.gz"
    assert summary["verdict"] == "VERIFIED"


def test_demo6_comparison_uses_verify_report_schema_and_strict_bool_float_matching() -> None:
    build_report = _load_build_report_module()

    fields = build_report._comparison_fields()
    assert "ablation_sanity_ok" in fields
    assert "seed_leakage_check_ok" in fields
    assert "notes" in fields
    assert build_report._same(True, 1) is False
    assert build_report._same(0.1, 0.1000000001) is True
    assert build_report._same(0.1, 0.101) is False


def test_demo6_visual_report_explains_offline_replay_evidence(tmp_path: Path) -> None:
    build_report = _load_build_report_module()
    _materialize_replay(tmp_path)

    output_dir = tmp_path / "runs" / "demo-6-showcase"
    phases = [{"name": "demo6", "cmd": "bash demos/demo-6-advisor-replay/run.sh", "exit_code": 0, "elapsed_s": 12.5}]
    summary = build_report.build_summary(tmp_path, output_dir, phases)
    markdown_path, html_path, summary_path = build_report.write_reports(summary, output_dir)

    html = html_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert payload["status"] == "pass"
    assert payload["reports_match"] is True
    assert "Demo 6 Advisor Replay Showcase" in html
    assert "No-LLM Run → Verify → Package → Offline Replay" in html
    assert "Original vs Replay" in html
    assert "How to view this report" in html
    assert "artifact_manifest.json" in html
    assert "verification_report.original.json" in html
    assert "verification_report.json" in html
    comparison_total = payload["comparison_total"]
    assert comparison_total == len(build_report._comparison_fields())
    assert f"Compared {comparison_total} verifier fields from verification_report.original.json and verification_report.json." in html
    assert f"{comparison_total} / {comparison_total} fields match" in html
    assert "reports_match=true" in html
    assert "exists=true" in html
    assert "Offline replay proof for advisor-facing reproducibility" not in html
    assert "backend and proxy environment variables are removed" not in html
    assert "reports match" not in html.lower()
    assert "drift detected" not in html.lower()
    assert "No comparison available" not in html
    assert "demo6 showcase healthy" in markdown


def test_demo6_visual_report_does_not_invent_empty_comparison_copy(tmp_path: Path) -> None:
    build_report = _load_build_report_module()

    output_dir = tmp_path / "runs" / "demo-6-showcase"
    summary = build_report.build_summary(tmp_path, output_dir, [])
    _, html_path, _ = build_report.write_reports(summary, output_dir)

    html = html_path.read_text(encoding="utf-8")

    assert "No comparison available" not in html
    assert "Offline replay proof" not in html
    assert "drift detected" not in html.lower()


def test_demo6_showcase_run_script_is_standalone_and_portable() -> None:
    script = (SHOWCASE_DIR / "run.sh").read_text(encoding="utf-8")
    assert 'if [[ -z "${PYTHON_BIN:-}" ]]; then' in script
    assert "refusing unsafe OUTPUT_DIR" in script
    assert "-s \"$PHASE_JSON\"" in script
    assert "PYTHONPATH" in script
    assert "demos/demo-6-advisor-replay/run.sh" in script
    assert "build_report.py" in script
    assert "demo-5" not in script
    assert "node" not in script.lower()


def test_demo6_readme_contains_copy_paste_agent_prompt() -> None:
    readme = DEMO_README.read_text(encoding="utf-8")
    assert "Copy-paste Agent Prompt" in readme
    assert "demos/demo-6-advisor-replay/showcase/run.sh" in readme
    assert "report.html" in readme
    assert "field-by-field verifier comparison" in readme.lower()
    assert "reports_match value" in readme
    assert "comparison_matches / comparison_total" in readme
    assert "offline replay proof" not in readme.lower()
    assert "/home/jinguxie/qec-ai-decoder/.venv/bin/python" not in readme
    assert "PYTHON_BIN=/home/jinguxie" not in readme
    assert "Use Python: ./.venv/bin/python" in readme
    assert "http://127.0.0.1" in readme
