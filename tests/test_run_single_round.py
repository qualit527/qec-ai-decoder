"""Unit tests for scripts.run_single_round (Task A2.2)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_plan_for_round_returns_round_dir_and_prompt(tmp_path: Path) -> None:
    from scripts.run_single_round import plan_for_round

    plan = plan_for_round(
        env_yaml="autoqec/envs/builtin/surface_d5_depol.yaml",
        run_dir=tmp_path / "run",
        round_idx=3,
        budget_s=3600,
    )
    assert plan["round_idx"] == 3
    assert plan["round_dir"].endswith("round_3")
    assert "IDEATOR" in plan["ideator_prompt"]
    # machine_state's budget should reach the prompt
    assert "budget" in plan["ideator_prompt"]


def test_run_single_round_cli_prints_valid_json(tmp_path: Path) -> None:
    """End-to-end shell-level contract: prints a JSON object to stdout."""
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_single_round.py",
            "--env-yaml",
            "autoqec/envs/builtin/surface_d5_depol.yaml",
            "--run-dir",
            str(tmp_path / "run"),
            "--round-idx",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    payload = json.loads(completed.stdout)
    assert payload["round_idx"] == 1
    assert payload["round_dir"].endswith("round_1")
    assert "IDEATOR" in payload["ideator_prompt"]


def test_kb_and_spec_excerpts_reach_ideator_prompt_from_foreign_cwd(tmp_path: Path) -> None:
    """Codex review (medium): KB/SPEC paths must be anchored to the repo root,
    not the caller's cwd. Without the fix, running from outside the repo
    silently yields an empty knowledge excerpt."""
    repo_root = Path(__file__).resolve().parents[1]
    env_abs = str(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/run_single_round.py"),
            "--env-yaml",
            env_abs,
            "--run-dir",
            str(tmp_path / "run"),
            "--round-idx",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        cwd=tmp_path,  # foreign cwd
    )
    payload = json.loads(completed.stdout)
    # DECODER_ROADMAP snippet should now be embedded in the Ideator prompt.
    # Signature: the roadmap has a top-level "Decoder" heading — any substring from it works.
    assert "knowledge_excerpts" in payload["ideator_prompt"]
    # and the spec dsl excerpt: the frozen spec title contains "AutoQEC"
    assert "AutoQEC" in payload["ideator_prompt"]
