from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_mode_defaults_cover_fast_dev_and_prod() -> None:
    from scripts.run_bb72_demo import ModeConfig, _mode_defaults

    assert _mode_defaults("fast") == ModeConfig(rounds=1, profile="dev")
    assert _mode_defaults("dev") == ModeConfig(rounds=3, profile="dev")
    assert _mode_defaults("prod") == ModeConfig(rounds=10, profile="prod")


def test_discover_python_bin_walks_up_to_shared_venv(tmp_path: Path) -> None:
    from scripts.run_bb72_demo import _discover_python_bin

    repo_root = tmp_path / "repo"
    worktree_root = repo_root / ".worktrees" / "issue-37"
    shared_python = repo_root / ".venv" / "bin" / "python"
    shared_python.parent.mkdir(parents=True)
    shared_python.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    shared_python.chmod(0o755)
    worktree_root.mkdir(parents=True)

    assert _discover_python_bin(worktree_root) == str(shared_python)


def test_main_runs_no_llm_smoke_and_reports_candidate_pareto(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    from scripts import run_bb72_demo

    env_yaml = tmp_path / "bb72.yaml"
    env_yaml.write_text("name: bb72\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / "20260424-010000"
    run_dir.mkdir(parents=True)
    (run_dir / "history.jsonl").write_text('{"round": 1}\n', encoding="utf-8")
    (run_dir / "candidate_pareto.json").write_text('[{"round": 1, "verified": false}]', encoding="utf-8")

    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_bb72_demo, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(run_bb72_demo, "DEFAULT_ENV_YAML", env_yaml)
    monkeypatch.setattr(
        run_bb72_demo,
        "_parse_args",
        lambda argv=None: SimpleNamespace(
            env_yaml=str(env_yaml),
            mode="fast",
            rounds=None,
            profile=None,
            python_bin=None,
        ),
    )
    monkeypatch.setattr(run_bb72_demo, "_discover_python_bin", lambda start_dir, explicit=None: "/shared/.venv/bin/python")
    monkeypatch.setattr(run_bb72_demo.subprocess, "run", fake_run)

    assert run_bb72_demo.main() == 0

    out = capsys.readouterr().out
    assert "Demo 2 (bb72 qLDPC) complete" in out
    assert "Candidate Pareto:" in out
    assert "verified" in out

    argv = calls[0][0]
    assert argv[:4] == ["/shared/.venv/bin/python", "-m", "cli.autoqec", "run"]
    assert str(env_yaml) in argv
    assert "--rounds" in argv and "1" in argv
    assert "--profile" in argv and "dev" in argv
    assert "--no-llm" in argv
