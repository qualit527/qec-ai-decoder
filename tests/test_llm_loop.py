import json
from unittest.mock import MagicMock, patch

import pytest

from autoqec.orchestration.llm_loop import _env_yaml_path, run_llm_loop


def _stub_ideator(round_idx):
    return {
        "hypothesis": f"round {round_idx} idea",
        "fork_from": "baseline",
        "compose_mode": None,
        "rationale": "try a thing",
    }


def _stub_coder(round_idx):
    return {
        "dsl_config": {
            "type": "gnn", "output_mode": "soft_priors",
            "hidden_dim": 4, "n_layers": 1,
        },
        "tier": "1",
        "rationale": "small gnn",
        "commit_message": "feat(round): small gnn",
    }


def _stub_analyst(round_idx):
    return {
        "summary_1line": f"round {round_idx} ok",
        "verdict": "candidate",
        "next_hypothesis_seed": "try bigger",
    }


def test_env_yaml_path_falls_back_to_builtin_path():
    from autoqec.envs.schema import load_env_yaml
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = load_env_yaml(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")

    assert _env_yaml_path(env) == str(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")


def test_run_llm_loop_happy_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    import pathlib
    # Resolve the env yaml against the repo root (not tmp_path).
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = load_env_yaml(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")

    stub_metrics = MagicMock(
        status="ok", delta_ler=0.001,
        model_dump=MagicMock(return_value={"status": "ok", "delta_ler": 0.001}),
    )
    stub_report = MagicMock(
        verdict="SUSPICIOUS",
        model_dump=MagicMock(return_value={"verdict": "SUSPICIOUS",
                                           "delta_vs_baseline_holdout": None}),
    )
    captured_verify = {}

    responses = {"ideator": [], "coder": [], "analyst": []}

    def fake_invoke(role, prompt, timeout=300.0):
        responses[role].append(prompt)
        idx = len(responses[role])
        return {
            "ideator": _stub_ideator,
            "coder": _stub_coder,
            "analyst": _stub_analyst,
        }[role](idx)

    captured_configs = []

    def fake_run_round(cfg, env):
        captured_configs.append(cfg)
        return stub_metrics

    def fake_independent_verify(**kwargs):
        captured_verify.update(kwargs)
        return stub_report

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke), \
         patch("autoqec.orchestration.llm_loop.run_round", side_effect=fake_run_round), \
         patch("autoqec.orchestration.llm_loop.independent_verify", side_effect=fake_independent_verify):
        run_dir = run_llm_loop(
            env=env,
            rounds=2,
            profile="dev",
            env_yaml_path=repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml",
            invocation_argv=["python", "-m", "cli.autoqec", "run", "autoqec/envs/builtin/surface_d5_depol.yaml"],
        )

    assert (run_dir / "history.jsonl").exists()
    hist = (run_dir / "history.jsonl").read_text().strip().splitlines()
    assert len(hist) == 2
    assert len(responses["ideator"]) == 2
    assert captured_configs[0].env_yaml_path == str(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")
    assert captured_configs[0].invocation_argv == [
        "python", "-m", "cli.autoqec", "run", "autoqec/envs/builtin/surface_d5_depol.yaml",
    ]
    assert captured_verify["n_shots"] == 2048

    # Trace file captures the chat-level narrative (C route).
    trace = (run_dir / "orchestrator_trace.md")
    assert trace.exists()
    text = trace.read_text(encoding="utf-8")
    assert "# Orchestrator trace" in text
    assert "## Round 1" in text
    assert "## Round 2" in text
    # At least one of each subagent/runner/verifier section per round.
    assert text.count("ideator prompt") == 2
    assert text.count("ideator response") == 2
    assert text.count("coder response") == 2
    assert text.count("runner metrics") == 2
    assert text.count("analyst response") == 2
    assert "run complete" in text


def test_run_llm_loop_rejects_compose_rounds_until_p11(tmp_path, monkeypatch):
    """P0.1 ships without compose support; Ideator emitting list fork_from should error clearly."""
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = load_env_yaml(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")

    def fake_invoke(role, prompt, timeout=300.0):
        if role == "ideator":
            return {"hypothesis": "merge A and B", "fork_from": ["a", "b"],
                    "compose_mode": "pure", "rationale": ""}
        return {}

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke):
        with pytest.raises(NotImplementedError, match="compose"):
            run_llm_loop(env=env, rounds=1, profile="dev")


def test_run_llm_loop_resume_skips_legacy_completed_round_without_metrics_attempt_id(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = load_env_yaml(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")
    run_dir = tmp_path / "runs" / "resume-existing"
    round_1_dir = run_dir / "round_1"
    round_1_dir.mkdir(parents=True)
    (run_dir / "history.jsonl").write_text(
        json.dumps({"round": 1, "round_attempt_id": "legacy-attempt-1"}) + "\n",
        encoding="utf-8",
    )
    (round_1_dir / "metrics.json").write_text(
        json.dumps({"status": "ok", "delta_ler": 0.001}),
        encoding="utf-8",
    )

    stub_metrics = MagicMock(
        status="compile_error",
        model_dump=MagicMock(return_value={"status": "compile_error", "status_reason": "stub"}),
    )
    executed_rounds = []

    def fake_invoke(role, prompt, timeout=300.0):
        assert role in {"ideator", "coder"}
        return {
            "ideator": _stub_ideator,
            "coder": _stub_coder,
        }[role](2)

    def fake_run_round(cfg, env):
        executed_rounds.append(cfg.seed)
        return stub_metrics

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke), \
         patch("autoqec.orchestration.llm_loop.run_round", side_effect=fake_run_round):
        run_llm_loop(
            env=env,
            rounds=2,
            profile="dev",
            run_dir=run_dir,
            env_yaml_path=repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml",
            invocation_argv=["python", "-m", "cli.autoqec", "run", "autoqec/envs/builtin/surface_d5_depol.yaml"],
        )

    assert executed_rounds == [2]
    history_rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["round"] for row in history_rows] == [1, 2]
