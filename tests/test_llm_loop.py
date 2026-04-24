from unittest.mock import patch, MagicMock

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

    with patch("autoqec.orchestration.llm_loop.invoke_subagent", side_effect=fake_invoke), \
         patch("autoqec.orchestration.llm_loop.run_round", side_effect=fake_run_round), \
         patch("autoqec.orchestration.llm_loop.independent_verify", return_value=stub_report):
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
