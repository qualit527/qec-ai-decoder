import json
from unittest.mock import MagicMock, patch

import pytest

from autoqec.orchestration.llm_loop import _env_yaml_path, run_llm_loop
from autoqec.runner.schema import RoundMetrics


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

    stub_metrics = RoundMetrics(
        status="ok",
        delta_ler=0.001,
        flops_per_syndrome=111,
        n_params=222,
        round_attempt_id="attempt-1",
        branch="exp/test/01-a",
        commit_sha="deadbeef",
        fork_from="baseline",
    )
    stub_report = MagicMock(
        verdict="SUSPICIOUS",
        model_dump=MagicMock(return_value={"verdict": "SUSPICIOUS",
                                           "delta_vs_baseline_holdout": None}),
    )

    responses = {"ideator": [], "coder": [], "analyst": []}

    def fake_invoke(role, prompt, timeout=300.0, cwd=None):
        responses[role].append(prompt)
        idx = len(responses[role])
        payload = {
            "ideator": _stub_ideator,
            "coder": _stub_coder,
            "analyst": _stub_analyst,
        }[role](idx)
        return payload, {"usage": {"input_tokens": 1, "output_tokens": 1}}

    captured_configs = []
    created_worktrees = []

    def fake_create_round_worktree(repo_root, run_id, round_idx, slug, fork_from):
        worktree_dir = tmp_path / ".worktrees" / f"exp-{run_id}-{round_idx:02d}-{slug}"
        worktree_dir.mkdir(parents=True, exist_ok=True)
        created_worktrees.append(
            {
                "worktree_dir": str(worktree_dir),
                "branch": f"exp/{run_id}/{round_idx:02d}-{slug}",
                "fork_from": fork_from,
            }
        )
        return created_worktrees[-1]

    def fake_run_round(cfg, env, round_attempt_id=None):
        captured_configs.append(cfg)
        return stub_metrics.model_copy(
            update={
                "round_attempt_id": round_attempt_id,
                "branch": cfg.branch,
                "fork_from": cfg.fork_from,
            }
        )

    with patch("autoqec.orchestration.llm_loop.invoke_subagent_with_metadata", side_effect=fake_invoke), \
         patch("autoqec.orchestration.llm_loop.create_round_worktree", side_effect=fake_create_round_worktree), \
         patch("autoqec.orchestration.llm_loop.cleanup_round_worktree"), \
         patch("autoqec.orchestration.llm_loop.run_round_in_subprocess", side_effect=fake_run_round), \
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
    assert created_worktrees[0]["fork_from"] == "HEAD"
    assert captured_configs[0].env_yaml_path == str(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")
    assert captured_configs[0].invocation_argv == [
        "python", "-m", "cli.autoqec", "run", "autoqec/envs/builtin/surface_d5_depol.yaml",
    ]

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

    def fake_invoke(role, prompt, timeout=300.0, cwd=None):
        if role == "ideator":
            return (
                {
                    "hypothesis": "merge A and B",
                    "fork_from": ["a", "b"],
                    "compose_mode": "pure",
                    "rationale": "",
                },
                {"usage": {"input_tokens": 1, "output_tokens": 1}},
            )
        return {}, {"usage": {"input_tokens": 1, "output_tokens": 1}}

    with patch("autoqec.orchestration.llm_loop.invoke_subagent_with_metadata", side_effect=fake_invoke):
        with pytest.raises(NotImplementedError, match="compose"):
            run_llm_loop(env=env, rounds=1, profile="dev")


def test_run_llm_loop_uses_worktree_rounds_and_persists_live_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = load_env_yaml(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")

    created = []
    cleaned = []
    runner_cfgs = []
    seen_prompts = {"ideator": [], "coder": [], "analyst": []}
    seen_cwds = {"ideator": [], "coder": [], "analyst": []}

    def fake_invoke(role, prompt, timeout=300.0, cwd=None):
        seen_prompts[role].append(prompt)
        seen_cwds[role].append(cwd)
        payload = {
            "ideator": {
                "hypothesis": "round 1 idea",
                "fork_from": "baseline",
                "compose_mode": None,
                "rationale": "try a thing",
            },
            "coder": {
                "dsl_config": {
                    "type": "gnn",
                    "output_mode": "soft_priors",
                    "hidden_dim": 4,
                    "n_layers": 1,
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 1e-3,
                    },
                },
                "tier": "1",
                "rationale": "small gnn",
                "commit_message": "feat(round): small gnn",
            },
            "analyst": {
                "summary_1line": "candidate survived smoke",
                "verdict": "candidate",
                "next_hypothesis_seed": "try bigger",
            },
        }[role]
        meta = {
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "duration_ms": 123,
            "backend": "claude-cli",
            "model": "claude-haiku-4-5",
        }
        return payload, meta

    def fake_create_round_worktree(repo_root, run_id, round_idx, slug, fork_from):
        worktree_dir = tmp_path / ".worktrees" / f"exp-{run_id}-{round_idx:02d}-{slug}"
        worktree_dir.mkdir(parents=True, exist_ok=True)
        created.append(
            {
                "repo_root": str(repo_root),
                "run_id": run_id,
                "round_idx": round_idx,
                "slug": slug,
                "fork_from": fork_from,
                "worktree_dir": str(worktree_dir),
                "branch": f"exp/{run_id}/{round_idx:02d}-{slug}",
            }
        )
        return created[-1]

    def fake_cleanup_round_worktree(repo_root, worktree_dir):
        cleaned.append({"repo_root": str(repo_root), "worktree_dir": worktree_dir})

    def fake_run_round_in_subprocess(cfg, env, round_attempt_id=None):
        runner_cfgs.append(cfg)
        return RoundMetrics(
            status="ok",
            delta_ler=0.001,
            flops_per_syndrome=1234,
            n_params=4321,
            train_wallclock_s=1.25,
            checkpoint_path=str(pathlib.Path(cfg.round_dir) / "checkpoint.pt"),
            training_log_path=str(pathlib.Path(cfg.round_dir) / "train.log"),
            round_attempt_id=round_attempt_id,
            branch=cfg.branch,
            commit_sha="deadbeef",
            fork_from=cfg.fork_from,
        )

    stub_report = MagicMock(
        verdict="SUSPICIOUS",
        model_dump=MagicMock(
            return_value={
                "verdict": "SUSPICIOUS",
                "delta_vs_baseline_holdout": None,
            }
        ),
    )

    with patch(
        "autoqec.orchestration.llm_loop.invoke_subagent_with_metadata",
        side_effect=fake_invoke,
    ), patch(
        "autoqec.orchestration.llm_loop.create_round_worktree",
        side_effect=fake_create_round_worktree,
    ), patch(
        "autoqec.orchestration.llm_loop.cleanup_round_worktree",
        side_effect=fake_cleanup_round_worktree,
    ), patch(
        "autoqec.orchestration.llm_loop.run_round_in_subprocess",
        side_effect=fake_run_round_in_subprocess,
    ), patch(
        "autoqec.orchestration.llm_loop.independent_verify",
        return_value=stub_report,
    ):
        run_dir = run_llm_loop(
            env=env,
            rounds=1,
            profile="dev",
            env_yaml_path=repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml",
            invocation_argv=[
                "python",
                "-m",
                "cli.autoqec",
                "run",
                "autoqec/envs/builtin/surface_d5_depol.yaml",
            ],
        )

    assert created[0]["fork_from"] == "HEAD"
    assert runner_cfgs[0].code_cwd == created[0]["worktree_dir"]
    assert runner_cfgs[0].branch == created[0]["branch"]
    assert cleaned[0]["worktree_dir"] == created[0]["worktree_dir"]
    assert created[0]["worktree_dir"] in seen_prompts["coder"][0]
    assert seen_cwds["coder"][0] == created[0]["worktree_dir"]

    ideator_payload = json.loads((run_dir / "round_1" / "ideator_prompt.json").read_text())
    assert "fork_graph" in ideator_payload
    assert "machine_state_hint" in ideator_payload
    assert "last_5_hypotheses" not in ideator_payload

    ideator_resp = json.loads((run_dir / "round_1" / "ideator_response.json").read_text())
    assert ideator_resp["payload"]["fork_from"] == "baseline"
    assert ideator_resp["meta"]["usage"]["input_tokens"] == 11

    log_text = (run_dir / "log.md").read_text(encoding="utf-8")
    assert '"tool_call": "machine_state"' in log_text
    trace_text = (run_dir / "orchestrator_trace.md").read_text(encoding="utf-8")
    assert "coder response" in trace_text
    assert "runner metrics" in trace_text


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

    executed_rounds = []

    def fake_invoke(role, prompt, timeout=300.0, cwd=None):
        assert role in {"ideator", "coder"}
        payload = {
            "ideator": _stub_ideator,
            "coder": _stub_coder,
        }[role](2)
        return payload, {"usage": {"input_tokens": 1, "output_tokens": 1}}

    def fake_create_round_worktree(repo_root, run_id, round_idx, slug, fork_from):
        worktree_dir = tmp_path / ".worktrees" / f"exp-{run_id}-{round_idx:02d}-{slug}"
        worktree_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_id": run_id,
            "round_idx": round_idx,
            "slug": slug,
            "fork_from": fork_from,
            "worktree_dir": str(worktree_dir),
            "branch": f"exp/{run_id}/{round_idx:02d}-{slug}",
        }

    def fake_run_round(cfg, env, round_attempt_id=None):
        executed_rounds.append(cfg.seed)
        return RoundMetrics(
            status="compile_error",
            status_reason="stub",
            round_attempt_id=round_attempt_id,
            branch=cfg.branch,
            commit_sha="deadbeef-resume",
            fork_from=cfg.fork_from,
        )

    with patch(
        "autoqec.orchestration.llm_loop.invoke_subagent_with_metadata",
        side_effect=fake_invoke,
    ), patch(
        "autoqec.orchestration.llm_loop.create_round_worktree",
        side_effect=fake_create_round_worktree,
    ), patch(
        "autoqec.orchestration.llm_loop.cleanup_round_worktree",
    ), patch(
        "autoqec.orchestration.llm_loop.run_round_in_subprocess",
        side_effect=fake_run_round,
    ):
        run_llm_loop(
            env=env,
            rounds=2,
            profile="dev",
            run_dir=run_dir,
            env_yaml_path=repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml",
            invocation_argv=[
                "python",
                "-m",
                "cli.autoqec",
                "run",
                "autoqec/envs/builtin/surface_d5_depol.yaml",
            ],
        )

    assert executed_rounds == [2]
    history_rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["round"] for row in history_rows] == [1, 2]

def test_run_llm_loop_warns_when_all_rounds_stay_on_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from autoqec.envs.schema import load_env_yaml
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    env = load_env_yaml(repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml")

    def fake_invoke(role, prompt, timeout=300.0, cwd=None):
        payload = {
            "ideator": {
                "hypothesis": "baseline only",
                "fork_from": "baseline",
                "compose_mode": None,
                "rationale": "stay conservative",
            },
            "coder": {
                "dsl_config": {
                    "type": "gnn",
                    "output_mode": "soft_priors",
                    "hidden_dim": 4,
                    "n_layers": 1,
                    "training": {
                        "epochs": 1,
                        "batch_size": 8,
                        "learning_rate": 1e-3,
                    },
                },
                "tier": "1",
                "rationale": "small gnn",
                "commit_message": "feat(round): small gnn",
            },
            "analyst": {
                "summary_1line": "candidate survived smoke",
                "verdict": "candidate",
                "next_hypothesis_seed": "try bigger",
            },
        }[role]
        return payload, {"usage": {"input_tokens": 1, "output_tokens": 1}}

    def fake_create_round_worktree(repo_root, run_id, round_idx, slug, fork_from):
        worktree_dir = tmp_path / ".worktrees" / f"exp-{run_id}-{round_idx:02d}-{slug}"
        worktree_dir.mkdir(parents=True, exist_ok=True)
        return {
            "run_id": run_id,
            "round_idx": round_idx,
            "slug": slug,
            "fork_from": fork_from,
            "worktree_dir": str(worktree_dir),
            "branch": f"exp/{run_id}/{round_idx:02d}-{slug}",
        }

    def fake_run_round_in_subprocess(cfg, env, round_attempt_id=None):
        return RoundMetrics(
            status="ok",
            delta_ler=0.001,
            flops_per_syndrome=10,
            n_params=20,
            round_attempt_id=round_attempt_id,
            branch=cfg.branch,
            commit_sha=f"deadbeef-{cfg.branch}",
            fork_from=cfg.fork_from,
        )

    stub_report = MagicMock(
        verdict="SUSPICIOUS",
        model_dump=MagicMock(return_value={"verdict": "SUSPICIOUS", "delta_vs_baseline_holdout": None}),
    )

    with patch(
        "autoqec.orchestration.llm_loop.invoke_subagent_with_metadata",
        side_effect=fake_invoke,
    ), patch(
        "autoqec.orchestration.llm_loop.create_round_worktree",
        side_effect=fake_create_round_worktree,
    ), patch(
        "autoqec.orchestration.llm_loop.cleanup_round_worktree",
    ), patch(
        "autoqec.orchestration.llm_loop.run_round_in_subprocess",
        side_effect=fake_run_round_in_subprocess,
    ), patch(
        "autoqec.orchestration.llm_loop.independent_verify",
        return_value=stub_report,
    ):
        run_dir = run_llm_loop(env=env, rounds=2, profile="dev")

    log_text = (run_dir / "log.md").read_text(encoding="utf-8")
    assert '"warning": "all_rounds_baseline_fork_from"' in log_text
