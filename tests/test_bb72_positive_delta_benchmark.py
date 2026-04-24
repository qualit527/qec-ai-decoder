from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from autoqec.runner.schema import RoundMetrics


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "experiments/bb72-positive-delta/run.py"


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("bb72_positive_delta_run", RUNNER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_returns_explicit_paths(tmp_path: Path) -> None:
    module = _load_benchmark_module()
    env_yaml = tmp_path / "env.yaml"
    config_dir = tmp_path / "configs"
    output_root = tmp_path / "runs"

    args = module._parse_args(
        [
            "--env-yaml",
            str(env_yaml),
            "--config-dir",
            str(config_dir),
            "--output-root",
            str(output_root),
        ]
    )

    assert args.env_yaml == env_yaml
    assert args.config_dir == config_dir
    assert args.output_root == output_root


def test_script_help_runs_from_repo_root() -> None:
    completed = subprocess.run(
        [sys.executable, str(RUNNER_PATH), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--env-yaml" in completed.stdout


def test_run_benchmark_writes_positive_summary_and_report(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_benchmark_module()
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for idx in range(1, 4):
        (config_dir / f"round_{idx}.yaml").write_text(
            "type: gnn\n"
            "output_mode: soft_priors\n"
            "gnn:\n"
            "  layers: 1\n"
            "  hidden_dim: 4\n"
            "  message_fn: mlp\n"
            "  aggregation: sum\n"
            "  normalization: layer\n"
            "  residual: false\n"
            "  edge_features: [syndrome_bit]\n"
            "head: linear\n"
            "training:\n"
            "  learning_rate: 0.001\n"
            "  batch_size: 2\n"
            "  epochs: 1\n"
            "  loss: bce\n"
            "  profile: benchmark\n",
            encoding="utf-8",
        )
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("name: fake\n", encoding="utf-8")

    deltas = iter([-0.01, 0.0, 0.02])

    def fake_load_env_yaml(path):
        assert path == env_yaml
        return type("Env", (), {"name": "bb72_perf"})()

    def fake_run_round(config, env):
        assert config.training_profile == "benchmark"
        delta = next(deltas)
        return RoundMetrics(
            status="ok",
            ler_plain_classical=0.12,
            ler_predecoder=0.12 - delta,
            delta_ler=delta,
            flops_per_syndrome=1000 + int(config.seed),
            n_params=2000 + int(config.seed),
            train_wallclock_s=1.5,
            eval_wallclock_s=0.25,
            checkpoint_path=str(Path(config.round_dir) / "checkpoint.pt"),
            training_log_path=str(Path(config.round_dir) / "train.log"),
        )

    monkeypatch.setattr(module, "load_env_yaml", fake_load_env_yaml)
    monkeypatch.setattr(module, "run_round", fake_run_round)
    monkeypatch.setattr(module, "_run_id", lambda: "test-run")

    run_dir = module.run_benchmark(
        env_yaml=env_yaml,
        config_dir=config_dir,
        output_root=tmp_path / "runs",
    )

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert summary["run_id"] == "test-run"
    assert summary["has_positive_delta"] is True
    assert summary["best_delta_ler"] == 0.02
    assert summary["best_round"] == 3
    assert [row["best_delta_ler_so_far"] for row in summary["rounds"]] == [
        -0.01,
        0.0,
        0.02,
    ]
    assert "benchmark evidence" in report
    assert "not a VERIFIED holdout claim" in report
