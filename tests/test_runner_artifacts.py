from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from autoqec.envs.schema import EnvSpec
from autoqec.runner.runner import run_round
from autoqec.runner.safety import RunnerSafety
from autoqec.runner.schema import RoundMetrics, RunnerConfig
from tests.fixture_utils import load_json_fixture


ARTIFACT_MANIFEST_CONTRACTS = load_json_fixture("public_api", "artifact_manifest_output_contracts.json")
ARTIFACT_MANIFEST_INVALID_CASES = load_json_fixture("public_api", "artifact_manifest_invalid_cases.json")


def _toy_env() -> EnvSpec:
    return EnvSpec(
        name="toy_osd",
        code={"type": "parity_check_matrix", "source": "toy.npy"},
        noise={
            "type": "depolarizing",
            "p": [1e-3],
            "seed_policy": {"train": [1, 4], "val": [5, 8], "holdout": [9, 12]},
        },
        constraints={
            "latency_flops_budget": 10_000,
            "param_budget": 10_000,
            "target_ler": 1e-4,
            "target_p": 1e-3,
        },
        baseline_decoders=["bposd"],
        classical_backend="osd",
        eval_protocol={
            "min_shots_train": 4,
            "min_shots_val": 2,
            "min_shots_verify": 2,
            "bootstrap_ci": 0.95,
            "osd_orders_reported": [0],
            "x_z_decoding": "x_only",
        },
    )


def _toy_cfg(round_dir: Path) -> RunnerConfig:
    return RunnerConfig(
        env_name="toy_osd",
        predecoder_config={
            "type": "gnn",
            "output_mode": "soft_priors",
            "gnn": {
                "layers": 1,
                "hidden_dim": 8,
                "message_fn": "mlp",
                "aggregation": "sum",
                "normalization": "none",
                "residual": False,
                "edge_features": [],
            },
            "head": "linear",
            "training": {
                "learning_rate": 1e-3,
                "batch_size": 2,
                "epochs": 1,
                "loss": "bce",
                "profile": "dev",
            },
        },
        training_profile="dev",
        seed=0,
        round_dir=str(round_dir),
        env_yaml_path="autoqec/envs/builtin/surface_d5_depol.yaml",
        invocation_argv=["python", "-m", "cli.autoqec", "run", "toy.yaml", "--no-llm"],
    )


def _toy_artifacts() -> SimpleNamespace:
    return SimpleNamespace(
        code_type="parity_check_matrix",
        code_artifact=np.eye(2, dtype=np.uint8),
        edge_index=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        n_var=2,
        n_check=2,
        prior_p=torch.full((2,), 0.001, dtype=torch.float32),
        parity_check_matrix=np.eye(2, dtype=np.uint8),
    )


class _TinyModel(torch.nn.Module):
    output_mode = "soft_priors"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, syndrome: torch.Tensor, ctx: dict) -> torch.Tensor:
        del ctx
        return torch.sigmoid(self.linear(syndrome.float()))


def _valid_manifest_payload() -> dict[str, object]:
    return load_json_fixture("public_api", "artifact_manifest_valid_payload.json")


def _apply_patch(payload: dict[str, object], patch: dict[str, object]) -> dict[str, object]:
    patched = deepcopy(payload)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(patched.get(key), dict):
            patched[key] = _apply_patch(patched[key], value)
        else:
            patched[key] = value
    return patched


def _materialize_manifest_artifacts(round_dir: Path) -> None:
    (round_dir / "config.yaml").write_text("type: gnn\n", encoding="utf-8")
    (round_dir / "checkpoint.pt").write_text("stub", encoding="utf-8")
    (round_dir / "metrics.json").write_text('{"status":"ok"}', encoding="utf-8")
    (round_dir / "train.log").write_text("0\t0.1\n", encoding="utf-8")


def test_run_round_success_writes_consumable_artifacts(monkeypatch, tmp_path: Path) -> None:
    from autoqec.runner import runner
    contract = ARTIFACT_MANIFEST_CONTRACTS["runner_round_success"]

    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _toy_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: _TinyModel())
    monkeypatch.setattr(
        runner,
        "sample_syndromes",
        lambda *_args, **_kwargs: (
            torch.zeros((4, 2), dtype=torch.float32),
            torch.zeros((4, 2), dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        runner,
        "decode_with_predecoder",
        lambda preds, *_args, **_kwargs: np.zeros((len(preds), 2), dtype=np.int64),
    )
    monkeypatch.setattr(runner, "estimate_flops", lambda *_args, **_kwargs: 123)

    round_dir = tmp_path / "round_success"
    cfg = _toy_cfg(round_dir)
    metrics = run_round(cfg, _toy_env(), safety=RunnerSafety(VRAM_PRE_CHECK=False))

    metrics_path = round_dir / "metrics.json"
    checkpoint_path = round_dir / "checkpoint.pt"
    train_log_path = round_dir / "train.log"
    config_path = round_dir / "config.yaml"
    manifest_path = round_dir / "artifact_manifest.json"

    assert metrics.status == "ok"
    assert config_path.exists()
    assert train_log_path.exists()
    assert checkpoint_path.exists()
    assert metrics_path.exists()
    assert manifest_path.exists()

    parsed = RoundMetrics.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    assert parsed.status == "ok"
    assert parsed.checkpoint_path == str(checkpoint_path.resolve())
    assert parsed.training_log_path == str(train_log_path.resolve())
    assert Path(parsed.checkpoint_path).exists()
    assert Path(parsed.training_log_path).exists()

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == cfg.predecoder_config
    assert train_log_path.read_text(encoding="utf-8").strip()
    checkpoint = torch.load(checkpoint_path)
    assert checkpoint["dsl_config"] == cfg.predecoder_config

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    for key in contract["required_repo_truthy_fields"]:
        assert manifest["repo"][key]
    for key in contract["required_repo_present_fields"]:
        assert key in manifest["repo"]
    assert manifest["environment"]["env_yaml_path"] == contract["expected_environment"]["env_yaml_path"]
    for key in contract["required_environment_truthy_fields"]:
        assert manifest["environment"][key]
    assert manifest["round"]["round_dir"] == contract["expected_round"]["round_dir"]
    assert manifest["round"]["round"] == contract["expected_round"]["round"]
    assert manifest["round"]["command_line"] == contract["expected_round"]["command_line"]
    for key in contract["required_round_truthy_fields"]:
        assert manifest["round"][key]
    assert manifest["artifacts"] == contract["expected_artifacts"]
    for key in contract["required_package_truthy_fields"]:
        assert manifest["packages"][key]
    for key in contract["required_package_present_fields"]:
        assert key in manifest["packages"]


def test_git_output_returns_none_when_git_command_fails(monkeypatch, tmp_path: Path) -> None:
    from autoqec.runner import artifact_manifest

    def fake_check_output(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr(artifact_manifest.subprocess, "check_output", fake_check_output)

    assert artifact_manifest._git_output(tmp_path, "rev-parse", "HEAD") is None


@pytest.mark.parametrize(
    "case",
    ARTIFACT_MANIFEST_INVALID_CASES,
    ids=[case["name"] for case in ARTIFACT_MANIFEST_INVALID_CASES],
)
def test_validate_artifact_manifest_rejects_invalid_payloads(tmp_path: Path, case: dict[str, object]) -> None:
    from autoqec.runner.artifact_manifest import validate_artifact_manifest

    round_dir = tmp_path / "round_1"
    round_dir.mkdir()
    _materialize_manifest_artifacts(round_dir)
    payload = _apply_patch(_valid_manifest_payload(), case["patch"])

    with pytest.raises(ValueError, match=case["expected_error"]):
        validate_artifact_manifest(round_dir, payload)


def test_run_round_compile_error_emits_metrics_without_claiming_missing_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from autoqec.runner import runner

    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _toy_artifacts())

    def _boom(*_args, **_kwargs):
        raise ValueError("bad decoder config")

    monkeypatch.setattr(runner, "compile_predecoder", _boom)

    round_dir = tmp_path / "round_compile_error"
    metrics = run_round(_toy_cfg(round_dir), _toy_env(), safety=RunnerSafety(VRAM_PRE_CHECK=False))

    metrics_path = round_dir / "metrics.json"
    assert metrics.status == "compile_error"
    assert "bad decoder config" in (metrics.status_reason or "")
    assert (round_dir / "config.yaml").exists()
    assert metrics_path.exists()
    assert not (round_dir / "train.log").exists()
    assert not (round_dir / "checkpoint.pt").exists()

    parsed = RoundMetrics.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    assert parsed.status == "compile_error"
    assert parsed.checkpoint_path is None
    assert parsed.training_log_path is None
    assert not (round_dir / "artifact_manifest.json").exists()


def test_run_round_safety_kill_emits_metrics_without_claiming_missing_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from autoqec.runner import runner

    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _toy_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: _TinyModel())
    monkeypatch.setattr(
        runner,
        "sample_syndromes",
        lambda *_args, **_kwargs: (
            torch.zeros((4, 2), dtype=torch.float32),
            torch.zeros((4, 2), dtype=torch.int64),
        ),
    )

    times = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(times))

    round_dir = tmp_path / "round_killed"
    metrics = run_round(
        _toy_cfg(round_dir),
        _toy_env(),
        safety=RunnerSafety(WALL_CLOCK_HARD_CUTOFF_S=0, VRAM_PRE_CHECK=False),
    )

    metrics_path = round_dir / "metrics.json"
    assert metrics.status == "killed_by_safety"
    assert metrics.n_params is not None
    assert metrics.train_wallclock_s > 0
    assert metrics_path.exists()
    assert not (round_dir / "train.log").exists()
    assert not (round_dir / "checkpoint.pt").exists()

    parsed = RoundMetrics.model_validate_json(metrics_path.read_text(encoding="utf-8"))
    assert parsed.status == "killed_by_safety"
    assert parsed.checkpoint_path is None
    assert parsed.training_log_path is None
    assert not (round_dir / "artifact_manifest.json").exists()
