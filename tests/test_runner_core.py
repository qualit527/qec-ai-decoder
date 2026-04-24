from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from autoqec.envs.schema import load_env_yaml
from autoqec.runner import runner
from autoqec.runner.data import CodeArtifacts, SampleBatch
from autoqec.runner.safety import RunnerSafety
from autoqec.runner.schema import RunnerConfig


class FakeModel(nn.Module):
    def __init__(self, output_mode: str, output_width: int):
        super().__init__()
        self.output_mode = output_mode
        self.output_width = output_width
        self.weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, syndrome: torch.Tensor, ctx: dict) -> torch.Tensor:
        batch = syndrome.shape[0]
        if self.output_mode == "soft_priors":
            return torch.sigmoid(self.weight).expand(batch, self.output_width)
        return self.weight.expand(batch, self.output_width)


class FakeNaNModel(nn.Module):
    def __init__(self, output_mode: str, output_width: int):
        super().__init__()
        self.output_mode = output_mode
        self.output_width = output_width
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, syndrome: torch.Tensor, ctx: dict) -> torch.Tensor:
        batch = syndrome.shape[0]
        return torch.full((batch, self.output_width), float("nan"), device=syndrome.device)


def _parity_artifacts(n_var: int = 5, n_check: int = 3) -> CodeArtifacts:
    parity = np.zeros((n_check, n_var), dtype=np.uint8)
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    return CodeArtifacts(
        code_type="parity_check_matrix",
        code_artifact=parity,
        edge_index=edge_index,
        n_var=n_var,
        n_check=n_check,
        prior_p=torch.full((n_var,), 0.05, dtype=torch.float32),
        parity_check_matrix=parity,
    )


def _stim_artifacts(n_var: int = 5, n_check: int = 3) -> CodeArtifacts:
    circuit = object()
    edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    return CodeArtifacts(
        code_type="stim_circuit",
        code_artifact=circuit,
        edge_index=edge_index,
        n_var=n_var,
        n_check=n_check,
        prior_p=torch.full((n_var,), 0.05, dtype=torch.float32),
    )


def test_profile_params_and_failure_rate_cover_both_modes() -> None:
    stim_env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    parity_env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")

    assert runner._profile_params(stim_env, "dev") == {
        "n_shots_train": 4096,
        "n_shots_val": 256,
        "epochs_cap": 3,
    }
    assert runner._profile_params(stim_env, "prod") == {
        "n_shots_train": 16384,
        "n_shots_val": 1024,
        "epochs_cap": 10,
    }

    preds = np.array([[0, 1], [1, 1]])
    targets = np.array([[0, 1], [0, 1]])
    assert runner._failure_rate(stim_env, preds, targets) == 0.5
    assert runner._failure_rate(parity_env, preds, targets) == 0.5


def test_set_seed_calls_cpu_and_cuda_seeders(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(runner.random, "seed", lambda seed: called.append(("random", seed)))
    monkeypatch.setattr(runner.np.random, "seed", lambda seed: called.append(("numpy", seed)))
    monkeypatch.setattr(runner.torch, "manual_seed", lambda seed: called.append(("torch", seed)))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        runner.torch.cuda,
        "manual_seed_all",
        lambda seed: called.append(("cuda", seed)),
    )

    runner._set_seed(7)

    assert called == [
        ("random", 7),
        ("numpy", 7),
        ("torch", 7),
        ("cuda", 7),
    ]


def test_run_round_returns_compile_error_when_model_build_fails(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={"training": {"learning_rate": 1e-3, "batch_size": 1, "epochs": 1}},
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )

    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _stim_artifacts())
    monkeypatch.setattr(
        runner,
        "compile_predecoder",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    metrics = runner.run_round(cfg, env)
    assert metrics.status == "compile_error"
    assert metrics.status_reason == "boom"


def test_run_round_returns_vram_precheck_kill(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "gnn": {"hidden_dim": 128},
            "training": {"learning_rate": 1e-3, "batch_size": 4, "epochs": 1},
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )
    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _parity_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: FakeModel("soft_priors", 5))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner, "estimate_vram_gb", lambda *_args, **_kwargs: 10.0)
    monkeypatch.setattr(runner.torch.cuda, "mem_get_info", lambda: (1_000_000, 1_000_000))

    metrics = runner.run_round(cfg, env, safety=RunnerSafety(VRAM_PRE_CHECK=True))
    assert metrics.status == "killed_by_safety"
    assert "VRAM estimate" in (metrics.status_reason or "")


def test_run_round_returns_wall_clock_cutoff(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "soft_priors",
            "training": {"learning_rate": 1e-3, "batch_size": 1, "epochs": 1},
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )
    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _parity_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: FakeModel("soft_priors", 5))
    monkeypatch.setattr(
        runner,
        "sample_syndromes",
        lambda *_args, **_kwargs: SampleBatch(
            syndrome=torch.zeros((2, 3), dtype=torch.float32),
            errors=torch.zeros((2, 5), dtype=torch.int64),
            observables=torch.zeros((2, 5), dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    time_points = iter([0.0, 1.0, 1.0])
    monkeypatch.setattr(runner.time, "time", lambda: next(time_points))

    metrics = runner.run_round(
        cfg,
        env,
        safety=RunnerSafety(WALL_CLOCK_HARD_CUTOFF_S=0, VRAM_PRE_CHECK=False),
    )
    assert metrics.status == "killed_by_safety"
    assert "wall_clock_cutoff" in (metrics.status_reason or "")


def test_run_round_returns_nan_rate_kill(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "hard_flip",
            "training": {"learning_rate": 1e-3, "batch_size": 1, "epochs": 1},
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )
    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _parity_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: FakeNaNModel("hard_flip", 3))
    monkeypatch.setattr(
        runner,
        "sample_syndromes",
        lambda *_args, **_kwargs: SampleBatch(
            syndrome=torch.zeros((2, 3), dtype=torch.float32),
            errors=torch.zeros((2, 5), dtype=torch.int64),
            observables=torch.zeros((2, 5), dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    metrics = runner.run_round(
        cfg,
        env,
        safety=RunnerSafety(VRAM_PRE_CHECK=False, MAX_NAN_RATE=0.0),
    )
    assert metrics.status == "killed_by_safety"
    assert "NaN rate" in (metrics.status_reason or "")


def test_run_round_success_soft_priors_parity_path(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/bb72_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "soft_priors",
            "training": {"learning_rate": 1e-3, "batch_size": 2, "epochs": 1},
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )
    train_batch = SampleBatch(
        syndrome=torch.zeros((2, 3), dtype=torch.float32),
        errors=torch.zeros((2, 5), dtype=torch.int64),
        observables=torch.zeros((2, 5), dtype=torch.int64),
    )
    val_batch = SampleBatch(
        syndrome=torch.zeros((2, 3), dtype=torch.float32),
        errors=torch.zeros((2, 5), dtype=torch.int64),
        observables=torch.zeros((2, 5), dtype=torch.int64),
    )
    calls = iter([train_batch, val_batch])

    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _parity_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: FakeModel("soft_priors", 5))
    monkeypatch.setattr(runner, "sample_syndromes", lambda *_args, **_kwargs: next(calls))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "decode_with_predecoder", lambda *_args, **_kwargs: np.zeros((2, 5), dtype=np.int64))
    monkeypatch.setattr(runner, "estimate_flops", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no flop")))

    metrics = runner.run_round(cfg, env, safety=RunnerSafety(VRAM_PRE_CHECK=False))
    round_dir = Path(cfg.round_dir)
    assert metrics.status == "ok"
    assert metrics.flops_per_syndrome == 2
    assert round_dir.joinpath("metrics.json").exists()
    assert round_dir.joinpath("checkpoint.pt").exists()
    assert round_dir.joinpath("train.log").exists()


def test_run_round_success_hard_flip_mwpm_path(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "hard_flip",
            "training": {"learning_rate": 1e-3, "batch_size": 2, "epochs": 1},
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )
    train_batch = SampleBatch(
        syndrome=torch.zeros((2, 3), dtype=torch.float32),
        errors=torch.zeros((2, 3), dtype=torch.int64),
        observables=torch.zeros((2, 1), dtype=torch.int64),
    )
    val_batch = SampleBatch(
        syndrome=torch.zeros((2, 3), dtype=torch.float32),
        errors=torch.zeros((2, 3), dtype=torch.int64),
        observables=torch.zeros((2, 1), dtype=torch.int64),
    )
    calls = iter([train_batch, val_batch])

    class FakeBaseline:
        def decode_batch(self, detections: np.ndarray) -> np.ndarray:
            return np.zeros((detections.shape[0], 1), dtype=np.int64)

    fake_class = type(
        "FakeBaselineClass",
        (),
        {"from_circuit": staticmethod(lambda _artifact: FakeBaseline())},
    )

    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: _stim_artifacts())
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_args, **_kwargs: FakeModel("hard_flip", 3))
    monkeypatch.setattr(runner, "sample_syndromes", lambda *_args, **_kwargs: next(calls))
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runner, "PymatchingBaseline", fake_class)
    monkeypatch.setattr(runner, "decode_with_predecoder", lambda *_args, **_kwargs: np.zeros((2, 1), dtype=np.int64))
    monkeypatch.setattr(runner, "estimate_flops", lambda *_args, **_kwargs: 123)

    metrics = runner.run_round(cfg, env, safety=RunnerSafety(VRAM_PRE_CHECK=False))
    assert metrics.status == "ok"
    assert metrics.flops_per_syndrome == 123


def test_stim_soft_priors_training_target_is_errors_not_syndrome(monkeypatch, tmp_path) -> None:
    """Regression (2026-04-24): soft_priors BCE target on stim_circuit must
    be the DEM errors (shape (B, n_var)), not the raw syndrome
    (shape (B, n_check)). The earlier ``target = batch_syndrome`` branch
    made the model learn an identity map from syndromes to syndromes,
    locking delta_ler at 0 on every surface-code run.
    """
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    n_var, n_check = 10, 6
    batch_size = 4
    n_shots = 8

    artifacts = CodeArtifacts(
        code_type="stim_circuit",
        code_artifact=object(),
        edge_index=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        n_var=n_var,
        n_check=n_check,
        prior_p=torch.full((n_var,), 0.05, dtype=torch.float32),
    )
    train_batch = SampleBatch(
        syndrome=torch.randint(0, 2, (n_shots, n_check)).float(),
        errors=torch.randint(0, 2, (n_shots, n_var), dtype=torch.int64),
        observables=torch.zeros((n_shots, 1), dtype=torch.int64),
    )
    val_batch = SampleBatch(
        syndrome=torch.zeros((n_shots, n_check), dtype=torch.float32),
        errors=torch.zeros((n_shots, n_var), dtype=torch.int64),
        observables=torch.zeros((n_shots, 1), dtype=torch.int64),
    )
    calls = iter([train_batch, val_batch])

    captured_shapes: list[tuple[int, int]] = []
    orig_bce = F.binary_cross_entropy

    def spy_bce(pred, target, *a, **kw):
        captured_shapes.append(tuple(target.shape))
        return orig_bce(pred, target, *a, **kw)

    class FakeBaseline:
        def decode_batch(self, detections: np.ndarray) -> np.ndarray:
            return np.zeros((detections.shape[0], 1), dtype=np.int64)

    fake_class = type(
        "FakeBaselineClass",
        (),
        {"from_circuit": staticmethod(lambda _artifact: FakeBaseline())},
    )

    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "soft_priors",
            "training": {
                "learning_rate": 1e-3,
                "batch_size": batch_size,
                "epochs": 1,
            },
        },
        training_profile="dev",
        seed=0,
        round_dir=str(tmp_path / "round_1"),
    )
    monkeypatch.setattr(runner, "load_code_artifacts", lambda _env: artifacts)
    monkeypatch.setattr(runner, "compile_predecoder", lambda *_a, **_k: FakeModel("soft_priors", n_var))
    monkeypatch.setattr(runner, "sample_syndromes", lambda *_a, **_k: next(calls))
    monkeypatch.setattr(runner.F, "binary_cross_entropy", spy_bce)
    monkeypatch.setattr(runner, "PymatchingBaseline", fake_class)
    monkeypatch.setattr(
        runner, "decode_with_predecoder",
        lambda *_a, **_k: np.zeros((n_shots, 1), dtype=np.int64),
    )
    monkeypatch.setattr(runner, "estimate_flops", lambda *_a, **_k: 100)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    metrics = runner.run_round(cfg, env, safety=RunnerSafety(VRAM_PRE_CHECK=False))

    assert metrics.status == "ok"
    assert captured_shapes, "BCE was never called — training loop never ran"
    for shape in captured_shapes:
        assert shape == (batch_size, n_var), (
            f"BCE target shape must be (batch, n_var)=({batch_size}, {n_var}); "
            f"got {shape}. This means the runner is still feeding the syndrome "
            f"as the training target."
        )
