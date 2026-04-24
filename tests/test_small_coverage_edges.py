from __future__ import annotations

import sys
import types

import click
from click.testing import CliRunner
import numpy as np
import pytest
import stim
import torch

import cli.autoqec as cli
from autoqec.agents.dispatch import parse_response
from autoqec.decoders.backend_adapter import decode_with_predecoder
from autoqec.decoders.dsl_schema import PredecoderDSL
from autoqec.envs.schema import load_env_yaml
from autoqec.runner.flops import estimate_flops
from autoqec.runner.schema import RoundMetrics


def test_estimate_flops_success_and_fallback(monkeypatch) -> None:
    class FakeAnalysis:
        def __init__(self, _model, _inputs):
            pass

        def total(self):
            return 123

    fake_module = types.ModuleType("fvcore.nn")
    fake_module.FlopCountAnalysis = FakeAnalysis
    monkeypatch.setitem(sys.modules, "fvcore.nn", fake_module)

    model = torch.nn.Linear(2, 3)
    assert estimate_flops(model, (torch.randn(1, 2),)) == 123

    class RaisingAnalysis:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("nope")

    fake_module.FlopCountAnalysis = RaisingAnalysis
    assert estimate_flops(model, (torch.randn(1, 2),)) == 2 * sum(p.numel() for p in model.parameters())


def test_backend_adapter_error_branches(monkeypatch) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")

    with pytest.raises(TypeError, match="Stim circuit artifact"):
        decode_with_predecoder(np.zeros((1, 2)), env, np.zeros((1, 2)), np.zeros((2, 2)), "hard_flip")

    class FakeMatching:
        def decode_batch(self, arr):
            return arr.astype(np.int64)

        def decode(self, arr):
            # Single-shot variant used by the reweighted-MWPM path. Return
            # zero observables regardless of the input so callers only need
            # the shape to line up.
            return np.zeros(1, dtype=np.int64)

    fake_pymatching = types.ModuleType("pymatching")
    fake_pymatching.Matching = type(
        "Matching",
        (),
        {"from_detector_error_model": staticmethod(lambda _dem: FakeMatching())},
    )
    monkeypatch.setitem(sys.modules, "pymatching", fake_pymatching)
    circuit = stim.Circuit.from_file(env.code.source)
    syndrome = np.zeros((2, circuit.num_detectors), dtype=np.uint8)
    # soft_priors MWPM path rebuilds a DEM per shot, so priors must have
    # one column per DEM error mechanism (not per detector).
    dem = circuit.detector_error_model(decompose_errors=True)
    priors = np.ones((2, dem.num_errors), dtype=float) * 0.01
    out = decode_with_predecoder(priors, env, syndrome, circuit, "soft_priors")
    assert out.shape[0] == 2

    bad_env = env.model_copy(update={"classical_backend": "unknown"})
    with pytest.raises(ValueError, match="Unknown backend"):
        decode_with_predecoder(np.zeros((1, 2)), bad_env, np.zeros((1, 2)), circuit, "hard_flip")


def test_dsl_schema_family_specific_errors() -> None:
    base_training = {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "epochs": 1,
        "loss": "bce",
        "profile": "dev",
    }
    with pytest.raises(ValueError, match="gnn spec required"):
        PredecoderDSL(type="gnn", output_mode="soft_priors", head="linear", training=base_training)

    with pytest.raises(ValueError, match="not allowed when type='gnn'"):
        PredecoderDSL(
            type="gnn",
            output_mode="soft_priors",
            gnn={
                "layers": 1,
                "hidden_dim": 8,
                "message_fn": "mlp",
                "aggregation": "sum",
                "normalization": "none",
                "residual": False,
                "edge_features": [],
            },
            neural_bp={
                "iterations": 1,
                "weight_sharing": "per_layer",
                "damping": "learnable_scalar",
                "attention_aug": False,
                "attention_heads": 1,
            },
            head="linear",
            training=base_training,
        )

    with pytest.raises(ValueError, match="neural_bp spec required"):
        PredecoderDSL(type="neural_bp", output_mode="soft_priors", head="linear", training=base_training)

    with pytest.raises(ValueError, match="not allowed when type='neural_bp'"):
        PredecoderDSL(
            type="neural_bp",
            output_mode="soft_priors",
            gnn={
                "layers": 1,
                "hidden_dim": 8,
                "message_fn": "mlp",
                "aggregation": "sum",
                "normalization": "none",
                "residual": False,
                "edge_features": [],
            },
            neural_bp={
                "iterations": 1,
                "weight_sharing": "per_layer",
                "damping": "learnable_scalar",
                "attention_aug": False,
                "attention_heads": 1,
            },
            head="linear",
            training=base_training,
        )


def test_machine_state_gpu_snapshot_success(monkeypatch) -> None:
    fake_torch = types.ModuleType("torch")

    class _Props:
        total_memory = 8_000_000_000

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def mem_get_info():
            return (4_000_000_000, 8_000_000_000)

        @staticmethod
        def get_device_name(_idx: int) -> str:
            return "Fake GPU"

        @staticmethod
        def get_device_properties(_idx: int):
            return _Props()

    fake_torch.cuda = _FakeCuda()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    from autoqec.tools.machine_state import _gpu_snapshot

    snapshot = _gpu_snapshot()
    assert snapshot["name"] == "Fake GPU"
    assert snapshot["vram_free_gb"] == 4.0


def test_parse_response_unknown_role_raises() -> None:
    with pytest.raises(ValueError, match="Unknown role"):
        parse_response("ghost", "```json\n{}\n```")  # type: ignore[arg-type]


def test_run_round_impl_dispatches_subprocess_branch(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("type: gnn\noutput_mode: soft_priors\ntraining:\n  learning_rate: 0.001\n  batch_size: 1\n  epochs: 1\n  loss: bce\n  profile: dev\n")
    captured = {}

    monkeypatch.setattr(cli, "load_env_yaml", lambda _path: env)
    monkeypatch.setattr(
        cli,
        "yaml",
        types.SimpleNamespace(safe_load=lambda _f: {"type": "gnn", "output_mode": "soft_priors", "training": {"learning_rate": 0.001, "batch_size": 1, "epochs": 1, "loss": "bce", "profile": "dev"}}),
    )

    fake_module = types.ModuleType("autoqec.orchestration.subprocess_runner")

    def fake_run_round_in_subprocess(cfg, _env, round_attempt_id=None):
        captured["cfg"] = cfg
        captured["round_attempt_id"] = round_attempt_id
        return RoundMetrics(status="ok")

    fake_module.run_round_in_subprocess = fake_run_round_in_subprocess
    monkeypatch.setitem(sys.modules, "autoqec.orchestration.subprocess_runner", fake_module)
    monkeypatch.setattr(click, "echo", lambda msg: captured.setdefault("echo", msg))

    cli._run_round_impl(
        env_yaml="env.yaml",
        config_yaml=str(cfg_path),
        round_dir=str(tmp_path / "round_1"),
        profile="dev",
        code_cwd=str(tmp_path),
        branch="exp/t/01-a",
        fork_from=None,
        compose_mode=None,
        round_attempt_id="uuid-1",
        _internal_execute_locally=False,
    )

    assert captured["cfg"].code_cwd == str(tmp_path)
    assert captured["round_attempt_id"] == "uuid-1"


def test_run_command_errors_when_dev_safe_templates_are_missing(monkeypatch, tmp_path) -> None:
    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_env_yaml", lambda _path: env)
    monkeypatch.setattr(cli.time, "strftime", lambda _fmt: "20260423-000002")
    monkeypatch.setattr(cli, "RunMemory", lambda *_args, **_kwargs: types.SimpleNamespace(append_round=lambda _r: None, update_pareto=lambda _p: None))
    monkeypatch.setattr(cli, "load_example_templates", lambda: [("other", {"type": "gnn", "training": {"learning_rate": 1e-3, "batch_size": 1, "epochs": 1}})])
    monkeypatch.setattr(cli.random, "choice", lambda items: items[0])

    runner = CliRunner()
    result = runner.invoke(
        cli.run,
        [env.model_dump()["name"], "--rounds", "1", "--profile", "dev", "--no-llm"],
        catch_exceptions=True,
    )
    assert result.exit_code != 0
    assert "No bundled templates are available" in result.output
