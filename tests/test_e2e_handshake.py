"""Tests for scripts.e2e_handshake (Task A2.1).

The end-to-end training run is marked `integration` because it needs torch
+ stim + pymatching installed. The contract checks (config loads, env
loads, RunnerConfig builds) are unit tests and run in CI without GPU.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import pytest


def test_handshake_stub_yaml_validates_against_predecoder_dsl() -> None:
    """If the stub drifts from PredecoderDSL, every future handshake run
    will fail at `compile_predecoder` time — catch it at unit-test time."""
    import yaml

    from autoqec.decoders.dsl_schema import PredecoderDSL

    data = yaml.safe_load(
        Path("autoqec/example_db/handshake_stub.yaml").read_text(encoding="utf-8")
    )
    PredecoderDSL(**data)  # raises on schema drift


def test_build_runner_config_composes_env_and_stub(tmp_path: Path) -> None:
    """Contract between orchestration and Runner: RunnerConfig fields line
    up with what Lin's run_round expects."""
    from scripts.e2e_handshake import build_runner_config

    cfg = build_runner_config(
        env_yaml="autoqec/envs/builtin/surface_d5_depol.yaml",
        stub_yaml="autoqec/example_db/handshake_stub.yaml",
        round_dir=tmp_path / "round_0",
        seed=7,
    )
    assert cfg.env_name == "surface_d5_depol"
    assert cfg.training_profile == "dev"
    assert cfg.seed == 7
    assert Path(cfg.round_dir).is_absolute()
    # predecoder_config was parsed as a plain dict (Runner expects dict, not pydantic)
    assert cfg.predecoder_config["type"] == "gnn"
    assert cfg.predecoder_config["gnn"]["hidden_dim"] == 16


def test_build_runner_config_min_steps_scales_dev_batch_budget(tmp_path: Path) -> None:
    """The verification gate can request a taller train.log without making
    the default handshake slow for normal smoke runs."""
    from scripts.e2e_handshake import build_runner_config

    cfg = build_runner_config(
        env_yaml="autoqec/envs/builtin/surface_d5_depol.yaml",
        stub_yaml="autoqec/example_db/handshake_stub.yaml",
        round_dir=tmp_path / "round_0",
        min_steps=100,
    )

    training = cfg.predecoder_config["training"]
    assert cfg.training_profile == "dev"
    assert training["profile"] == "dev"
    assert training["epochs"] == 1
    assert training["batch_size"] == 2


def test_main_min_steps_invocation_writes_train_log_with_100_wc_lines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit-test the supported invocation's wc -l contract without importing torch."""
    from autoqec.runner.schema import RoundMetrics
    from scripts.e2e_handshake import main

    def fake_run_round(config, env_spec):
        round_dir = Path(config.round_dir)
        round_dir.mkdir(parents=True, exist_ok=True)
        n_shots_train = min(env_spec.eval_protocol.min_shots_train, 256)
        epochs = min(int(config.predecoder_config["training"]["epochs"]), 1)
        batch_size = int(config.predecoder_config["training"]["batch_size"])
        rows = math.ceil(n_shots_train / batch_size) * epochs
        train_log = round_dir / "train.log"
        train_log.write_text(
            "\n".join(f"{idx}\t0.1" for idx in range(rows)),
            encoding="utf-8",
        )
        return RoundMetrics(status="ok", training_log_path=str(train_log))

    fake_runner_module = types.ModuleType("autoqec.runner.runner")
    fake_runner_module.run_round = fake_run_round
    monkeypatch.setitem(sys.modules, "autoqec.runner.runner", fake_runner_module)

    round_dir = tmp_path / "round_0"
    main(round_dir=round_dir, seed=0, min_steps=100)

    assert round_dir.joinpath("train.log").read_bytes().count(b"\n") >= 100


@pytest.mark.integration
def test_handshake_script_main_runs_e2e(tmp_path: Path) -> None:
    """Actually runs one dev-profile round. Requires torch + stim + pymatching."""
    from scripts.e2e_handshake import main

    result = main(round_dir=tmp_path / "round_0", seed=0)
    assert result["status"] == "ok"
    metrics_path = tmp_path / "round_0" / "metrics.json"
    assert metrics_path.exists()


def test_build_runner_config_works_from_foreign_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex review (medium): handshake must work from any cwd, not only repo root."""
    import os

    from scripts.e2e_handshake import build_runner_config

    monkeypatch.chdir(tmp_path)  # foreign cwd — repo paths must still resolve
    cfg = build_runner_config(
        env_yaml=None,
        stub_yaml=None,
        round_dir=tmp_path / "round_0",
    )
    assert cfg.env_name == "surface_d5_depol"
    assert cfg.predecoder_config["type"] == "gnn"
    # sanity: cwd is not the repo root
    assert Path(os.getcwd()) != Path(__file__).resolve().parents[1]
