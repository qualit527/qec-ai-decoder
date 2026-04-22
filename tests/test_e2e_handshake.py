"""Tests for scripts.e2e_handshake (Task A2.1).

The end-to-end training run is marked `integration` because it needs torch
+ stim + pymatching installed. The contract checks (config loads, env
loads, RunnerConfig builds) are unit tests and run in CI without GPU.
"""
from __future__ import annotations

from pathlib import Path

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


@pytest.mark.integration
def test_handshake_script_main_runs_e2e(tmp_path: Path) -> None:
    """Actually runs one dev-profile round. Requires torch + stim + pymatching."""
    from scripts.e2e_handshake import main

    result = main(round_dir=tmp_path / "round_0", seed=0)
    assert result["status"] == "ok"
    metrics_path = tmp_path / "round_0" / "metrics.json"
    assert metrics_path.exists()
