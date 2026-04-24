from __future__ import annotations

from pathlib import Path

import yaml

from autoqec.decoders.dsl_compiler import compile_predecoder
from autoqec.envs.schema import load_env_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / "autoqec/envs/builtin/bb72_perf.yaml"
CONFIG_DIR = REPO_ROOT / "experiments/bb72-positive-delta/configs"


def test_bb72_perf_env_uses_osd_and_nonzero_error_budget() -> None:
    env = load_env_yaml(ENV_PATH)

    assert env.name == "bb72_perf"
    assert env.code.type == "parity_check_matrix"
    assert Path(env.code.source).name == "bb72_Hx.npy"
    assert env.classical_backend == "osd"
    assert env.noise.p[0] == 0.05
    assert env.eval_protocol.osd_orders_reported == [0]
    assert env.eval_protocol.min_shots_val >= 4096


def test_positive_delta_round_configs_compile() -> None:
    config_paths = sorted(CONFIG_DIR.glob("round_*.yaml"))

    assert [path.name for path in config_paths] == [
        "round_1_gnn_small.yaml",
        "round_2_gnn_gated.yaml",
        "round_3_neural_bp.yaml",
    ]
    for path in config_paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = compile_predecoder(cfg, n_var=72, n_check=36)
        assert model.output_mode == "soft_priors"
        assert sum(parameter.numel() for parameter in model.parameters()) > 0
        assert cfg["training"]["epochs"] >= 3


def test_positive_delta_readme_states_scope_and_reproduction_command() -> None:
    readme = (REPO_ROOT / "experiments/bb72-positive-delta/README.md").read_text(encoding="utf-8")

    assert "BB72/OSD" in readme
    assert "benchmark evidence" in readme
    assert "not a VERIFIED holdout claim" in readme
    assert "python experiments/bb72-positive-delta/run.py" in readme
    assert "runs/YYYYMMDDTHHMMSSZ-bb72-positive-delta/" in readme
    assert "surface_d5 + mwpm + soft_priors" in readme


def test_positive_delta_expected_summary_schema() -> None:
    import json

    summary_path = REPO_ROOT / "experiments/bb72-positive-delta/expected_output/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["claim"] == "benchmark evidence, not a VERIFIED holdout claim"
    assert summary["has_positive_delta"] is True
    assert summary["best_delta_ler"] > 0
    assert summary["improvement_vs_round_1"] > 0
    assert summary["best_round"] in {2, 3}
    assert len(summary["rounds"]) == 3
    for row in summary["rounds"]:
        assert row["status"] == "ok"
        assert "delta_ler" in row
        assert "best_delta_ler_so_far" in row
        assert "ler_plain_classical" in row
        assert "ler_predecoder" in row
