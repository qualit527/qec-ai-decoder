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

    assert set(summary) == {
        "best_delta_ler",
        "best_round",
        "claim",
        "env_yaml",
        "has_positive_delta",
        "improvement_vs_round_1",
        "rounds",
        "run_id",
    }
    assert summary["claim"] == "benchmark evidence, not a VERIFIED holdout claim"
    assert summary["has_positive_delta"] is True
    assert len(summary["rounds"]) == 3
    assert [row["round"] for row in summary["rounds"]] == [1, 2, 3]
    assert [row["config"] for row in summary["rounds"]] == [
        "round_1_gnn_small.yaml",
        "round_2_gnn_gated.yaml",
        "round_3_neural_bp.yaml",
    ]

    def assert_portable_path(value: str) -> None:
        assert not Path(value).is_absolute()
        assert "/home/" not in value
        assert ".worktrees" not in value

    for key, value in summary.items():
        if key.endswith("_path") or key.endswith("_yaml"):
            assert isinstance(value, str)
            assert_portable_path(value)

    required_row_keys = {
        "best_delta_ler_so_far",
        "checkpoint_path",
        "config",
        "delta_ler",
        "delta_ler_ci_high",
        "delta_ler_ci_low",
        "eval_wallclock_s",
        "flops_per_syndrome",
        "ler_plain_classical",
        "ler_predecoder",
        "n_params",
        "round",
        "round_dir",
        "status",
        "status_reason",
        "train_wallclock_s",
        "training_log_path",
    }
    for row in summary["rounds"]:
        assert set(row) == required_row_keys
        assert row["status"] == "ok"
        for key, value in row.items():
            if key.endswith("_path") or key.endswith("_dir"):
                assert isinstance(value, str)
                assert_portable_path(value)

    best_row = max(summary["rounds"], key=lambda row: row["delta_ler"])
    assert summary["best_delta_ler"] == best_row["delta_ler"]
    assert summary["best_round"] == best_row["round"]
    assert summary["improvement_vs_round_1"] == best_row["delta_ler"] - summary["rounds"][0]["delta_ler"]
    assert summary["best_delta_ler"] > 0
    assert summary["improvement_vs_round_1"] > 0
