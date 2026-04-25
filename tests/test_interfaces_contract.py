"""Phase 2.1.3 — cross-check pydantic models against `docs/contracts/interfaces.md`.

Each pydantic model documented in the contract file has its authoritative
field list frozen here. Drift (a new field added in code without updating
the contract, or the contract declaring a field the code doesn't have)
fails the test with a readable diff, flagging the missing `contract-change`
sign-off per CLAUDE.md.

This is NOT a markdown parser — it is a brittle-on-purpose inventory. When
the contract changes, update both files in the same commit.
"""

from __future__ import annotations

from pydantic import BaseModel

import pytest


# Expected field sets, transcribed from docs/contracts/interfaces.md §2.1–§15.
EXPECTED_FIELDS: dict[str, set[str]] = {
    # §2.1 EnvSpec and friends
    "SeedPolicy": {"train", "val", "holdout"},
    "NoiseSpec": {"type", "p", "seed_policy"},
    "CodeSpec": {"type", "source"},
    "ConstraintsSpec": {
        "latency_flops_budget",
        "param_budget",
        "target_ler",
        "target_p",
    },
    "EvalProtocol": {
        "min_shots_train",
        "min_shots_val",
        "min_shots_verify",
        "bootstrap_ci",
        "osd_orders_reported",
        "x_z_decoding",
    },
    "EnvSpec": {
        "name",
        "code",
        "noise",
        "constraints",
        "baseline_decoders",
        "classical_backend",
        "eval_protocol",
    },
    # §2.2 + §15 RunnerConfig
    "RunnerConfig": {
        "env_name",
        "predecoder_config",
        "training_profile",
        "seed",
        "round_dir",
        # §15 worktree additions
        "code_cwd",
        "branch",
        "fork_from",
        "fork_from_canonical",
        "fork_from_ordered",
        "compose_mode",
        "round_attempt_id",
        "commit_message",
        "env_yaml_path",
        "invocation_argv",
    },
    # §2.2 + §15 RoundMetrics
    "RoundMetrics": {
        # §2.2 base
        "status",
        "status_reason",
        "ler_plain_classical",
        "ler_predecoder",
        "delta_ler",
        "delta_ler_ci_low",
        "delta_ler_ci_high",
        "flops_per_syndrome",
        "n_params",
        "train_wallclock_s",
        "eval_wallclock_s",
        "vram_peak_gb",
        "checkpoint_path",
        "training_log_path",
        # Training-loss telemetry (added 2026-04-24 for deltaLER=0 diagnostics)
        "train_loss_initial",
        "train_loss_final",
        "train_loss_mean_last_epoch",
        "train_batches_total",
        # §15 additions
        "round_attempt_id",
        "reconcile_id",
        "branch",
        "commit_sha",
        "fork_from",
        "fork_from_canonical",
        "fork_from_ordered",
        "compose_mode",
        "delta_vs_parent",
        "parent_ler",
        "conflicting_files",
        "train_seed",
    },
    # §2.3 + §15 VerifyReport
    "VerifyReport": {
        "verdict",
        "ler_holdout",
        "ler_holdout_ci",
        "delta_ler_holdout",
        "ler_shuffled",
        "ablation_sanity_ok",
        "holdout_seeds_used",
        "seed_leakage_check_ok",
        "notes",
        # §15 additions
        "branch",
        "commit_sha",
        "delta_vs_baseline_holdout",
        "paired_eval_bundle_id",
    },
    # §2.5 / §15 subagent message format
    "IdeatorResponse": {
        "hypothesis",
        "expected_delta_ler",
        "expected_cost_s",
        "rationale",
        "dsl_hint",
        "fork_from",
        "compose_mode",
    },
    "CoderResponse": {
        "tier",
        "dsl_config",
        "rationale",
        "commit_message",
    },
    "AnalystResponse": {
        "summary_1line",
        "verdict",
        "next_hypothesis_seed",
        "branch",
        "commit_sha",
    },
}


def _live_models() -> dict[str, type[BaseModel]]:
    """Import the live pydantic models from their authoritative modules."""
    from autoqec.agents.schemas import (
        AnalystResponse,
        CoderResponse,
        IdeatorResponse,
    )
    from autoqec.envs.schema import (
        CodeSpec,
        ConstraintsSpec,
        EnvSpec,
        EvalProtocol,
        NoiseSpec,
        SeedPolicy,
    )
    from autoqec.eval.schema import VerifyReport
    from autoqec.runner.schema import RoundMetrics, RunnerConfig

    return {
        "SeedPolicy": SeedPolicy,
        "NoiseSpec": NoiseSpec,
        "CodeSpec": CodeSpec,
        "ConstraintsSpec": ConstraintsSpec,
        "EvalProtocol": EvalProtocol,
        "EnvSpec": EnvSpec,
        "RunnerConfig": RunnerConfig,
        "RoundMetrics": RoundMetrics,
        "VerifyReport": VerifyReport,
        "IdeatorResponse": IdeatorResponse,
        "CoderResponse": CoderResponse,
        "AnalystResponse": AnalystResponse,
    }


@pytest.mark.parametrize("model_name", sorted(EXPECTED_FIELDS))
def test_interface_model_fields_match_contract(model_name: str) -> None:
    live_models = _live_models()
    model = live_models[model_name]
    actual = set(model.model_fields.keys())
    expected = EXPECTED_FIELDS[model_name]
    missing_in_code = expected - actual
    extra_in_code = actual - expected
    assert not missing_in_code and not extra_in_code, (
        f"{model_name} drift vs docs/contracts/interfaces.md:\n"
        f"  missing in code (documented but not defined): {sorted(missing_in_code)}\n"
        f"  extra in code   (defined but not documented):  {sorted(extra_in_code)}\n"
        "Contract changes require 3-of-3 owner sign-off (CLAUDE.md §Commit)."
    )


def test_all_contracted_models_importable() -> None:
    """Every model named in interfaces.md must actually exist."""
    live = _live_models()
    assert set(live) == set(EXPECTED_FIELDS), (
        f"coverage drift: live={sorted(live)} vs expected={sorted(EXPECTED_FIELDS)}"
    )
