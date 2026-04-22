import pytest
from autoqec.runner.schema import RunnerConfig, RoundMetrics


def test_runner_config_accepts_legacy_form():
    cfg = RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir="/tmp/round_1",
    )
    assert cfg.code_cwd is None
    assert cfg.branch is None
    assert cfg.fork_from is None


def test_runner_config_rejects_code_cwd_without_branch():
    with pytest.raises(ValueError, match="branch is required"):
        RunnerConfig(
            env_name="surface_d5_depol",
            predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
            training_profile="dev",
            seed=0,
            round_dir="/tmp/round_1",
            code_cwd="/abs/path/.worktrees/exp-x",
        )


def test_runner_config_rejects_compose_without_mode():
    with pytest.raises(ValueError, match="compose_mode is required"):
        RunnerConfig(
            env_name="surface_d5_depol",
            predecoder_config={},
            training_profile="dev",
            seed=0,
            round_dir="/tmp/round_1",
            fork_from=["exp/.../02-a", "exp/.../04-b"],
        )


def test_round_metrics_requires_round_attempt_id_on_worktree_path():
    with pytest.raises(ValueError, match="round_attempt_id"):
        RoundMetrics(
            status="ok",
            branch="exp/20260422-140000/01-small",
            commit_sha="abc123",
        )


def test_round_metrics_requires_commit_sha_when_branch_set():
    with pytest.raises(ValueError, match="commit_sha is required"):
        RoundMetrics(
            status="ok",
            branch="exp/20260422-140000/01-small",
            round_attempt_id="8b4f2c1e-9d3a-4f1b-b2e7-56f0ab7c3def",
        )


def test_round_metrics_compose_conflict_rejects_branch():
    with pytest.raises(ValueError, match="compose_conflict rows must have branch=None"):
        RoundMetrics(
            status="compose_conflict",
            branch="exp/20260422-140000/12-compose",
            round_attempt_id="8b4f2c1e-9d3a-4f1b-b2e7-56f0ab7c3def",
        )


def test_round_metrics_mutually_exclusive_ids():
    with pytest.raises(ValueError, match="mutually exclusive"):
        RoundMetrics(
            status="orphaned_branch",
            round_attempt_id="8b4f2c1e-9d3a-4f1b-b2e7-56f0ab7c3def",
            reconcile_id="c3def-8b4f2c1e-abc",
        )


def test_round_metrics_legacy_in_process_path_still_validates():
    # No branch / fork_from / round_attempt_id — existing Lin-era callers.
    m = RoundMetrics(status="ok", ler_plain_classical=1e-3, ler_predecoder=5e-4, delta_ler=5e-4)
    assert m.branch is None
    assert m.status == "ok"


def test_round_metrics_branch_manually_deleted_allows_commit_sha_none():
    """§15.10 follow-up rows carry `branch` but the branch is gone — no commit_sha.

    The `branch implies commit_sha` invariant must carve out this exception
    so reconcile-emitted rows round-trip through the schema.
    """
    m = RoundMetrics(
        status="branch_manually_deleted",
        branch="exp/t/03-z",
        commit_sha=None,
        round_attempt_id="abc-123",
    )
    assert m.branch == "exp/t/03-z"
    assert m.commit_sha is None


def test_round_metrics_ok_status_still_requires_commit_sha_when_branch_set():
    """The exception only covers branch_manually_deleted, not normal rows."""
    with pytest.raises(ValueError, match="commit_sha is required"):
        RoundMetrics(
            status="ok",
            branch="exp/t/03-z",
            commit_sha=None,
            round_attempt_id="abc-123",
        )
