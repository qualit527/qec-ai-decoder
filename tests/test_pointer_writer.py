import json
import types

import pytest

from autoqec.runner.pointer import write_round_pointer
from autoqec.runner.schema import RoundMetrics, RunnerConfig


def _cfg(tmp, *, branch=None, code_cwd=None, rid="uuid-1"):
    rd = tmp / "round_1"
    rd.mkdir(parents=True)
    return RunnerConfig(
        env_name="surface_d5_depol",
        predecoder_config={"type": "gnn", "output_mode": "soft_priors"},
        training_profile="dev",
        seed=0,
        round_dir=str(rd),
        code_cwd=code_cwd,
        branch=branch,
        round_attempt_id=rid,
    )


def test_pointer_all_fields(tmp_path):
    cfg = _cfg(tmp_path, branch="exp/t/01-x",
               code_cwd=str(tmp_path / ".worktrees" / "exp-x"))
    m = RoundMetrics(
        status="ok", delta_ler=0.003, flops_per_syndrome=100, n_params=500,
        train_wallclock_s=1.0, eval_wallclock_s=0.1,
        branch=cfg.branch, commit_sha="abc", round_attempt_id="uuid-1",
    )
    out = write_round_pointer(cfg=cfg, metrics=m, round_idx=1)
    assert out.name == "round_1_pointer.json"
    d = json.loads(out.read_text())
    assert d == {
        "round_attempt_id": "uuid-1",
        "reconcile_id": None,
        "branch": "exp/t/01-x",
        "commit_sha": "abc",
        "worktree_path": str(tmp_path / ".worktrees" / "exp-x"),
        "fork_from": None,
        "fork_from_ordered": None,
        "compose_mode": None,
        "status": "ok",
        "status_reason": None,
    }


def test_pointer_baseline_round(tmp_path):
    cfg = _cfg(tmp_path)
    m = RoundMetrics(status="ok", delta_ler=0.0, flops_per_syndrome=1,
                     n_params=1, train_wallclock_s=0.1, eval_wallclock_s=0.01,
                     round_attempt_id="uuid-2")
    d = json.loads(write_round_pointer(cfg=cfg, metrics=m, round_idx=2).read_text())
    assert d["branch"] is None and d["commit_sha"] is None


def test_pointer_compose_conflict(tmp_path):
    cfg = _cfg(tmp_path, rid="uuid-3")
    m = RoundMetrics(status="compose_conflict", round_attempt_id="uuid-3",
                     status_reason="parent-A vs parent-B conflict on foo.py")
    d = json.loads(write_round_pointer(cfg=cfg, metrics=m, round_idx=3).read_text())
    assert d["status"] == "compose_conflict"
    assert d["branch"] is None
    assert d["status_reason"].startswith("parent-A")


def test_pointer_requires_id(tmp_path):
    cfg = _cfg(tmp_path, rid="")
    m = RoundMetrics(status="ok", round_attempt_id="")
    with pytest.raises(ValueError, match="round_attempt_id or reconcile_id"):
        write_round_pointer(cfg=cfg, metrics=m, round_idx=4)


def test_pointer_rejects_both_ids(tmp_path):
    """§15.2 mutual-exclusion: round_attempt_id and reconcile_id cannot both be set."""
    cfg = _cfg(tmp_path, rid="uuid-xor")
    # RoundMetrics._provenance_integrity blocks both-set at pydantic time,
    # so use a loose SimpleNamespace to smuggle the invalid combination
    # directly into the pointer writer and prove it raises.
    m = types.SimpleNamespace(
        status="ok",
        status_reason=None,
        branch=None,
        commit_sha=None,
        round_attempt_id="uuid-xor",
        reconcile_id="reconcile-xor",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        write_round_pointer(cfg=cfg, metrics=m, round_idx=5)
