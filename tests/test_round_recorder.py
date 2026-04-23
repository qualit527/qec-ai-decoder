"""Unit tests for the round recorder (B2 + §15.2 non-dominated Pareto).

The recorder now takes a single `round_metrics` dict (flattened row shape
matching `RoundMetrics` plus the superset fields from §15.7) and a
separate `verify_verdict` gate — "VERIFIED" is the only value that admits
the row into the Pareto archive. Per §15.2/§15.7 Pareto rows are built
from BOTH RoundMetrics (cost/provenance) AND VerifyReport (holdout quality).
"""
from __future__ import annotations

import json
from pathlib import Path

from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import (
    admit_verified_round_to_pareto,
    record_round,
)


def _admit(mem: RunMemory, **overrides) -> None:
    """Helper: simulate a VERIFIED candidate admission (full superset row).

    Populates both the RoundMetrics-side cost/provenance fields AND a
    fabricated VerifyReport-side payload so Pareto admission works under
    the §15.2/§15.7 contract.
    """
    base = {
        "status": "ok",
        "hypothesis": "h",
        "delta_ler": 1e-4,
        "flops_per_syndrome": 100_000,
        "n_params": 50_000,
        "checkpoint_path": "runs/foo/round_1/checkpoint.pt",
        "round": 1,
        "branch": None,
        "commit_sha": None,
        "round_attempt_id": None,
    }
    base.update(overrides)
    # Default VerifyReport mirrors delta_ler into holdout space unless overridden.
    holdout_delta = overrides.pop("delta_vs_baseline_holdout", base.get("delta_ler"))
    verify_report = {
        "verdict": "VERIFIED",
        "delta_vs_baseline_holdout": holdout_delta,
        "ler_holdout": overrides.get("ler_holdout", 5e-4),
        "paired_eval_bundle_id": overrides.get(
            "paired_eval_bundle_id", "bundle-default"
        ),
    }
    record_round(
        mem,
        round_metrics=base,
        verify_verdict="VERIFIED",
        verify_report=verify_report,
    )


# ─── §15.2 non-dominated Pareto tests ────────────────────────────────────


def test_pareto_keeps_dominated_points_no_more(tmp_path: Path) -> None:
    # Three points: a and b are non-dominated, c is dominated by a.
    # Holdout delta mirrors training delta_ler via the _admit helper default.
    mem = RunMemory(tmp_path)
    _admit(mem, round=1, delta_ler=4e-4, flops_per_syndrome=200_000, n_params=40_000,
           branch="exp/t/01-a", commit_sha="a1", round_attempt_id="u1")
    _admit(mem, round=2, delta_ler=2e-4, flops_per_syndrome=50_000, n_params=20_000,
           branch="exp/t/02-b", commit_sha="b1", round_attempt_id="u2")
    _admit(mem, round=3, delta_ler=1e-4, flops_per_syndrome=300_000, n_params=60_000,
           branch="exp/t/03-c", commit_sha="c1", round_attempt_id="u3")
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    branches = {row["branch"] for row in pareto}
    assert branches == {"exp/t/01-a", "exp/t/02-b"}  # c is dominated


def test_pareto_has_no_size_cap(tmp_path: Path) -> None:
    # Admit 7 distinct non-dominated points (vary the axes to avoid dominance).
    mem = RunMemory(tmp_path)
    for i in range(7):
        _admit(
            mem,
            round=i + 1,
            delta_ler=1e-4 + i * 1e-5,           # higher delta = better
            flops_per_syndrome=400_000 - i * 50_000,  # lower flops = better
            n_params=10_000 + i * 10_000,        # lower params = better (tradeoff)
            branch=f"exp/t/{i+1:02d}-x",
            commit_sha=f"sha{i}",
            round_attempt_id=f"u{i}",
        )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert len(pareto) == 7  # no truncation to 5


def test_pareto_preview_is_top_5_by_holdout_delta(tmp_path: Path) -> None:
    """Preview sort key is -delta_vs_baseline_holdout per §15.7."""
    mem = RunMemory(tmp_path)
    for i in range(7):
        _admit(
            mem,
            round=i + 1,
            delta_ler=1e-4 + i * 1e-5,
            flops_per_syndrome=400_000 - i * 50_000,
            n_params=10_000 + i * 10_000,
            branch=f"exp/t/{i+1:02d}-x",
            commit_sha=f"sha{i}",
            round_attempt_id=f"u{i}",
        )
    preview = json.loads((tmp_path / "pareto_preview.json").read_text())
    assert len(preview) == 5
    holdout_deltas = [row["delta_vs_baseline_holdout"] for row in preview]
    assert holdout_deltas == sorted(holdout_deltas, reverse=True)


def test_admit_verified_round_to_pareto_rejects_non_verified_report(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    admitted = admit_verified_round_to_pareto(
        mem,
        round_metrics={"round": 1},
        verify_report={"verdict": "FAILED", "delta_vs_baseline_holdout": 1e-4},
    )

    assert admitted is False
    assert json.loads((tmp_path / "pareto.json").read_text()) == []
    assert not (tmp_path / "pareto_preview.json").exists()


def test_admit_verified_round_to_pareto_requires_holdout_delta(
    tmp_path: Path,
    caplog,
) -> None:
    mem = RunMemory(tmp_path)
    admitted = admit_verified_round_to_pareto(
        mem,
        round_metrics={"round": 1},
        verify_report={"verdict": "VERIFIED"},
    )

    assert admitted is False
    assert "no delta_vs_baseline_holdout" in caplog.text
    assert json.loads((tmp_path / "pareto.json").read_text()) == []


# ─── Fix 2 — Pareto uses VerifyReport holdout fields, not training delta ──


def test_pareto_row_uses_verify_report_holdout_delta(tmp_path: Path) -> None:
    """Row's delta_vs_baseline_holdout comes from verify_report, not round_metrics."""
    mem = RunMemory(tmp_path)
    round_metrics = {
        "round": 1,
        "status": "ok",
        "hypothesis": "h",
        "delta_ler": 9e-4,  # training-side — must NOT be used for Pareto axis
        "flops_per_syndrome": 100_000,
        "n_params": 50_000,
        "branch": "exp/t/01-a",
        "commit_sha": "sha_a",
        "round_attempt_id": "u1",
    }
    verify_report = {
        "verdict": "VERIFIED",
        "delta_vs_baseline_holdout": 3e-4,  # holdout-side — this is what counts
        "ler_holdout": 4e-4,
        "paired_eval_bundle_id": "bundle-1",
    }
    record_round(
        mem,
        round_metrics=round_metrics,
        verify_verdict="VERIFIED",
        verify_report=verify_report,
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert len(pareto) == 1
    row = pareto[0]
    assert row["delta_vs_baseline_holdout"] == 3e-4
    assert row["paired_eval_bundle_id"] == "bundle-1"
    assert row["verdict"] == "VERIFIED"
    assert row["ler_holdout"] == 4e-4


def test_pareto_dominance_uses_holdout_not_training_delta(tmp_path: Path) -> None:
    """Training delta inverts under holdout — dominance must rank on holdout."""
    mem = RunMemory(tmp_path)
    # Point A: high training delta, low holdout delta (worse in the space that matters).
    record_round(
        mem,
        round_metrics={
            "round": 1, "status": "ok", "hypothesis": "a",
            "delta_ler": 9e-4, "flops_per_syndrome": 100_000, "n_params": 50_000,
            "branch": "exp/t/01-a", "commit_sha": "sha_a", "round_attempt_id": "u1",
        },
        verify_verdict="VERIFIED",
        verify_report={
            "verdict": "VERIFIED", "delta_vs_baseline_holdout": 1e-4,
            "ler_holdout": 5e-4, "paired_eval_bundle_id": "b1",
        },
    )
    # Point B: low training delta, HIGH holdout delta (dominates A on every axis).
    record_round(
        mem,
        round_metrics={
            "round": 2, "status": "ok", "hypothesis": "b",
            "delta_ler": 1e-4, "flops_per_syndrome": 50_000, "n_params": 20_000,
            "branch": "exp/t/02-b", "commit_sha": "sha_b", "round_attempt_id": "u2",
        },
        verify_verdict="VERIFIED",
        verify_report={
            "verdict": "VERIFIED", "delta_vs_baseline_holdout": 9e-4,
            "ler_holdout": 3e-4, "paired_eval_bundle_id": "b2",
        },
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    branches = {r["branch"] for r in pareto}
    # Under holdout dominance, B dominates A → only B survives.
    # Under training-delta dominance (the bug), A and B would both be kept.
    assert branches == {"exp/t/02-b"}


def test_pareto_skips_verified_round_without_flops(tmp_path: Path) -> None:
    """VERIFIED round missing flops_per_syndrome must NOT enter the archive.

    Before this guard, ``_dominates`` silently coerced missing cost fields
    to 0, so a row with no flops instantly "dominated" every real candidate
    on the flops axis. That could prune a clean Pareto front with a single
    malformed submission.
    """
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1, "status": "ok", "hypothesis": "h",
            "delta_ler": 1e-4,
            # flops_per_syndrome intentionally omitted
            "n_params": 50_000,
            "branch": "exp/t/01-a", "commit_sha": "sha_a", "round_attempt_id": "u1",
        },
        verify_verdict="VERIFIED",
        verify_report={
            "verdict": "VERIFIED", "delta_vs_baseline_holdout": 1e-4,
            "ler_holdout": 4e-4, "paired_eval_bundle_id": "b1",
        },
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert pareto == []


def test_pareto_skips_verified_round_without_n_params(tmp_path: Path) -> None:
    """VERIFIED round missing n_params must NOT enter the archive (symmetric M1)."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1, "status": "ok", "hypothesis": "h",
            "delta_ler": 1e-4,
            "flops_per_syndrome": 100_000,
            # n_params intentionally omitted
            "branch": "exp/t/01-a", "commit_sha": "sha_a", "round_attempt_id": "u1",
        },
        verify_verdict="VERIFIED",
        verify_report={
            "verdict": "VERIFIED", "delta_vs_baseline_holdout": 1e-4,
            "ler_holdout": 4e-4, "paired_eval_bundle_id": "b1",
        },
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert pareto == []


def test_pareto_missing_cost_does_not_evict_real_candidate(tmp_path: Path) -> None:
    """A row missing cost axes must not displace an existing valid row.

    Regression for M1: the old ``_dominates`` turned missing axes into 0,
    so a row with delta≥existing, flops=None, n_params=None would
    "dominate" everything on disk. After the fix the bad row is dropped
    at admission time, keeping the true non-dominated front intact.
    """
    mem = RunMemory(tmp_path)
    # Valid candidate first.
    _admit(mem, round=1, delta_ler=1e-4, flops_per_syndrome=100_000, n_params=40_000,
           branch="exp/t/01-a", commit_sha="a1", round_attempt_id="u1")
    # Malformed VERIFIED row arrives — same delta, no cost fields.
    record_round(
        mem,
        round_metrics={
            "round": 2, "status": "ok", "hypothesis": "h",
            "delta_ler": 1e-4,
            # both cost fields absent
            "branch": "exp/t/02-b", "commit_sha": "b1", "round_attempt_id": "u2",
        },
        verify_verdict="VERIFIED",
        verify_report={
            "verdict": "VERIFIED", "delta_vs_baseline_holdout": 1e-4,
            "ler_holdout": 5e-4, "paired_eval_bundle_id": "b2",
        },
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert len(pareto) == 1
    assert pareto[0]["branch"] == "exp/t/01-a"


def test_pareto_skips_verified_round_without_verify_report(tmp_path: Path) -> None:
    """VERIFIED verdict without verify_report cannot be admitted — no holdout axis."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1, "status": "ok", "hypothesis": "h",
            "delta_ler": 1e-4, "flops_per_syndrome": 100_000, "n_params": 50_000,
            "branch": "exp/t/01-a", "commit_sha": "sha_a", "round_attempt_id": "u1",
        },
        verify_verdict="VERIFIED",
        verify_report=None,
    )
    # History row was still written.
    history_rows = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    # But Pareto was skipped because there's no holdout delta to compare.
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    assert pareto == []


def test_pareto_row_contains_worktree_provenance_fields(tmp_path: Path) -> None:
    """§15.7 superset: row must carry compose_mode, fork_from_ordered, etc."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1, "status": "ok", "hypothesis": "h",
            "delta_ler": 1e-4, "flops_per_syndrome": 100_000, "n_params": 50_000,
            "branch": "exp/t/05-compose", "commit_sha": "sha_c",
            "round_attempt_id": "u5",
            "fork_from": ["exp/t/02-a", "exp/t/03-b"],
            "fork_from_ordered": ["exp/t/02-a", "exp/t/03-b"],
            "compose_mode": "pure",
        },
        verify_verdict="VERIFIED",
        verify_report={
            "verdict": "VERIFIED", "delta_vs_baseline_holdout": 5e-4,
            "ler_holdout": 4e-4, "paired_eval_bundle_id": "b5",
        },
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text())
    row = pareto[0]
    assert row["compose_mode"] == "pure"
    assert row["fork_from"] == ["exp/t/02-a", "exp/t/03-b"]
    assert row["fork_from_ordered"] == ["exp/t/02-a", "exp/t/03-b"]


# ─── Regression tests for history/log/verdict-gating ────────────────────


def test_record_round_writes_history_row(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1,
            "status": "ok",
            "hypothesis": "gated_mlp + 3 layers",
            "delta_ler": 8e-5,
            "flops_per_syndrome": 10_000,
            "n_params": 5000,
            "checkpoint_path": "/tmp/round_1/checkpoint.pt",
        },
        verify_verdict="VERIFIED",
    )
    history = [
        json.loads(line)
        for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(history) == 1
    assert history[0]["round"] == 1
    assert history[0]["hypothesis"] == "gated_mlp + 3 layers"
    assert history[0]["delta_ler"] == 8e-5


def test_record_round_appends_log(tmp_path: Path) -> None:
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={"round": 3, "status": "ok", "hypothesis": "try attention"},
        verify_verdict="VERIFIED",
    )
    log_text = (tmp_path / "log.md").read_text(encoding="utf-8")
    assert "round 3" in log_text


def test_non_verified_rounds_skip_pareto(tmp_path: Path) -> None:
    """Only verify_verdict=='VERIFIED' admits into the Pareto archive."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 1,
            "status": "ok",
            "hypothesis": "useless",
            "delta_ler": -2e-5,
            "flops_per_syndrome": 10_000,
            "n_params": 5000,
        },
        verify_verdict="FAILED",
    )
    history_rows = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    assert pareto == []


def test_non_verified_rounds_skip_pareto_when_verdict_missing(tmp_path: Path) -> None:
    """A round without a verdict (e.g., compile_error) must not enter Pareto."""
    mem = RunMemory(tmp_path)
    record_round(
        mem,
        round_metrics={
            "round": 7,
            "status": "compile_error",
            "status_reason": "pydantic ValidationError on gnn.hidden_dim < 4",
            "hypothesis": "h7",
        },
        verify_verdict=None,
    )
    history_rows = (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    assert pareto == []


def test_pareto_rows_include_worktree_fields(tmp_path: Path) -> None:
    """§15.7 superset fields flow through into pareto.json rows."""
    mem = RunMemory(tmp_path)
    _admit(
        mem,
        round=1,
        branch="exp/t/01-a",
        commit_sha="a1",
        round_attempt_id="u1",
        fork_from="baseline",
    )
    pareto = json.loads((tmp_path / "pareto.json").read_text(encoding="utf-8"))
    assert len(pareto) == 1
    row = pareto[0]
    assert row["branch"] == "exp/t/01-a"
    assert row["commit_sha"] == "a1"
    assert row["round_attempt_id"] == "u1"
    assert row["fork_from"] == "baseline"
