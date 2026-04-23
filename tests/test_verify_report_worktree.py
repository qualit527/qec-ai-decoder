import pytest
from autoqec.eval.schema import VerifyReport


def test_verify_report_legacy_form():
    r = VerifyReport(
        verdict="VERIFIED",
        ler_holdout=1e-4,
        ler_holdout_ci=(5e-5, 2e-4),
        delta_ler_holdout=5e-5,
        ler_shuffled=1e-3,
        ablation_sanity_ok=True,
        holdout_seeds_used=[9000, 9001, 9002],
        seed_leakage_check_ok=True,
        notes="",
    )
    assert r.branch is None
    assert r.commit_sha is None


def test_verify_report_with_branch_requires_commit_sha():
    with pytest.raises(ValueError, match="commit_sha is required"):
        VerifyReport(
            verdict="VERIFIED",
            ler_holdout=1e-4,
            ler_holdout_ci=(5e-5, 2e-4),
            delta_ler_holdout=5e-5,
            ler_shuffled=1e-3,
            ablation_sanity_ok=True,
            holdout_seeds_used=[9000],
            seed_leakage_check_ok=True,
            notes="",
            branch="exp/20260422/02-a",
        )


def test_verify_report_paired_bundle_id_accepted():
    r = VerifyReport(
        verdict="VERIFIED",
        ler_holdout=1e-4,
        ler_holdout_ci=(5e-5, 2e-4),
        delta_ler_holdout=5e-5,
        ler_shuffled=1e-3,
        ablation_sanity_ok=True,
        holdout_seeds_used=[9000],
        seed_leakage_check_ok=True,
        notes="",
        branch="exp/20260422/02-a",
        commit_sha="abc123",
        delta_vs_baseline_holdout=5e-5,
        paired_eval_bundle_id="bundle-20260422-1500",
    )
    assert r.paired_eval_bundle_id == "bundle-20260422-1500"
