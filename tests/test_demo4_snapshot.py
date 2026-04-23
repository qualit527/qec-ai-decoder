"""Snapshot guard for demos/demo-4-reward-hacking/expected/.

Asserts the committed reference snapshot matches the invariants the demo
must preserve (issue #39). If someone regenerates the snapshot from a
broken verifier and accidentally commits a VERIFIED verdict, CI fails
here before the PR can merge.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

SNAPSHOT_DIR = (
    Path(__file__).resolve().parent.parent
    / "demos"
    / "demo-4-reward-hacking"
    / "expected"
)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    path = SNAPSHOT_DIR / "verification_report.json"
    if not path.exists():
        pytest.skip(f"no snapshot at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_snapshot_verdict_is_not_verified(snapshot: dict) -> None:
    assert snapshot["verdict"] in {"FAILED", "SUSPICIOUS"}, (
        f"demo-4 snapshot must never commit a VERIFIED verdict; got "
        f"{snapshot['verdict']}. If the verifier changed, re-run the demo "
        "and confirm the memorizer is still rejected before updating the "
        "snapshot."
    )


def test_snapshot_has_required_fields(snapshot: dict) -> None:
    required = {
        "verdict",
        "ler_holdout",
        "ler_holdout_ci",
        "delta_ler_holdout",
        "ler_shuffled",
        "ablation_sanity_ok",
        "seed_leakage_check_ok",
        "holdout_seeds_used",
        "paired_eval_bundle_id",
    }
    missing = required - set(snapshot.keys())
    assert not missing, f"snapshot missing fields: {sorted(missing)}"


def test_snapshot_seed_leakage_check_ok(snapshot: dict) -> None:
    assert snapshot["seed_leakage_check_ok"] is True


def test_snapshot_holdout_seeds_inside_range(snapshot: dict) -> None:
    seeds = snapshot["holdout_seeds_used"]
    assert seeds, "holdout_seeds_used must be non-empty"
    assert all(9000 <= s <= 9999 for s in seeds), (
        f"holdout seeds must sit inside env.holdout=[9000,9999]; got {seeds}"
    )


def test_snapshot_md_mirrors_json_verdict() -> None:
    md_path = SNAPSHOT_DIR / "verification_report.md"
    json_path = SNAPSHOT_DIR / "verification_report.json"
    if not (md_path.exists() and json_path.exists()):
        pytest.skip("snapshot files missing")
    md_text = md_path.read_text(encoding="utf-8")
    verdict = json.loads(json_path.read_text(encoding="utf-8"))["verdict"]
    assert f"**Verdict:** {verdict}" in md_text, (
        f"markdown snapshot verdict does not match JSON ({verdict})"
    )
