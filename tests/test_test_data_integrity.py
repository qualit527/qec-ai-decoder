"""Phase 1 test-data audit — pin the three fixtures every other test leans on.

- ``baseline_benchmark.json`` — the surface_d5 LER anchor (0.01394, seed 42,
  1 M shots). Must remain self-consistent and inside the plan's tolerance
  window; any drift invalidates comparisons downstream.
- ``autoqec/envs/builtin/surface_d5_depol.yaml`` — EnvSpec fields + MWPM
  backend + noise point at 5e-3 (the anchor noise level).
- ``autoqec/envs/builtin/bb72_depol.yaml`` — EnvSpec fields + OSD backend +
  OSD order 10 reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from autoqec.envs.schema import EnvSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_surface_d5_baseline_benchmark_is_self_consistent() -> None:
    path = REPO_ROOT / "demos/demo-1-surface-d5/expected_output/baseline_benchmark.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    # ler × n_shots must be an exact integer and equal n_errors.
    product = payload["ler"] * payload["n_shots"]
    assert abs(product - payload["n_errors"]) < 1e-6, (
        f"ler * n_shots != n_errors ({product} vs {payload['n_errors']})"
    )
    # The plan's ±5e-4 window around the anchor.
    assert 0.01344 <= payload["ler"] <= 0.01444, (
        f"baseline LER {payload['ler']} outside the [0.01344, 0.01444] window"
    )
    assert payload["seed"] == 42
    assert payload["n_shots"] == 1_000_000
    # Referenced circuit must exist.
    assert (REPO_ROOT / payload["circuit_path"]).exists(), (
        f"circuit missing: {payload['circuit_path']}"
    )


def test_surface_d5_env_yaml_contract() -> None:
    path = REPO_ROOT / "autoqec/envs/builtin/surface_d5_depol.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = EnvSpec.model_validate(raw)

    assert spec.name == "surface_d5_depol"
    assert spec.classical_backend == "mwpm"
    assert spec.code.type == "stim_circuit"
    # The 5e-3 anchor noise level must appear in the env noise sweep.
    assert 5.0e-3 in spec.noise.p, f"anchor noise 5e-3 missing from {spec.noise.p}"
    # Holdout / train / val seed ranges are disjoint (§10 isolation).
    assert spec.noise.seed_policy.train[1] < spec.noise.seed_policy.val[0]
    assert spec.noise.seed_policy.val[1] < spec.noise.seed_policy.holdout[0]
    # Referenced stim file exists.
    assert Path(spec.code.source).exists() or (REPO_ROOT / spec.code.source).exists(), (
        f"stim circuit not found: {spec.code.source}"
    )


def test_bb72_env_yaml_contract() -> None:
    path = REPO_ROOT / "autoqec/envs/builtin/bb72_depol.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec = EnvSpec.model_validate(raw)

    assert spec.name == "bb72_depol"
    assert spec.classical_backend == "osd"
    assert spec.code.type == "parity_check_matrix"
    # OSD order 10 must be reported for qLDPC benchmarking.
    assert 10 in spec.eval_protocol.osd_orders_reported, (
        f"OSD order 10 missing from {spec.eval_protocol.osd_orders_reported}"
    )
    # Seed ranges disjoint (same §10 guarantee).
    assert spec.noise.seed_policy.train[1] < spec.noise.seed_policy.val[0]
    assert spec.noise.seed_policy.val[1] < spec.noise.seed_policy.holdout[0]
