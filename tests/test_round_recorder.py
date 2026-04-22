"""Unit tests for the round recorder (B2)."""
from __future__ import annotations

import json
from pathlib import Path


def _mk_metrics(delta_ler: float, status: str = "ok", n_params: int = 5000, flops: int = 10000) -> dict:
    return {
        "status": status,
        "ler_plain_classical": 0.01394,
        "ler_predecoder": 0.01394 - delta_ler,
        "delta_ler": delta_ler,
        "delta_ler_ci_low": delta_ler - 1e-5,
        "delta_ler_ci_high": delta_ler + 1e-5,
        "flops_per_syndrome": flops,
        "n_params": n_params,
        "train_wallclock_s": 12.0,
        "eval_wallclock_s": 3.0,
        "vram_peak_gb": 0.0,
        "checkpoint_path": "/tmp/round_1/checkpoint.pt",
        "training_log_path": "/tmp/round_1/train.log",
    }


def test_record_round_writes_history_and_updates_pareto(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    record_round(
        mem=mem,
        round_idx=1,
        hypothesis="gated_mlp + 3 layers",
        dsl_config={"type": "gnn"},
        metrics=_mk_metrics(delta_ler=8e-5),
        verdict="candidate",
        summary_1line="round 1: gated_mlp gave Δ=8e-5 at 5k params; fastest so far.",
    )
    history = [json.loads(line) for line in (tmp_path / "run" / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(history) == 1
    row = history[0]
    assert row["round"] == 1
    assert row["hypothesis"] == "gated_mlp + 3 layers"
    assert row["verdict"] == "candidate"
    # metrics fields should be flattened in
    assert row["delta_ler"] == 8e-5
    assert row["n_params"] == 5000
    # pareto: the candidate joins (1 entry)
    pareto = json.loads((tmp_path / "run" / "pareto.json").read_text(encoding="utf-8"))
    assert len(pareto) == 1
    assert pareto[0]["round"] == 1
    assert pareto[0]["delta_ler"] == 8e-5


def test_pareto_deterministic_tiebreak_on_equal_metrics(tmp_path: Path) -> None:
    """Codex LOW: equal (delta_ler, flops, n_params) points must not rely on
    insertion order — tie-break on `round` so pareto.json is reproducible."""
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    metrics = _mk_metrics(delta_ler=1e-4, n_params=5000, flops=10000)
    # Record 3 candidates with identical metrics, different round indices
    for idx in (5, 2, 4):
        record_round(
            mem=mem,
            round_idx=idx,
            hypothesis=f"h{idx}",
            dsl_config={"type": "gnn"},
            metrics=dict(metrics),
            verdict="candidate",
            summary_1line=f"round {idx} tied",
        )
    pareto = json.loads((tmp_path / "run" / "pareto.json").read_text(encoding="utf-8"))
    assert [e["round"] for e in pareto] == [2, 4, 5]  # ascending round as final tie-break


def test_pareto_sorted_desc_by_delta_ler_and_capped_at_five(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    deltas = [1e-5, 9e-5, 3e-5, 7e-5, 5e-5, 11e-5, 2e-5]  # 7 candidates
    for idx, delta in enumerate(deltas, start=1):
        record_round(
            mem=mem,
            round_idx=idx,
            hypothesis=f"h{idx}",
            dsl_config={"type": "gnn"},
            metrics=_mk_metrics(delta_ler=delta),
            verdict="candidate",
            summary_1line=f"round {idx}",
        )
    pareto = json.loads((tmp_path / "run" / "pareto.json").read_text(encoding="utf-8"))
    assert len(pareto) == 5  # capped
    assert [entry["delta_ler"] for entry in pareto] == sorted(
        [1e-5, 9e-5, 3e-5, 7e-5, 5e-5, 11e-5, 2e-5], reverse=True
    )[:5]


def test_ignored_rounds_do_not_enter_pareto(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    record_round(
        mem=mem,
        round_idx=1,
        hypothesis="useless",
        dsl_config={"type": "gnn"},
        metrics=_mk_metrics(delta_ler=-2e-5, status="ok"),
        verdict="ignore",
        summary_1line="negative delta_ler",
    )
    # history gets the row; pareto stays empty
    history_rows = (tmp_path / "run" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(history_rows) == 1
    pareto = json.loads((tmp_path / "run" / "pareto.json").read_text(encoding="utf-8"))
    assert pareto == []


def test_record_round_also_appends_log_md(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    record_round(
        mem=mem,
        round_idx=3,
        hypothesis="try attention",
        dsl_config={"type": "gnn"},
        metrics=_mk_metrics(delta_ler=5e-5),
        verdict="candidate",
        summary_1line="round 3: attention variant; Δ=5e-5",
    )
    log_text = (tmp_path / "run" / "log.md").read_text(encoding="utf-8")
    assert "round 3" in log_text
    assert "attention" in log_text


def test_record_round_synthesises_summary_on_compile_error(tmp_path: Path) -> None:
    """Codex HIGH: skill step 6 says skip the Analyst on non-ok status.
    record_round must accept that — otherwise the error path stalls on the
    API it is supposed to handle."""
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    failed_metrics = {
        "status": "compile_error",
        "status_reason": "pydantic ValidationError on gnn.hidden_dim < 4",
        "ler_plain_classical": None,
        "ler_predecoder": None,
        "delta_ler": None,
        "n_params": None,
        "flops_per_syndrome": None,
        "checkpoint_path": None,
        "train_wallclock_s": 0.0,
        "eval_wallclock_s": 0.0,
        "vram_peak_gb": 0.0,
    }
    row = record_round(
        mem=mem,
        round_idx=7,
        hypothesis="h7",
        dsl_config={"type": "gnn"},
        metrics=failed_metrics,
        verdict="ignore",
        summary_1line=None,  # Analyst skipped
    )
    assert "compile_error" in row["summary_1line"]
    assert "pydantic" in row["summary_1line"] or "hidden_dim" in row["summary_1line"]
    log_text = (tmp_path / "run" / "log.md").read_text(encoding="utf-8")
    assert "round 7" in log_text
    # pareto must stay empty — failed rounds never enter
    pareto = json.loads((tmp_path / "run" / "pareto.json").read_text(encoding="utf-8"))
    assert pareto == []


def test_record_round_synthesises_summary_on_killed_by_safety(tmp_path: Path) -> None:
    from autoqec.orchestration.memory import RunMemory
    from autoqec.orchestration.round_recorder import record_round

    mem = RunMemory(tmp_path / "run")
    row = record_round(
        mem=mem,
        round_idx=1,
        hypothesis="too big",
        dsl_config={"type": "gnn"},
        metrics={
            "status": "killed_by_safety",
            "status_reason": "VRAM estimate 40.00 > free 20.00",
            "delta_ler": None,
        },
        verdict="ignore",
        summary_1line=None,
    )
    assert "killed_by_safety" in row["summary_1line"]
    assert "VRAM" in row["summary_1line"]
