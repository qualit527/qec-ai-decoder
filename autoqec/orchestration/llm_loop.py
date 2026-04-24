"""Headless live-LLM research loop — Python driver for `cli.autoqec run`.

Mirrors the `/autoqec-run` SKILL but dispatches subagents via
`autoqec.agents.cli_backend.invoke_subagent` (subprocess-spawn codex / claude)
instead of the Agent tool. Runs the Runner inline (not in a worktree) by
default; worktree scheduling + compose rounds land in P1.1.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from autoqec.agents.cli_backend import invoke_subagent
from autoqec.agents.dispatch import build_prompt
from autoqec.envs.schema import EnvSpec
from autoqec.eval.independent_eval import independent_verify
from autoqec.orchestration.loop import build_analyst_prompt, build_coder_prompt
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig
from autoqec.tools.machine_state import machine_state


def _env_yaml_path(env: EnvSpec) -> str | None:
    builtin_path = Path(__file__).resolve().parents[1] / "envs" / "builtin" / f"{env.name}.yaml"
    return str(builtin_path) if builtin_path.exists() else None


def _history_rows_by_round(run_dir: Path) -> dict[int, dict[str, Any]]:
    history_path = run_dir / "history.jsonl"
    if not history_path.exists():
        return {}

    rows_by_round: dict[int, dict[str, Any]] = {}
    for line in history_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        round_idx = row.get("round")
        if isinstance(round_idx, int):
            rows_by_round[round_idx] = row
    return rows_by_round


def _parse_metrics(round_dir: Path) -> dict[str, Any] | None:
    metrics_path = round_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return metrics if isinstance(metrics, dict) else None


def _round_is_complete(run_dir: Path, round_idx: int, history_rows: dict[int, dict[str, Any]]) -> bool:
    """Return True when a previous invocation durably completed ``round_idx``.

    Resume is gated by both the append-only orchestration record
    (``history.jsonl`` row with ``round == N``) and the Runner artifact
    (``round_N/metrics.json``). If both surfaces carry ``round_attempt_id`` they
    must match, preventing a stale metrics file from making a different attempt
    look complete. No explicit marker file is used.
    """
    history_row = history_rows.get(round_idx)
    if history_row is None:
        return False

    metrics = _parse_metrics(run_dir / f"round_{round_idx}")
    if metrics is None:
        return False

    history_attempt_id = history_row.get("round_attempt_id")
    metrics_attempt_id = metrics.get("round_attempt_id")
    if (
        isinstance(history_attempt_id, str)
        and history_attempt_id
        and isinstance(metrics_attempt_id, str)
        and metrics_attempt_id
    ):
        return history_attempt_id == metrics_attempt_id
    return True


def run_llm_loop(
    env: EnvSpec,
    rounds: int,
    profile: str,
    run_dir: Path | str | None = None,
    *,
    env_yaml_path: Path | str | None = None,
    invocation_argv: list[str] | None = None,
) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir) if run_dir else (Path("runs") / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    mem = RunMemory(run_dir)
    completed_rounds = _history_rows_by_round(run_dir)

    for round_idx in range(1, rounds + 1):
        if _round_is_complete(run_dir, round_idx, completed_rounds):
            print(f"Skipping completed round {round_idx}")
            continue

        round_dir = run_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # --- Ideator ---
        ideator_ctx = mem.l3_for_ideator(
            env_spec=env.model_dump(),
            kb_excerpt="",
            machine_state=machine_state(run_dir),
            run_id=run_dir.name,
        )
        ideator_resp = invoke_subagent("ideator", build_prompt("ideator", ideator_ctx))
        if isinstance(ideator_resp.get("fork_from"), list):
            raise NotImplementedError(
                "compose rounds not supported in P0.1 LLM loop; "
                "land P1.1 (autoqec.orchestration.composer) first"
            )

        # --- Coder ---
        coder_prompt = build_coder_prompt(
            hypothesis=ideator_resp, mem=mem,
            dsl_schema_md=_dsl_schema_md(),
        )
        coder_resp = invoke_subagent("coder", coder_prompt)

        # --- Runner ---
        cfg = RunnerConfig(
            env_name=env.name,
            predecoder_config=coder_resp["dsl_config"],
            training_profile=profile,
            seed=round_idx,
            round_dir=str(round_dir),
            round_attempt_id=str(uuid.uuid4()),
            fork_from=ideator_resp.get("fork_from", "baseline"),
            commit_message=coder_resp.get("commit_message"),
            env_yaml_path=str(env_yaml_path) if env_yaml_path is not None else _env_yaml_path(env),
            invocation_argv=invocation_argv or ["python", "-m", "cli.autoqec", "run"],
        )
        metrics = run_round(cfg, env)

        # --- Analyst + Verifier (SKILL parity §9) ---
        # On runner failure: skip both Analyst and Verifier; record_round
        # synthesises a fallback summary from status / status_reason.
        # On runner success: Analyst first, then Verifier ONLY if Analyst
        # verdict is "candidate" (compose_conflict is also skipped via the
        # status == "ok" check).
        verify_verdict: str | None = None
        verify_report: dict[str, Any] | None = None
        analyst_resp: dict[str, Any] | None = None

        if metrics.status == "ok":
            analyst_prompt = build_analyst_prompt(
                mem=mem, round_dir=round_dir, prev_summary="",
            )
            analyst_resp = invoke_subagent("analyst", analyst_prompt)

            if analyst_resp.get("verdict") == "candidate":
                try:
                    sp = env.noise.seed_policy
                    holdout = list(range(sp.holdout[0], sp.holdout[1] + 1))[:50]
                    report = independent_verify(
                        checkpoint=round_dir / "checkpoint.pt",
                        env_spec=env,
                        holdout_seeds=holdout,
                    )
                    (round_dir / "verification_report.json").write_text(
                        report.model_dump_json(indent=2), encoding="utf-8",
                    )
                    verify_verdict = report.verdict
                    verify_report = report.model_dump()
                except Exception as exc:
                    # Never fail a whole round because the verifier crashed.
                    (round_dir / "verification_error.txt").write_text(str(exc))
                    verify_verdict = "FAILED"

        # --- Record ---
        record = metrics.model_dump()
        record["round"] = round_idx
        record["hypothesis"] = ideator_resp.get("hypothesis")
        if analyst_resp is not None:
            record["verdict"] = analyst_resp.get("verdict")
            record["summary_1line"] = analyst_resp.get("summary_1line")
        record_round(
            mem,
            round_metrics=record,
            verify_verdict=verify_verdict,
            verify_report=verify_report,
        )
        completed_rounds[round_idx] = record

    # Final bookkeeping line for smoke scripts.
    print(f"AUTOQEC_RESULT_JSON={json.dumps({'run_dir': str(run_dir)})}")
    return run_dir


def _dsl_schema_md() -> str:
    from autoqec.decoders.dsl_schema import PredecoderDSL
    return json.dumps(PredecoderDSL.model_json_schema(), indent=2)
