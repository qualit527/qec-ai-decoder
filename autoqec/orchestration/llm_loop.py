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
from autoqec.orchestration.trace import append_note, append_section, init_trace
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig
from autoqec.tools.machine_state import machine_state


def run_llm_loop(
    env: EnvSpec,
    rounds: int,
    profile: str,
    run_dir: Path | str | None = None,
    env_yaml_path: str | None = None,
) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(run_dir) if run_dir else (Path("runs") / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    mem = RunMemory(run_dir)
    init_trace(
        run_dir,
        env_yaml=env_yaml_path or env.name,
        rounds=rounds,
        profile=profile,
    )

    for round_idx in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        # P1.4 will add a skip-if-complete check here.

        # --- Ideator ---
        ideator_ctx = mem.l3_for_ideator(
            env_spec=env.model_dump(),
            kb_excerpt="",
            machine_state=machine_state(run_dir),
            run_id=run_dir.name,
        )
        ideator_prompt = build_prompt("ideator", ideator_ctx)
        append_section(run_dir, round_idx, "ideator prompt", ideator_prompt)
        append_note(run_dir, round_idx, "dispatching autoqec-ideator")
        ideator_resp = invoke_subagent("ideator", ideator_prompt)
        append_section(run_dir, round_idx, "ideator response", ideator_resp)
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
        append_section(run_dir, round_idx, "coder prompt", coder_prompt)
        append_note(run_dir, round_idx, "dispatching autoqec-coder")
        coder_resp = invoke_subagent("coder", coder_prompt)
        append_section(run_dir, round_idx, "coder response", coder_resp)

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
        )
        append_note(
            run_dir, round_idx,
            f"invoking Runner (profile={profile}, seed={round_idx})",
        )
        metrics = run_round(cfg, env)
        append_section(run_dir, round_idx, "runner metrics", metrics.model_dump())

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
            append_section(run_dir, round_idx, "analyst prompt", analyst_prompt)
            append_note(run_dir, round_idx, "dispatching autoqec-analyst")
            analyst_resp = invoke_subagent("analyst", analyst_prompt)
            append_section(run_dir, round_idx, "analyst response", analyst_resp)

            if analyst_resp.get("verdict") == "candidate":
                append_note(run_dir, round_idx, "running independent verifier")
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
                    append_section(
                        run_dir, round_idx, "verifier report", verify_report,
                    )
                except Exception as exc:
                    # Never fail a whole round because the verifier crashed.
                    (round_dir / "verification_error.txt").write_text(str(exc))
                    verify_verdict = "FAILED"
                    append_section(
                        run_dir, round_idx, "verifier error", str(exc),
                    )

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
        append_note(
            run_dir, round_idx,
            f"round {round_idx} recorded (status={metrics.status}, "
            f"verify={verify_verdict or 'skipped'})",
        )

    append_note(run_dir, None, f"run complete — {rounds} round(s) in {run_dir.name}")
    # Final bookkeeping line for smoke scripts.
    print(f"AUTOQEC_RESULT_JSON={json.dumps({'run_dir': str(run_dir)})}")
    return run_dir


def _dsl_schema_md() -> str:
    from autoqec.decoders.dsl_schema import PredecoderDSL
    return json.dumps(PredecoderDSL.model_json_schema(), indent=2)
