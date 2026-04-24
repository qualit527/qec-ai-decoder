"""Headless live-LLM research loop — Python driver for `cli.autoqec run`.

Mirrors the `/autoqec-run` SKILL but dispatches subagents via
`autoqec.agents.cli_backend.invoke_subagent_with_metadata` (subprocess-spawn
codex / claude) instead of the Agent tool. Each round runs inside its own git
worktree so the Coder edits an isolated branch and the Runner observes that
branch's exact filesystem state.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from autoqec.agents.cli_backend import invoke_subagent_with_metadata
from autoqec.agents.dispatch import build_prompt
from autoqec.envs.schema import EnvSpec
from autoqec.eval.independent_eval import independent_verify
from autoqec.orchestration.loop import build_analyst_prompt, build_coder_prompt
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.round_recorder import record_round
from autoqec.orchestration.subprocess_runner import run_round_in_subprocess
from autoqec.orchestration.worktree import (
    cleanup_round_worktree,
    create_round_worktree,
)
from autoqec.runner.schema import RunnerConfig
from autoqec.tools.machine_state import machine_state


def _env_yaml_path(env: EnvSpec) -> str | None:
    builtin_path = Path(__file__).resolve().parents[1] / "envs" / "builtin" / f"{env.name}.yaml"
    return str(builtin_path) if builtin_path.exists() else None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def _log_json(mem: RunMemory, payload: dict[str, Any]) -> None:
    mem.append_log(json.dumps(payload, ensure_ascii=False))


def _worktree_fork_ref(fork_from: str) -> str:
    return "HEAD" if fork_from == "baseline" else fork_from


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
    repo_root = _repo_root()
    saw_non_baseline_parent = False

    for round_idx in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        # P1.4 will add a skip-if-complete check here.

        # --- Ideator ---
        machine_state_hint = machine_state(run_dir)
        _log_json(
            mem,
            {
                "tool_call": "machine_state",
                "round": round_idx,
                "run_dir": str(run_dir),
            },
        )
        ideator_ctx = mem.l3_for_ideator(
            env_spec=env.model_dump(),
            kb_excerpt="",
            machine_state=machine_state_hint,
            run_id=run_dir.name,
        )
        _write_json(round_dir / "ideator_prompt.json", ideator_ctx)
        ideator_prompt = build_prompt("ideator", ideator_ctx)
        ideator_resp, ideator_meta = invoke_subagent_with_metadata(
            "ideator",
            ideator_prompt,
            cwd=str(repo_root),
        )
        _write_json(
            round_dir / "ideator_response.json",
            {"payload": ideator_resp, "meta": ideator_meta},
        )
        if isinstance(ideator_resp.get("fork_from"), list):
            raise NotImplementedError(
                "compose rounds not supported in P0.1 LLM loop; "
                "land P1.1 (autoqec.orchestration.composer) first"
            )
        fork_from = str(ideator_resp.get("fork_from", "baseline"))
        if fork_from != "baseline":
            saw_non_baseline_parent = True

        worktree_plan = create_round_worktree(
            repo_root=repo_root,
            run_id=run_dir.name,
            round_idx=round_idx,
            slug=ideator_resp.get("hypothesis", f"round-{round_idx}"),
            fork_from=_worktree_fork_ref(fork_from),
        )

        try:
            # --- Coder ---
            coder_prompt = build_coder_prompt(
                hypothesis=ideator_resp,
                mem=mem,
                dsl_schema_md=_dsl_schema_md(),
                worktree_dir=worktree_plan["worktree_dir"],
            )
            _write_text(round_dir / "coder_prompt.txt", coder_prompt)
            coder_resp, coder_meta = invoke_subagent_with_metadata(
                "coder",
                coder_prompt,
                cwd=worktree_plan["worktree_dir"],
            )
            _write_json(
                round_dir / "coder_response.json",
                {"payload": coder_resp, "meta": coder_meta},
            )

            # --- Runner ---
            cfg = RunnerConfig(
                env_name=env.name,
                predecoder_config=coder_resp["dsl_config"],
                training_profile=profile,
                seed=round_idx,
                round_dir=str(round_dir),
                code_cwd=worktree_plan["worktree_dir"],
                branch=worktree_plan["branch"],
                round_attempt_id=str(uuid.uuid4()),
                fork_from=fork_from,
                commit_message=coder_resp.get("commit_message"),
                env_yaml_path=str(env_yaml_path) if env_yaml_path is not None else _env_yaml_path(env),
                invocation_argv=invocation_argv or ["python", "-m", "cli.autoqec", "run"],
            )
            metrics = run_round_in_subprocess(
                cfg,
                env,
                round_attempt_id=cfg.round_attempt_id,
            )

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
                _write_text(round_dir / "analyst_prompt.txt", analyst_prompt)
                analyst_resp, analyst_meta = invoke_subagent_with_metadata(
                    "analyst",
                    analyst_prompt,
                    cwd=str(repo_root),
                )
                _write_json(
                    round_dir / "analyst_response.json",
                    {"payload": analyst_resp, "meta": analyst_meta},
                )

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
        finally:
            cleanup_round_worktree(repo_root, worktree_plan["worktree_dir"])

    if rounds > 0 and not saw_non_baseline_parent:
        _log_json(
            mem,
            {"warning": "all_rounds_baseline_fork_from", "rounds": rounds},
        )

    # Final bookkeeping line for smoke scripts.
    print(f"AUTOQEC_RESULT_JSON={json.dumps({'run_dir': str(run_dir)})}")
    return run_dir


def _dsl_schema_md() -> str:
    from autoqec.decoders.dsl_schema import PredecoderDSL
    return json.dumps(PredecoderDSL.model_json_schema(), indent=2)
