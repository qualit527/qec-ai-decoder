from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


HELPER_SCRIPT = r"""
from __future__ import annotations

import os
import signal
from pathlib import Path

from autoqec.envs.schema import load_env_yaml
from autoqec.orchestration import llm_loop
from autoqec.runner.schema import RoundMetrics


run_dir = Path(os.environ["AUTOQEC_TEST_RUN_DIR"])
env_yaml = Path(os.environ["AUTOQEC_TEST_ENV_YAML"])
interrupt_after_round = os.environ.get("AUTOQEC_TEST_INTERRUPT_AFTER_ROUND")


def fake_invoke_subagent(role, prompt, timeout=300.0):
    if role == "ideator":
        return {
            "hypothesis": "try a resumable round",
            "fork_from": "baseline",
            "compose_mode": None,
            "rationale": "exercise resume bookkeeping",
        }
    if role == "coder":
        return {
            "dsl_config": {
                "type": "gnn",
                "output_mode": "soft_priors",
                "gnn": {
                    "layers": 1,
                    "hidden_dim": 16,
                    "message_fn": "mlp",
                    "aggregation": "sum",
                    "normalization": "layer",
                    "residual": False,
                    "edge_features": ["syndrome_bit"],
                },
                "head": "linear",
                "training": {
                    "learning_rate": 1e-3,
                    "batch_size": 64,
                    "epochs": 3,
                    "loss": "bce",
                    "profile": "dev",
                },
            },
            "tier": "1",
            "rationale": "small fake gnn",
            "commit_message": "feat(round): fake",
        }
    if role == "analyst":
        return {
            "summary_1line": "fake round complete",
            "verdict": "ignore",
            "next_hypothesis_seed": "continue",
        }
    raise AssertionError(role)


def fake_run_round(cfg, env):
    round_idx = int(Path(cfg.round_dir).name.removeprefix("round_"))
    print(f"EXECUTED_ROUND={round_idx}", flush=True)
    metrics = RoundMetrics(
        status="ok",
        delta_ler=0.001 * round_idx,
        flops_per_syndrome=100 + round_idx,
        n_params=10 + round_idx,
        round_attempt_id=cfg.round_attempt_id,
    )
    round_dir = Path(cfg.round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "metrics.json").write_text(
        metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return metrics


original_record_round = llm_loop.record_round


def interrupting_record_round(mem, round_metrics, verify_verdict=None, verify_report=None):
    original_record_round(mem, round_metrics, verify_verdict, verify_report)
    if str(round_metrics.get("round")) == interrupt_after_round:
        os.kill(os.getpid(), signal.SIGINT)


llm_loop.invoke_subagent = fake_invoke_subagent
llm_loop.run_round = fake_run_round
llm_loop.record_round = interrupting_record_round

env = load_env_yaml(env_yaml)
llm_loop.run_llm_loop(
    env=env,
    rounds=3,
    profile="dev",
    run_dir=run_dir,
    env_yaml_path=env_yaml,
    invocation_argv=["python", "-m", "cli.autoqec", "run", str(env_yaml)],
)
"""


def _run_helper(
    tmp_path: Path,
    run_dir: Path,
    env_yaml: Path,
    *,
    interrupt_after_round: int | None,
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "resume_helper.py"
    script.write_text(HELPER_SCRIPT, encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["AUTOQEC_TEST_RUN_DIR"] = str(run_dir)
    env["AUTOQEC_TEST_ENV_YAML"] = str(env_yaml)
    if interrupt_after_round is None:
        env.pop("AUTOQEC_TEST_INTERRUPT_AFTER_ROUND", None)
    else:
        env["AUTOQEC_TEST_INTERRUPT_AFTER_ROUND"] = str(interrupt_after_round)
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(repo_root)
    )
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        check=False,
    )


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGINT delivery differs on Windows; resume semantics are covered by unit helpers there.",
)
def test_ctrl_c_resume_skips_completed_rounds_without_duplicate_attempts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_yaml = repo_root / "autoqec/envs/builtin/surface_d5_depol.yaml"
    run_dir = tmp_path / "runs" / "resume-run"

    interrupted = _run_helper(
        tmp_path,
        run_dir,
        env_yaml,
        interrupt_after_round=2,
    )
    assert interrupted.returncode != 0
    assert "KeyboardInterrupt" in interrupted.stderr
    assert "EXECUTED_ROUND=1" in interrupted.stdout
    assert "EXECUTED_ROUND=2" in interrupted.stdout

    resumed = _run_helper(
        tmp_path,
        run_dir,
        env_yaml,
        interrupt_after_round=None,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "EXECUTED_ROUND=1" not in resumed.stdout
    assert "EXECUTED_ROUND=2" not in resumed.stdout
    assert "EXECUTED_ROUND=3" in resumed.stdout

    history_rows = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["round"] for row in history_rows] == [1, 2, 3]
    round_attempt_ids = [row["round_attempt_id"] for row in history_rows]
    assert len(round_attempt_ids) == len(set(round_attempt_ids))
