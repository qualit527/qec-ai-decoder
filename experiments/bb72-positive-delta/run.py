from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoqec.envs.schema import load_env_yaml  # noqa: E402
from autoqec.runner.schema import RoundMetrics, RunnerConfig  # noqa: E402


DEFAULT_ENV_YAML = REPO_ROOT / "autoqec/envs/builtin/bb72_perf.yaml"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs"
CLAIM = "benchmark evidence, not a VERIFIED holdout claim"
RunRound = Callable[[RunnerConfig, Any], RoundMetrics]
run_round: RunRound | None = None


class BenchmarkFailure(RuntimeError):
    """Raised when the benchmark driver cannot execute the requested run."""


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-bb72-positive-delta"


def _round_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem.removeprefix("round_")
    prefix = stem.split("_", 1)[0]
    if prefix.isdigit():
        return (int(prefix), path.name)
    return (10**9, path.name)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise BenchmarkFailure(f"Config is not a mapping: {path}")
    return config


def _round_row(
    *,
    round_number: int,
    config_path: Path,
    round_dir: Path,
    metrics: RoundMetrics,
) -> dict[str, Any]:
    return {
        "round": round_number,
        "config": config_path.name,
        "round_dir": str(round_dir),
        "status": metrics.status,
        "status_reason": metrics.status_reason,
        "ler_plain_classical": metrics.ler_plain_classical,
        "ler_predecoder": metrics.ler_predecoder,
        "delta_ler": metrics.delta_ler,
        "delta_ler_ci_low": metrics.delta_ler_ci_low,
        "delta_ler_ci_high": metrics.delta_ler_ci_high,
        "flops_per_syndrome": metrics.flops_per_syndrome,
        "n_params": metrics.n_params,
        "train_wallclock_s": metrics.train_wallclock_s,
        "eval_wallclock_s": metrics.eval_wallclock_s,
        "checkpoint_path": metrics.checkpoint_path,
        "training_log_path": metrics.training_log_path,
    }


def _build_summary(*, run_id: str, env_yaml: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    best_delta: float | None = None
    best_round: int | None = None
    round_1_delta: float | None = None

    for row in rows:
        delta = row["delta_ler"]
        if row["round"] == 1:
            round_1_delta = delta
        if delta is not None and (best_delta is None or delta > best_delta):
            best_delta = delta
            best_round = row["round"]
        row["best_delta_ler_so_far"] = best_delta

    improvement = None
    if best_delta is not None and round_1_delta is not None:
        improvement = best_delta - round_1_delta

    return {
        "run_id": run_id,
        "env_yaml": str(env_yaml),
        "claim": CLAIM,
        "has_positive_delta": best_delta is not None and best_delta > 0,
        "best_round": best_round,
        "best_delta_ler": best_delta,
        "improvement_vs_round_1": improvement,
        "rounds": rows,
    }


def _write_report(summary: dict[str, Any], report_path: Path) -> None:
    lines = [
        "# BB72 Positive Delta Benchmark",
        "",
        f"Run ID: `{summary['run_id']}`",
        "",
        f"Claim: {summary['claim']}.",
        "",
        "| Round | Config | Status | Delta LER | Best Delta So Far |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summary["rounds"]:
        lines.append(
            "| {round} | `{config}` | {status} | {delta_ler} | {best_delta_ler_so_far} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Best round: {summary['best_round']}",
            f"Best delta LER: {summary['best_delta_ler']}",
            f"Improvement vs round 1: {summary['improvement_vs_round_1']}",
            f"Has positive delta: {summary['has_positive_delta']}",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark(env_yaml: Path, config_dir: Path, output_root: Path) -> Path:
    global run_round

    if run_round is None:
        from autoqec.runner.runner import run_round as imported_run_round

        run_round = imported_run_round

    env_yaml = Path(env_yaml)
    config_dir = Path(config_dir)
    output_root = Path(output_root)
    config_paths = sorted(config_dir.glob("round_*.yaml"), key=_round_sort_key)
    if not config_paths:
        raise BenchmarkFailure(f"No round_*.yaml configs found in {config_dir}")

    run_id = _run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    env = load_env_yaml(env_yaml)

    rows: list[dict[str, Any]] = []
    for round_number, config_path in enumerate(config_paths, start=1):
        round_dir = run_dir / f"round_{round_number}"
        config = RunnerConfig(
            env_name=env.name,
            predecoder_config=_load_config(config_path),
            training_profile="benchmark",
            seed=round_number,
            round_dir=str(round_dir),
            env_yaml_path=str(env_yaml),
        )
        metrics = run_round(config, env)
        rows.append(
            _round_row(
                round_number=round_number,
                config_path=config_path,
                round_dir=round_dir,
                metrics=metrics,
            )
        )

    summary = _build_summary(run_id=run_id, env_yaml=env_yaml, rows=rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(summary, run_dir / "report.md")
    return run_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-yaml", type=Path, default=DEFAULT_ENV_YAML)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    run_dir = run_benchmark(
        env_yaml=args.env_yaml,
        config_dir=args.config_dir,
        output_root=args.output_root,
    )
    print(f"BB72_POSITIVE_DELTA_RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
