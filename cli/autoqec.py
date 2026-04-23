from __future__ import annotations

from importlib import resources
import json
import os
import random
import subprocess
import time
from pathlib import Path

import click
import yaml

from autoqec.envs.schema import EnvSpec, load_env_yaml
from autoqec.orchestration.memory import RunMemory
from autoqec.orchestration.subprocess_runner import (
    AUTOQEC_CHILD_BRANCH,
    AUTOQEC_CHILD_CODE_CWD,
    AUTOQEC_CHILD_COMPOSE_MODE,
    AUTOQEC_CHILD_CONFIG_YAML,
    AUTOQEC_CHILD_ENV_YAML,
    AUTOQEC_CHILD_FORK_FROM,
    AUTOQEC_CHILD_PROFILE,
    AUTOQEC_CHILD_ROUND_ATTEMPT_ID,
    AUTOQEC_CHILD_ROUND_DIR,
)
from autoqec.runner.schema import RoundMetrics, RunnerConfig


@click.group()
def main() -> None:
    """AutoQEC CLI."""


RESULT_PREFIX = "AUTOQEC_RESULT_JSON="


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _first_matching_line(candidates: list[str], predicate) -> str | None:
    for text in candidates:
        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            if line and predicate(line.lower()):
                return line
    return None


def _diagnose_failure_signature(metrics: dict | None, train_log_text: str, config_text: str) -> tuple[str, list[str]]:
    status_reason = ""
    if metrics is not None:
        status_reason = str(metrics.get("status_reason") or "")
    candidates = [status_reason, train_log_text, config_text]

    oom_signal = _first_matching_line(
        candidates,
        lambda line: "out of memory" in line or "oom" in line or "vram" in line,
    )
    if oom_signal is not None:
        return "oom", [oom_signal]

    nan_signal = _first_matching_line(
        candidates,
        lambda line: "nan" in line,
    )
    if nan_signal is not None:
        return "nan_loss", [nan_signal]

    degenerate_signal = _first_matching_line(
        candidates,
        lambda line: "p = 0" in line or "p=0" in line or "degenerate" in line,
    )
    if degenerate_signal is not None:
        return "degenerate_p_zero", [degenerate_signal]

    return "unknown", []
def _write_round_metrics(round_dir: str, metrics: RoundMetrics) -> RoundMetrics:
    metrics_path = Path(round_dir).resolve() / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return metrics


def _git_head_commit(cwd: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        text=True,
    ).strip()


def _enrich_local_worktree_metrics(
    metrics: RoundMetrics,
    *,
    round_dir: str,
    code_cwd: str,
    branch: str,
    fork_from: str | list[str] | None,
    compose_mode: str | None,
    round_attempt_id: str | None,
) -> RoundMetrics:
    updates = {
        "fork_from": fork_from,
        "compose_mode": compose_mode,
        "round_attempt_id": round_attempt_id or metrics.round_attempt_id,
    }
    if metrics.status == "compose_conflict":
        return _write_round_metrics(
            round_dir,
            metrics.model_copy(update=updates),
        )

    try:
        commit_sha = _git_head_commit(code_cwd)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        failure_metrics = metrics.model_copy(
            update={
                "status": "train_error",
                "status_reason": f"git rev-parse HEAD failed for branch {branch!r}: {exc}",
                "branch": None,
                "commit_sha": None,
                **updates,
            }
        )
        _write_round_metrics(round_dir, failure_metrics)
        raise

    return _write_round_metrics(
        round_dir,
        metrics.model_copy(
            update={
                "branch": branch,
                "commit_sha": commit_sha,
                **updates,
            }
        ),
    )


def load_example_templates() -> list[tuple[str, dict]]:
    template_root = resources.files("autoqec").joinpath("example_db")
    try:
        template_files = sorted(
            [entry for entry in template_root.iterdir() if entry.name.endswith(".yaml")],
            key=lambda entry: entry.name,
        )
    except FileNotFoundError as exc:
        raise click.ClickException(
            "Example templates are not available in this install. "
            "Reinstall from a source checkout or include autoqec example_db package data."
        ) from exc

    templates = [
        (Path(entry.name).stem, yaml.safe_load(entry.read_text(encoding="utf-8")))
        for entry in template_files
    ]
    if not templates:
        raise click.ClickException(
            "No example templates were found under autoqec/example_db. "
            "This install is missing packaged demo templates."
        )
    return templates


def _candidate_pareto(records: list[dict]) -> list[dict]:
    candidates = []
    for idx, record in enumerate(records, start=1):
        if record.get("status") != "ok":
            continue
        if any(record.get(key) is None for key in ("delta_ler", "flops_per_syndrome", "n_params")):
            continue
        candidates.append(
            {
                "round": int(record.get("round", idx)),
                "delta_ler": float(record["delta_ler"]),
                "flops_per_syndrome": int(record["flops_per_syndrome"]),
                "n_params": int(record["n_params"]),
                "checkpoint_path": record.get("checkpoint_path"),
                "verified": False,
            }
        )

    front: list[dict] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            no_worse = (
                other["delta_ler"] >= cand["delta_ler"]
                and other["flops_per_syndrome"] <= cand["flops_per_syndrome"]
                and other["n_params"] <= cand["n_params"]
            )
            strictly_better = (
                other["delta_ler"] > cand["delta_ler"]
                or other["flops_per_syndrome"] < cand["flops_per_syndrome"]
                or other["n_params"] < cand["n_params"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(cand)

    front.sort(key=lambda item: (-item["delta_ler"], item["flops_per_syndrome"], item["n_params"], item["round"]))

    unique_front: list[dict] = []
    seen = set()
    for item in front:
        key = (item["delta_ler"], item["flops_per_syndrome"], item["n_params"])
        if key in seen:
            continue
        seen.add(key)
        unique_front.append(item)
    return unique_front


def _load_round_metrics_for_verify(round_dir: Path) -> dict | None:
    metrics_path = round_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("round") is None and round_dir.name.startswith("round_"):
        suffix = round_dir.name.removeprefix("round_")
        if suffix.isdigit():
            metrics["round"] = int(suffix)
    return metrics


def _parse_fork_from_option(fork_from: str | None) -> str | list[str] | None:
    parsed_fork_from: str | list[str] | None = None
    if fork_from is not None:
        if fork_from.strip().startswith("["):
            try:
                parsed = json.loads(fork_from)
            except json.JSONDecodeError as e:
                raise click.BadParameter(
                    f"--fork-from looks like JSON but failed to parse: {e}"
                ) from e
            if not isinstance(parsed, list) or not all(
                isinstance(x, str) for x in parsed
            ):
                raise click.BadParameter(
                    "--fork-from JSON must be a list of strings"
                )
            parsed_fork_from = parsed
        else:
            parsed_fork_from = fork_from
    return parsed_fork_from


def _run_round_impl(
    env_yaml: str,
    config_yaml: str,
    round_dir: str,
    profile: str,
    code_cwd: str | None,
    branch: str | None,
    fork_from: str | None,
    compose_mode: str | None,
    round_attempt_id: str | None,
    _internal_execute_locally: bool,
) -> None:
    env = load_env_yaml(env_yaml)
    with open(config_yaml, encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    parsed_fork_from = _parse_fork_from_option(fork_from)

    # Worktree-path runs go through subprocess_runner; in-process runs use the legacy Runner.
    # ``_internal_execute_locally`` is set by the child hop (run-round-internal)
    # so the child can skip the subprocess dispatcher.
    if code_cwd is not None and not _internal_execute_locally:
        cfg = RunnerConfig(
            env_name=env.name,
            predecoder_config=cfg_dict,
            training_profile=profile,
            seed=0,
            round_dir=round_dir,
            code_cwd=code_cwd,
            branch=branch,
            fork_from=parsed_fork_from,
            compose_mode=compose_mode,
        )
        from autoqec.orchestration.subprocess_runner import run_round_in_subprocess

        metrics = run_round_in_subprocess(cfg, env, round_attempt_id=round_attempt_id)
    else:
        # Child hop (or plain in-process invocation): strip code_cwd so the
        # in-process Runner's §15.8 guard does not fire. Parent subprocess_runner
        # already pinned cwd + PYTHONPATH before spawning us; branch /
        # fork_from / compose_mode / round_attempt_id still flow through so
        # the metrics row carries full provenance.
        cfg = RunnerConfig(
            env_name=env.name,
            predecoder_config=cfg_dict,
            training_profile=profile,
            seed=0,
            round_dir=round_dir,
            code_cwd=None,
            branch=branch,
            fork_from=parsed_fork_from,
            compose_mode=compose_mode,
        )
        from autoqec.runner.runner import run_round

        metrics = run_round(cfg, env)
        if round_attempt_id is not None and metrics.round_attempt_id is None:
            metrics = metrics.model_copy(update={"round_attempt_id": round_attempt_id})
        if _internal_execute_locally and code_cwd is not None and branch is not None:
            metrics = _enrich_local_worktree_metrics(
                metrics,
                round_dir=round_dir,
                code_cwd=code_cwd,
                branch=branch,
                fork_from=parsed_fork_from,
                compose_mode=compose_mode,
                round_attempt_id=round_attempt_id,
            )
    click.echo(metrics.model_dump_json(indent=2))


@main.command(name="run-round")
@click.argument("env_yaml")
@click.argument("config_yaml")
@click.argument("round_dir")
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option(
    "--code-cwd",
    default=None,
    help="Absolute path to a worktree checkout; when set, Runner runs in that cwd",
)
@click.option(
    "--branch",
    default=None,
    help="Branch name (exp/<run_id>/<N>-<slug>); required when --code-cwd is set",
)
@click.option(
    "--fork-from",
    default=None,
    help='Parent branch name, or JSON list like \'["exp/.../a", "exp/.../b"]\' for compose rounds',
)
@click.option(
    "--compose-mode",
    type=click.Choice(["pure", "with_edit"]),
    default=None,
    help="Required when --fork-from is a list",
)
@click.option(
    "--round-attempt-id",
    default=None,
    help="UUID minted at Ideator emit-time; required on the worktree path",
)
@click.option(
    "--_internal-execute-locally",
    "_internal_execute_locally",
    is_flag=True,
    default=False,
    hidden=True,
    help=(
        "Internal: skip subprocess dispatch and run in-process even when "
        "--code-cwd is set. subprocess_runner sets this on the child argv "
        "to prevent run_round_cmd from re-entering itself."
    ),
)
def run_round_cmd(
    env_yaml: str,
    config_yaml: str,
    round_dir: str,
    profile: str,
    code_cwd: str | None,
    branch: str | None,
    fork_from: str | None,
    compose_mode: str | None,
    round_attempt_id: str | None,
    _internal_execute_locally: bool,
) -> None:
    _run_round_impl(
        env_yaml=env_yaml,
        config_yaml=config_yaml,
        round_dir=round_dir,
        profile=profile,
        code_cwd=code_cwd,
        branch=branch,
        fork_from=fork_from,
        compose_mode=compose_mode,
        round_attempt_id=round_attempt_id,
        _internal_execute_locally=_internal_execute_locally,
    )


@main.command(name="run-round-internal", hidden=True)
def run_round_internal_cmd() -> None:
    """Internal subprocess entrypoint using env vars instead of dynamic argv."""
    env_yaml = os.environ[AUTOQEC_CHILD_ENV_YAML]
    config_yaml = os.environ[AUTOQEC_CHILD_CONFIG_YAML]
    round_dir = os.environ[AUTOQEC_CHILD_ROUND_DIR]
    profile = os.environ[AUTOQEC_CHILD_PROFILE]
    code_cwd = os.environ[AUTOQEC_CHILD_CODE_CWD]
    branch = os.environ[AUTOQEC_CHILD_BRANCH]
    fork_from = os.environ.get(AUTOQEC_CHILD_FORK_FROM)
    compose_mode = os.environ.get(AUTOQEC_CHILD_COMPOSE_MODE)
    round_attempt_id = os.environ.get(AUTOQEC_CHILD_ROUND_ATTEMPT_ID)

    _run_round_impl(
        env_yaml=env_yaml,
        config_yaml=config_yaml,
        round_dir=round_dir,
        profile=profile,
        code_cwd=code_cwd,
        branch=branch,
        fork_from=fork_from,
        compose_mode=compose_mode,
        round_attempt_id=round_attempt_id,
        _internal_execute_locally=True,
    )


@main.command()
@click.argument("env_yaml")
@click.option("--rounds", type=int, default=10)
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option("--no-llm", is_flag=True, help="Pick random seed templates instead of calling subagents")
def run(env_yaml: str, rounds: int, profile: str, no_llm: bool) -> None:
    from autoqec.runner.runner import run_round

    env = load_env_yaml(env_yaml)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (Path("runs") / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    mem = RunMemory(run_dir, pareto_filename="candidate_pareto.json")
    templates = load_example_templates()
    dev_safe_templates = {"gnn_small", "gnn_gated", "neural_bp_min"}
    history: list[dict] = []
    for round_idx in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_idx}"
        if no_llm:
            candidates = templates
            if profile == "dev":
                candidates = [item for item in templates if item[0] in dev_safe_templates]
            if not candidates:
                raise click.ClickException(
                    f"No bundled templates are available for profile={profile!r}. "
                    "This install is missing the expected demo template assets."
                )
            _, cfg_dict = random.choice(candidates)
        else:
            raise click.ClickException("LLM mode is not wired in this branch yet; use --no-llm")
        cfg = RunnerConfig(
            env_name=env.name,
            predecoder_config=cfg_dict,
            training_profile=profile,
            seed=round_idx,
            round_dir=str(round_dir),
        )
        metrics = run_round(cfg, env)
        record = metrics.model_dump()
        record["round"] = round_idx
        history.append(record)
        mem.append_round(record)
        mem.update_pareto(_candidate_pareto(history))
        click.echo(f"Round {round_idx}: {metrics.status} Δ={metrics.delta_ler}")
    (run_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    click.echo(
        f"{RESULT_PREFIX}"
        + json.dumps(
            {
                "run_dir": str(run_dir),
                "rounds": rounds,
                "candidate_pareto_path": str(run_dir / "candidate_pareto.json"),
            },
            ensure_ascii=False,
        )
    )


@main.command()
@click.option("--out", required=True)
@click.option("--name", prompt="Env name (slug)", type=str)
@click.option("--code-source", prompt="Code source path (.stim / .npy)", type=str)
@click.option("--noise-p", prompt="Noise p values, comma-separated", default="1e-3,5e-3,1e-2")
@click.option("--backend", prompt="Classical backend", type=click.Choice(["mwpm", "osd"]), default="mwpm")
def add_env(out: str, name: str, code_source: str, noise_p: str, backend: str) -> None:
    ps = [float(item.strip()) for item in noise_p.split(",") if item.strip()]
    code_type = "stim_circuit" if code_source.endswith(".stim") else "parity_check_matrix"
    data = {
        "name": name,
        "code": {"type": code_type, "source": code_source},
        "noise": {
            "type": "depolarizing",
            "p": ps,
            "seed_policy": {"train": [1, 999], "val": [1000, 1999], "holdout": [9000, 9999]},
        },
        "constraints": {
            "latency_flops_budget": 10000000,
            "param_budget": 200000,
            "target_ler": 1e-4,
            "target_p": min(ps),
        },
        "baseline_decoders": ["pymatching"] if backend == "mwpm" else ["bposd"],
        "classical_backend": backend,
        "eval_protocol": {
            "min_shots_train": 1000000,
            "min_shots_val": 100000,
            "min_shots_verify": 200000,
            "bootstrap_ci": 0.95,
            "osd_orders_reported": [0] if backend == "mwpm" else [0, 10],
            "x_z_decoding": "circuit",
        },
    }
    EnvSpec(**data)
    with Path(out).open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    click.echo(f"Wrote {out}")


@main.command()
@click.argument("round_dir")
@click.option("--env", required=True, help="Env YAML path")
@click.option("--n-shots", type=int, default=None)
@click.option("--n-seeds", type=int, default=50, help="Number of holdout seeds to sample")
def verify(round_dir: str, env: str, n_shots: int | None, n_seeds: int) -> None:
    from autoqec.eval.independent_eval import independent_verify
    from autoqec.orchestration.round_recorder import admit_verified_round_to_pareto

    rd = Path(round_dir)
    env_spec = load_env_yaml(env)
    ckpt = rd / "checkpoint.pt"
    holdout = list(range(env_spec.noise.seed_policy.holdout[0],
                          env_spec.noise.seed_policy.holdout[1] + 1))[:n_seeds]
    report = independent_verify(ckpt, env_spec, holdout_seeds=holdout, n_shots=n_shots)
    (rd / "verification_report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    md_lines = [
        "# Verification Report",
        "",
        f"**Verdict:** {report.verdict}",
        "",
        f"- Holdout LER: {report.ler_holdout:.4g}",
        f"- Holdout LER CI: {report.ler_holdout_ci}",
        f"- Δ_LER (holdout): {report.delta_ler_holdout:.4g}",
        f"- Ablation sanity: {report.ablation_sanity_ok}",
        f"- Seed-leakage check: {report.seed_leakage_check_ok}",
    ]
    if report.paired_eval_bundle_id is not None:
        md_lines.append(f"- Paired eval bundle ID: {report.paired_eval_bundle_id}")
    md_lines.extend(
        [
            "",
            f"Notes: {report.notes}",
            "",
        ]
    )
    (rd / "verification_report.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )

    round_metrics = _load_round_metrics_for_verify(rd)
    if report.verdict == "VERIFIED" and round_metrics is not None:
        admit_verified_round_to_pareto(
            RunMemory(rd.parent),
            round_metrics,
            report.model_dump(),
        )

    click.echo(report.verdict)


@main.command(name="review-log")
@click.argument("run_dir")
def review_log(run_dir: str) -> None:
    rd = Path(run_dir)
    hist_path = rd / "history.jsonl"
    pareto_path = rd / "pareto.json"
    if not pareto_path.exists():
        pareto_path = rd / "candidate_pareto.json"
    if not hist_path.exists():
        click.echo("No history.jsonl")
        return
    with hist_path.open() as f:
        rounds = [json.loads(line) for line in f if line.strip()]
    pareto = json.loads(pareto_path.read_text()) if pareto_path.exists() else []
    killed = sum(1 for r in rounds if r.get("status") == "killed_by_safety")
    stats = {
        "n_rounds": len(rounds),
        "n_pareto": len(pareto),
        "n_killed_by_safety": killed,
        "mean_wallclock_s": sum(r.get("train_wallclock_s", 0) for r in rounds) / max(len(rounds), 1),
        "top_hypotheses": [r.get("hypothesis", "")[:80] for r in rounds[-5:]],
    }
    click.echo(json.dumps(stats, indent=2))


@main.command()
@click.argument("run_dir")
def diagnose(run_dir: str) -> None:
    rd = Path(run_dir)
    # Accept either a run_dir (contains round_*) or a round_dir directly
    if (rd / "metrics.json").exists() or (rd / "train.log").exists():
        target = rd
    else:
        round_dirs = sorted(rd.glob("round_*"))
        if not round_dirs:
            click.echo("No round dirs found and no metrics in given path")
            return
        target = round_dirs[-1]
    metrics_path = target / "metrics.json"
    train_log_path = target / "train.log"
    config_path = target / "config.yaml"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    train_log_text = _read_text_if_exists(train_log_path)
    config_text = _read_text_if_exists(config_path)
    root_cause, signals = _diagnose_failure_signature(metrics, train_log_text, config_text)

    out: dict = {
        "path": str(target),
        "root_cause": root_cause,
        "signals": signals,
        "read_only": True,
    }
    for fn in ("config.yaml", "metrics.json", "train.log"):
        p = target / fn
        out[f"has_{fn}"] = p.exists()
        if fn == "metrics.json" and p.exists():
            out["metrics"] = metrics
    click.echo(json.dumps(out, indent=2))

if __name__ == "__main__":  # pragma: no cover
    main()
