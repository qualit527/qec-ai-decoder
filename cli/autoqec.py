from __future__ import annotations

import json
import random
import subprocess
import time
from pathlib import Path

import click
import yaml

from autoqec.envs.schema import EnvSpec, load_env_yaml
from autoqec.runner.schema import RunnerConfig


@click.group()
def main() -> None:
    """AutoQEC CLI."""


def _git_head_commit(cwd: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        text=True,
    ).strip()


def _enrich_local_worktree_metrics(
    metrics,
    *,
    round_dir: str,
    code_cwd: str,
    branch: str,
    fork_from: str | list[str] | None,
    compose_mode: str | None,
    round_attempt_id: str | None,
):
    enriched = metrics.model_copy(
        update={
            "branch": branch,
            "commit_sha": _git_head_commit(code_cwd),
            "fork_from": fork_from,
            "compose_mode": compose_mode,
            "round_attempt_id": round_attempt_id or metrics.round_attempt_id,
        }
    )
    metrics_path = Path(round_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        enriched.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return enriched


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
    env = load_env_yaml(env_yaml)
    with open(config_yaml) as f:
        cfg_dict = yaml.safe_load(f)

    # Parse fork_from: JSON list -> list[str]; bare string -> str.
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

    # Worktree-path runs go through subprocess_runner; in-process runs use the legacy Runner.
    # --_internal-execute-locally is the recursion guard: when subprocess_runner spawns us,
    # it sets this flag so this branch collapses back to the in-process Runner.
    if code_cwd is not None and not _internal_execute_locally:
        from autoqec.orchestration.subprocess_runner import run_round_in_subprocess

        metrics = run_round_in_subprocess(cfg, env, round_attempt_id=round_attempt_id)
    else:
        from autoqec.runner.runner import run_round

        local_cfg = cfg
        if _internal_execute_locally and code_cwd is not None:
            local_cfg = cfg.model_copy(update={"code_cwd": None})
        metrics = run_round(local_cfg, env)
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
    templates = sorted(Path("autoqec/example_db").glob("*.yaml"))
    dev_safe_templates = {"gnn_small", "gnn_gated", "neural_bp_min"}
    history = []
    for round_idx in range(1, rounds + 1):
        round_dir = run_dir / f"round_{round_idx}"
        if no_llm:
            candidates = templates
            if profile == "dev":
                candidates = [path for path in templates if path.stem in dev_safe_templates]
            template = random.choice(candidates)
            cfg_dict = yaml.safe_load(template.read_text())
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
        history.append(metrics.model_dump())
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(metrics.model_dump_json() + "\n")
        click.echo(f"Round {round_idx}: {metrics.status} Δ={metrics.delta_ler}")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    click.echo(json.dumps({"run_dir": str(run_dir), "rounds": rounds}, indent=2))


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

    rd = Path(round_dir)
    env_spec = load_env_yaml(env)
    ckpt = rd / "checkpoint.pt"
    holdout = list(range(env_spec.noise.seed_policy.holdout[0],
                          env_spec.noise.seed_policy.holdout[1] + 1))[:n_seeds]
    report = independent_verify(ckpt, env_spec, holdout_seeds=holdout, n_shots=n_shots)
    (rd / "verification_report.json").write_text(report.model_dump_json(indent=2))
    (rd / "verification_report.md").write_text(
        f"# Verification Report\n\n**Verdict:** {report.verdict}\n\n"
        f"- Δ_LER (holdout): {report.delta_ler_holdout:.4g}\n"
        f"- CI: {report.ler_holdout_ci}\n"
        f"- Ablation sanity: {report.ablation_sanity_ok}\n"
        f"- Seed-leakage check: {report.seed_leakage_check_ok}\n\n"
        f"Notes: {report.notes}\n"
    )
    click.echo(report.verdict)


@main.command(name="review-log")
@click.argument("run_dir")
def review_log(run_dir: str) -> None:
    rd = Path(run_dir)
    hist_path = rd / "history.jsonl"
    pareto_path = rd / "pareto.json"
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
    out: dict = {"path": str(target)}
    for fn in ("config.yaml", "metrics.json", "train.log"):
        p = target / fn
        out[f"has_{fn}"] = p.exists()
        if fn == "metrics.json" and p.exists():
            out["metrics"] = json.loads(p.read_text())
    click.echo(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
