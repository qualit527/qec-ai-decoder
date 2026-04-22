from __future__ import annotations

import json
import random
import time
from pathlib import Path

import click
import yaml

from autoqec.envs.schema import EnvSpec, load_env_yaml
from autoqec.runner.schema import RunnerConfig


@click.group()
def main() -> None:
    """AutoQEC CLI."""


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
) -> None:
    env = load_env_yaml(env_yaml)
    with open(config_yaml) as f:
        cfg_dict = yaml.safe_load(f)

    # Parse fork_from: JSON list -> list[str]; bare string -> str.
    parsed_fork_from: str | list[str] | None = None
    if fork_from is not None:
        parsed_fork_from = (
            json.loads(fork_from) if fork_from.strip().startswith("[") else fork_from
        )

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
    if code_cwd is not None:
        from autoqec.orchestration.subprocess_runner import run_round_in_subprocess

        metrics = run_round_in_subprocess(cfg, env, round_attempt_id=round_attempt_id)
    else:
        from autoqec.runner.runner import run_round

        metrics = run_round(cfg, env)
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
    run_dir = Path("runs") / run_id
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
        with (run_dir / "history.jsonl").open("a") as f:
            f.write(metrics.model_dump_json() + "\n")
        click.echo(f"Round {round_idx}: {metrics.status} Δ={metrics.delta_ler}")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
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


if __name__ == "__main__":
    main()
