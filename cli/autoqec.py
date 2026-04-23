from __future__ import annotations

from importlib import resources
import json
import os
import random
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
from autoqec.runner.schema import RunnerConfig


@click.group()
def main() -> None:
    """AutoQEC CLI."""


RESULT_PREFIX = "AUTOQEC_RESULT_JSON="


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
    with open(config_yaml) as f:
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
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    click.echo(
        f"{RESULT_PREFIX}"
        + json.dumps(
            {
                "run_dir": str(run_dir),
                "rounds": rounds,
                "candidate_pareto_path": str(run_dir / "candidate_pareto.json"),
            }
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


if __name__ == "__main__":
    main()
