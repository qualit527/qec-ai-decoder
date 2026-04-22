from __future__ import annotations

import json
import random
import time
from pathlib import Path

import click
import yaml

from autoqec.envs.schema import EnvSpec, load_env_yaml
from autoqec.runner.runner import run_round
from autoqec.runner.schema import RunnerConfig


@click.group()
def main() -> None:
    """AutoQEC CLI."""


@main.command(name="run-round")
@click.argument("env_yaml")
@click.argument("config_yaml")
@click.argument("round_dir")
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
def run_round_cmd(env_yaml: str, config_yaml: str, round_dir: str, profile: str) -> None:
    env = load_env_yaml(env_yaml)
    with open(config_yaml) as f:
        cfg_dict = yaml.safe_load(f)
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config=cfg_dict,
        training_profile=profile,
        seed=0,
        round_dir=round_dir,
    )
    metrics = run_round(cfg, env)
    click.echo(metrics.model_dump_json(indent=2))


@main.command()
@click.argument("env_yaml")
@click.option("--rounds", type=int, default=10)
@click.option("--profile", type=click.Choice(["dev", "prod"]), default="dev")
@click.option("--no-llm", is_flag=True, help="Pick random seed templates instead of calling subagents")
def run(env_yaml: str, rounds: int, profile: str, no_llm: bool) -> None:
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
        rounds = [json.loads(l) for l in f if l.strip()]
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
