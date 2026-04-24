"""Phase-2 end-to-end handshake (Task A2.1).

Bypasses the LLM subagents: loads a hand-written Tier-1 DSL config and
invokes Lin's Runner directly. The point is to prove the orchestration
↔ Runner contract works before any Ideator/Coder are wired in.

Use
---

    python scripts/e2e_handshake.py --round-dir runs/handshake/round_0
    python scripts/e2e_handshake.py --min-steps 100 --round-dir runs/handshake/round_0

The script is intentionally small — anything more than `load env → build
config → run_round → print metrics` belongs in the research loop, not here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import sys
from pathlib import Path

# Make the repo root importable and resolve shipped-config paths relative to
# it, so `python /abs/path/to/scripts/e2e_handshake.py` works from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from autoqec.envs.schema import load_env_yaml  # noqa: E402
from autoqec.runner.schema import RunnerConfig  # noqa: E402

DEFAULT_ENV_YAML = _REPO_ROOT / "autoqec/envs/builtin/surface_d5_depol.yaml"
DEFAULT_STUB_YAML = _REPO_ROOT / "autoqec/example_db/handshake_stub.yaml"


def _training_budget(env, profile: str) -> tuple[int, int]:
    if profile == "dev":
        return min(env.eval_protocol.min_shots_train, 256), 1
    return min(env.eval_protocol.min_shots_train, 2048), 3


def _with_min_steps(cfg_dict: dict, env, min_steps: int | None) -> dict:
    """Return a config whose Runner train.log satisfies a wc -l step gate."""
    if min_steps is None:
        return cfg_dict
    if min_steps < 1:
        raise ValueError("--min-steps must be >= 1")

    cfg_with_steps = deepcopy(cfg_dict)
    training = cfg_with_steps.setdefault("training", {})
    profile = training.get("profile", "dev")
    n_shots_train, epochs_cap = _training_budget(env, profile)

    desired_rows = min_steps + 1
    current_epochs = max(1, int(training.get("epochs", 1)))
    required_epochs = math.ceil(desired_rows / n_shots_train)
    epochs = min(max(current_epochs, required_epochs), epochs_cap)
    max_rows = n_shots_train * epochs
    if desired_rows > max_rows:
        raise ValueError(
            f"--min-steps={min_steps} exceeds the {profile!r} profile capacity "
            f"of {max_rows - 1} wc -l lines"
        )

    batches_per_epoch = math.ceil(desired_rows / epochs)
    batch_size_for_gate = max(1, n_shots_train // batches_per_epoch)
    current_batch_size = max(1, int(training.get("batch_size", n_shots_train)))

    training["epochs"] = epochs
    training["batch_size"] = min(current_batch_size, batch_size_for_gate)
    return cfg_with_steps


def build_runner_config(
    env_yaml: str | Path | None,
    stub_yaml: str | Path | None,
    round_dir: str | Path,
    seed: int = 0,
    min_steps: int | None = None,
) -> RunnerConfig:
    """Load env + DSL stub and compose a RunnerConfig matching §2.2.

    `env_yaml` / `stub_yaml` default to the shipped defaults anchored at
    `_REPO_ROOT`, so callers running from any cwd get the same behaviour.
    Kept importable so the unit test can exercise the config-composition
    contract without touching torch.
    """
    env_path = Path(env_yaml) if env_yaml is not None else DEFAULT_ENV_YAML
    stub_path = Path(stub_yaml) if stub_yaml is not None else DEFAULT_STUB_YAML
    env = load_env_yaml(env_path)
    cfg_dict = yaml.safe_load(stub_path.read_text(encoding="utf-8"))
    cfg_dict = _with_min_steps(cfg_dict, env, min_steps)
    return RunnerConfig(
        env_name=env.name,
        predecoder_config=cfg_dict,
        training_profile=cfg_dict.get("training", {}).get("profile", "dev"),
        seed=seed,
        round_dir=str(Path(round_dir).resolve()),
    )


def main(
    env_yaml: str | Path | None = None,
    stub_yaml: str | Path | None = None,
    round_dir: str | Path = "runs/handshake/round_0",
    seed: int = 0,
    min_steps: int | None = None,
) -> dict:
    """Run one Runner round and return the metrics dict.

    Imports the heavy Runner lazily so importing this module for unit
    tests does not require torch.
    """
    from autoqec.runner.runner import run_round  # noqa: PLC0415 — intentional lazy import

    env_path = Path(env_yaml) if env_yaml is not None else DEFAULT_ENV_YAML
    env = load_env_yaml(env_path)
    cfg = build_runner_config(env_yaml, stub_yaml, round_dir, seed, min_steps=min_steps)
    metrics = run_round(cfg, env)
    print(metrics.model_dump_json(indent=2))
    return json.loads(metrics.model_dump_json())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-yaml", default=str(DEFAULT_ENV_YAML))
    p.add_argument("--stub-yaml", default=str(DEFAULT_STUB_YAML))
    p.add_argument("--round-dir", default=str(_REPO_ROOT / "runs/handshake/round_0"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Adjust the training batch budget so train.log has at least this many wc -l lines.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        env_yaml=args.env_yaml,
        stub_yaml=args.stub_yaml,
        round_dir=args.round_dir,
        seed=args.seed,
        min_steps=args.min_steps,
    )
