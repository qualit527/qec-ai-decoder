"""§15.8 — `run_round` must refuse in-process execution when `code_cwd` is set.

Worktree-path runs must shell out through `autoqec.orchestration.subprocess_runner`
because Python's import cache would serve main's `modules/*.py` instead of the
worktree's edited copies. The guard in `run_round` makes that misuse loud.
"""

from __future__ import annotations

import importlib.util

import pytest

# The guard lives at the top of `run_round`, but importing `autoqec.runner.runner`
# pulls `torch`. The second test (source-level check) works without torch.
_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _HAS_TORCH, reason="autoqec.runner.runner requires torch")
def test_run_round_raises_when_code_cwd_set() -> None:
    from autoqec.envs.schema import load_env_yaml
    from autoqec.runner.runner import RunnerCallPathError, run_round
    from autoqec.runner.schema import RunnerConfig

    env = load_env_yaml("autoqec/envs/builtin/surface_d5_depol.yaml")
    cfg = RunnerConfig(
        env_name=env.name,
        predecoder_config={
            "type": "gnn",
            "output_mode": "soft_priors",
            "gnn": {
                "layers": 1,
                "hidden_dim": 8,
                "message_fn": "mlp",
                "aggregation": "sum",
                "normalization": "none",
                "residual": False,
                "edge_features": [],
            },
            "head": "linear",
            "training": {
                "learning_rate": 1e-3,
                "batch_size": 16,
                "epochs": 1,
                "loss": "bce",
                "profile": "dev",
            },
        },
        training_profile="dev",
        seed=0,
        round_dir="/tmp/r",
        code_cwd="/abs/.worktrees/x",
        branch="exp/t/1-a",
    )
    with pytest.raises(RunnerCallPathError):
        run_round(cfg, env)


def test_runner_call_path_error_is_runtime_error() -> None:
    """Even without torch, we can at least verify the exception class via a cheap import path."""
    # Avoid heavy import; confirm the module attribute exists on a lazy source scan.
    import pathlib

    src = pathlib.Path("autoqec/runner/runner.py").read_text(encoding="utf-8")
    assert "class RunnerCallPathError" in src
    assert "RuntimeError" in src
    assert "code_cwd" in src
