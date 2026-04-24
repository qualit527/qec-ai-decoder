"""Compatibility wrapper for per-round artifact manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from autoqec.runner.artifact_manifest import write_artifact_manifest as _write_new_manifest
from autoqec.runner.schema import RunnerConfig


def write_artifact_manifest(
    round_dir: Path,
    env_yaml_path: Path,
    dsl_config: dict[str, Any],
    cmd_line: list[str],
) -> Path:
    """Write the canonical nested manifest schema using the legacy call shape."""
    config = RunnerConfig(
        env_name="unknown",
        predecoder_config=dsl_config,
        training_profile="dev",
        seed=0,
        round_dir=str(round_dir),
        env_yaml_path=str(env_yaml_path),
        invocation_argv=cmd_line,
    )
    return _write_new_manifest(
        round_dir=round_dir,
        config=config,
        checkpoint_path=round_dir / "checkpoint.pt",
        metrics_path=round_dir / "metrics.json",
        train_log_path=round_dir / "train.log",
    )
