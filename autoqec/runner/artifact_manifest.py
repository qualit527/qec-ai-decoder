from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys

from autoqec.runner.schema import RunnerConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _git_output(repo_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_metadata(repo_root: Path) -> dict[str, object]:
    status = _git_output(repo_root, "status", "--short")
    return {
        "commit_sha": _git_output(repo_root, "rev-parse", "HEAD"),
        "branch": _git_output(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
    }


def _round_number(round_dir: Path) -> int | None:
    suffix = round_dir.name.removeprefix("round_")
    return int(suffix) if suffix.isdigit() else None


def build_artifact_manifest(
    round_dir: Path,
    *,
    config: RunnerConfig,
    checkpoint_path: Path,
    metrics_path: Path,
    train_log_path: Path,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    env_path = Path(config.env_yaml_path).expanduser().resolve() if config.env_yaml_path is not None else None

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo": _git_metadata(repo_root),
        "environment": {
            "env_yaml_path": config.env_yaml_path,
            "env_yaml_sha256": _sha256_file(env_path) if env_path is not None and env_path.exists() else None,
        },
        "round": {
            "run_id": round_dir.parent.name if round_dir.parent != round_dir else None,
            "round_dir": round_dir.name,
            "round": _round_number(round_dir),
            "dsl_config_sha256": _sha256_json(config.predecoder_config),
            "command_line": list(config.invocation_argv or []),
        },
        "artifacts": {
            "config_yaml": "config.yaml",
            "checkpoint": checkpoint_path.name,
            "metrics": metrics_path.name,
            "train_log": train_log_path.name,
        },
        "packages": {
            "python": sys.version.split()[0],
            "torch": _package_version("torch"),
            "cuda": _package_version("nvidia-cuda-runtime-cu12") if _package_version("nvidia-cuda-runtime-cu12") != "unavailable" else "none",
            "stim": _package_version("stim"),
            "pymatching": _package_version("pymatching"),
            "ldpc": _package_version("ldpc"),
        },
    }


def validate_artifact_manifest(round_dir: Path, payload: dict[str, object] | None = None) -> dict[str, object]:
    manifest = payload
    if manifest is None:
        manifest_path = round_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"missing artifact manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema_version = manifest.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported artifact manifest schema_version: {schema_version!r}")

    for section in ("repo", "environment", "round", "artifacts", "packages"):
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"artifact manifest missing section: {section}")

    commit_sha = manifest["repo"].get("commit_sha")
    if not commit_sha:
        raise ValueError("artifact manifest missing repo.commit_sha")
    if not manifest["environment"].get("env_yaml_sha256"):
        raise ValueError("artifact manifest missing environment.env_yaml_sha256")
    if not manifest["round"].get("dsl_config_sha256"):
        raise ValueError("artifact manifest missing round.dsl_config_sha256")
    command_line = manifest["round"].get("command_line")
    if not isinstance(command_line, list) or not all(isinstance(item, str) for item in command_line):
        raise ValueError("artifact manifest round.command_line must be a list of strings")

    artifacts = manifest["artifacts"]
    for key in ("config_yaml", "checkpoint", "metrics", "train_log"):
        rel = artifacts.get(key)
        if not isinstance(rel, str) or not rel:
            raise ValueError(f"artifact manifest missing artifacts.{key}")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise ValueError(f"artifact manifest artifacts.{key} must be a safe relative path")
        if not (round_dir / rel_path).exists():
            raise ValueError(f"artifact manifest references missing file: {round_dir / rel_path}")

    return manifest


def write_artifact_manifest(
    round_dir: Path,
    *,
    config: RunnerConfig,
    checkpoint_path: Path,
    metrics_path: Path,
    train_log_path: Path,
) -> Path:
    payload = build_artifact_manifest(
        round_dir,
        config=config,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        train_log_path=train_log_path,
    )
    manifest_path = round_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path
