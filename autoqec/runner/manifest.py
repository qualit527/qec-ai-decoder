"""Per-round artifact manifest — reproducibility gate (test plan 4.5)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def _tool_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def _repo_sha(anchor: Path) -> Optional[str]:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(anchor), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _dirty_files(anchor: Path) -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(anchor), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    dirty = []
    for line in out:
        if not line.strip():
            continue
        path = line[3:].strip()
        file_path = anchor / path
        sha = ""
        if file_path.is_file():
            sha = hashlib.sha256(file_path.read_bytes()).hexdigest()
        dirty.append({"path": path, "sha256": sha})
    return dirty


def write_artifact_manifest(
    round_dir: Path,
    env_yaml_path: Path,
    dsl_config: dict[str, Any],
    cmd_line: list[str],
) -> Path:
    env_bytes = Path(env_yaml_path).read_bytes() if Path(env_yaml_path).exists() else b""
    env_sha = hashlib.sha256(env_bytes).hexdigest()
    dsl_sha = hashlib.sha256(
        json.dumps(dsl_config, sort_keys=True).encode()
    ).hexdigest()

    anchor = round_dir
    manifest = {
        "repo_sha": _repo_sha(anchor),
        "dirty_files": _dirty_files(anchor),
        "python_version": sys.version.split()[0],
        "torch_version": _tool_version("torch"),
        "stim_version": _tool_version("stim"),
        "pymatching_version": _tool_version("pymatching"),
        "ldpc_version": _tool_version("ldpc"),
        "numpy_version": _tool_version("numpy"),
        "env_yaml_path": str(env_yaml_path),
        "env_yaml_sha256": env_sha,
        "dsl_sha256": dsl_sha,
        "cmd_line": cmd_line,
    }
    out = round_dir / "artifact_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
