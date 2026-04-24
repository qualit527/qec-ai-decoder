from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from autoqec.runner.artifact_manifest import validate_artifact_manifest


BACKEND_ENV_VARS = (
    "AUTOQEC_IDEATOR_BACKEND",
    "AUTOQEC_CODER_BACKEND",
    "AUTOQEC_ANALYST_BACKEND",
)
NETWORK_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
)

NO_NETWORK_SITECUSTOMIZE = """\
from __future__ import annotations

import os
import socket


def _blocked(*args, **kwargs):
    raise RuntimeError("AutoQEC offline replay blocks network access")


if os.environ.get("AUTOQEC_REPLAY_NO_NETWORK") == "1":
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
    _socket_type = socket.socket

    class _OfflineSocket(_socket_type):
        def connect(self, *args, **kwargs):
            return _blocked(*args, **kwargs)

        def connect_ex(self, *args, **kwargs):
            return _blocked(*args, **kwargs)

    socket.socket = _OfflineSocket
"""


def package_run_dir(run_dir: Path, output_path: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    package_path = output_path or run_dir.parent / f"{run_dir.name}.tar.gz"
    if package_path.exists():
        raise FileExistsError(f"package path already exists: {package_path}")
    round_dirs = sorted(path for path in run_dir.glob("round_*") if path.is_dir())
    if not round_dirs:
        raise ValueError(f"run directory has no round_* subdirectories: {run_dir}")
    for round_dir in round_dirs:
        validate_artifact_manifest(round_dir)
    with tarfile.open(package_path, "w:gz") as archive:
        archive.add(run_dir, arcname=run_dir.name, recursive=False)
        for path in sorted(run_dir.rglob("*")):
            archive.add(path, arcname=Path(run_dir.name) / path.relative_to(run_dir), recursive=False)
    return package_path


def extract_run_package(package_path: Path, extract_root: Path) -> Path:
    package_path = package_path.resolve()
    extract_root = extract_root.resolve()
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(package_path, "r:gz") as archive:
        archive.extractall(extract_root, filter="data")
        top_level_names = [name.split("/", 1)[0] for name in archive.getnames() if name]
    if not top_level_names:
        raise ValueError(f"package {package_path} was empty")
    return extract_root / top_level_names[0]


def compare_verification_reports(
    original: dict,
    replay: dict,
    *,
    float_tol: float = 1e-6,
) -> None:
    for key in ("verdict", "holdout_seeds_used", "paired_eval_bundle_id"):
        if original.get(key) != replay.get(key):
            raise ValueError(f"{key} mismatch: {original.get(key)!r} != {replay.get(key)!r}")

    for key in ("ler_holdout", "delta_ler_holdout", "ler_shuffled"):
        if abs(float(original[key]) - float(replay[key])) > float_tol:
            raise ValueError(f"{key} drift exceeds tolerance")

    original_ci = list(original["ler_holdout_ci"])
    replay_ci = list(replay["ler_holdout_ci"])
    if len(original_ci) != 2 or len(replay_ci) != 2:
        raise ValueError("ler_holdout_ci must be length 2")
    for idx, (expected, actual) in enumerate(zip(original_ci, replay_ci, strict=True)):
        if abs(float(expected) - float(actual)) > float_tol:
            raise ValueError(f"ler_holdout_ci[{idx}] drift exceeds tolerance")


def _write_no_network_sitecustomize(guard_root: Path) -> Path:
    guard_root.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = guard_root / "sitecustomize.py"
    sitecustomize_path.write_text(NO_NETWORK_SITECUSTOMIZE, encoding="utf-8")
    return sitecustomize_path


def run_verify_offline(
    round_dir: Path,
    *,
    env_yaml: str,
    python_bin: str,
    n_shots: int,
    n_seeds: int,
) -> dict:
    env = os.environ.copy()
    for key in BACKEND_ENV_VARS:
        env.pop(key, None)
    for key in NETWORK_ENV_VARS:
        env.pop(key, None)
    env["NO_PROXY"] = "*"
    env["AUTOQEC_REPLAY_NO_NETWORK"] = "1"
    repo_root = Path.cwd().resolve()
    cmd = [
        python_bin,
        "-m",
        "cli.autoqec",
        "verify",
        str(round_dir),
        "--env",
        env_yaml,
        "--n-shots",
        str(n_shots),
        "--n-seeds",
        str(n_seeds),
    ]
    with tempfile.TemporaryDirectory(prefix="autoqec-offline-") as guard_dir:
        guard_root = Path(guard_dir)
        _write_no_network_sitecustomize(guard_root)
        pythonpath_entries = [str(guard_root), str(repo_root)]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
    return json.loads((round_dir / "verification_report.json").read_text(encoding="utf-8"))


def replay_packaged_run(
    run_dir: Path,
    *,
    package_path: Path | None = None,
    env_yaml: str,
    python_bin: str,
    n_shots: int,
    n_seeds: int,
    extract_root: Path,
    round_name: str = "round_1",
    float_tol: float = 1e-6,
) -> dict:
    resolved_package_path = package_path.resolve() if package_path is not None else package_run_dir(run_dir)
    extracted_run_dir = extract_run_package(resolved_package_path, extract_root)
    extracted_round_dir = extracted_run_dir / round_name

    original_report_path = extracted_round_dir / "verification_report.json"
    original_report_copy_path: Path | None = None
    original_report: dict | None = None
    if original_report_path.exists():
        original_report_copy_path = extracted_round_dir / "verification_report.original.json"
        shutil.copyfile(original_report_path, original_report_copy_path)
        original_report = json.loads(original_report_copy_path.read_text(encoding="utf-8"))

    replay_report = run_verify_offline(
        extracted_round_dir,
        env_yaml=env_yaml,
        python_bin=python_bin,
        n_shots=n_shots,
        n_seeds=n_seeds,
    )
    if original_report is not None:
        compare_verification_reports(original_report, replay_report, float_tol=float_tol)

    return {
        "package_path": str(resolved_package_path),
        "extracted_run_dir": str(extracted_run_dir),
        "round_dir": str(extracted_round_dir),
        "original_report_copy_path": str(original_report_copy_path) if original_report_copy_path is not None else None,
        "replay_report_path": str(extracted_round_dir / "verification_report.json"),
        "float_tol": float_tol,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and replay an AutoQEC run offline.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--package-path")
    parser.add_argument("--env", required=True, dest="env_yaml")
    parser.add_argument("--python-bin", required=True)
    parser.add_argument("--n-shots", type=int, required=True)
    parser.add_argument("--n-seeds", type=int, required=True)
    parser.add_argument("--extract-root", required=True)
    parser.add_argument("--round-name", default="round_1")
    parser.add_argument("--float-tol", type=float, default=1e-6)
    args = parser.parse_args()

    result = replay_packaged_run(
        Path(args.run_dir),
        package_path=Path(args.package_path) if args.package_path else None,
        env_yaml=args.env_yaml,
        python_bin=args.python_bin,
        n_shots=args.n_shots,
        n_seeds=args.n_seeds,
        extract_root=Path(args.extract_root),
        round_name=args.round_name,
        float_tol=args.float_tol,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
