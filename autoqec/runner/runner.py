"""Pure-Python runner for the AutoQEC MVP."""

from __future__ import annotations

from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from autoqec.decoders.backend_adapter import decode_with_predecoder
from autoqec.decoders.baselines.pymatching_wrap import PymatchingBaseline
from autoqec.decoders.dsl_compiler import compile_predecoder
from autoqec.envs.schema import EnvSpec
from autoqec.runner.artifact_manifest import write_artifact_manifest
from autoqec.runner.data import load_code_artifacts, sample_syndromes
from autoqec.runner.flops import estimate_flops
from autoqec.runner.safety import RunnerSafety, estimate_vram_gb, nan_rate
from autoqec.runner.schema import RoundMetrics, RunnerConfig


class RunnerCallPathError(RuntimeError):
    """Raised when run_round is called in-process with cfg.code_cwd set.

    Worktree-path runs must go through autoqec.orchestration.subprocess_runner
    because Python's import cache would serve main's modules/*.py instead of
    the worktree's edited copies (§15.8).
    """


def _profile_params(env_spec: EnvSpec, profile: str) -> dict[str, int]:
    if profile == "dev":
        return {
            "n_shots_train": min(env_spec.eval_protocol.min_shots_train, 256),
            "n_shots_val": min(env_spec.eval_protocol.min_shots_val, 64),
            "epochs_cap": 1,
        }
    if profile == "prod":
        return {
            "n_shots_train": min(env_spec.eval_protocol.min_shots_train, 2048),
            "n_shots_val": min(env_spec.eval_protocol.min_shots_val, 256),
            "epochs_cap": 3,
        }
    if profile == "benchmark":
        return {
            "n_shots_train": env_spec.eval_protocol.min_shots_train,
            "n_shots_val": env_spec.eval_protocol.min_shots_val,
            "epochs_cap": 6,
        }
    raise ValueError(f"unknown training profile: {profile}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _failure_rate(env_spec: EnvSpec, predictions: np.ndarray, targets: np.ndarray) -> float:
    if env_spec.code.type == "stim_circuit":
        return float((predictions != targets).any(axis=1).mean())
    return float((predictions != targets).any(axis=1).mean())


def _write_metrics(round_dir: Path, metrics: RoundMetrics) -> RoundMetrics:
    (round_dir / "metrics.json").write_text(
        metrics.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return metrics


def _with_round_attempt_id(metrics: RoundMetrics, config: RunnerConfig) -> RoundMetrics:
    if config.round_attempt_id is None or metrics.round_attempt_id is not None:
        return metrics
    return metrics.model_copy(update={"round_attempt_id": config.round_attempt_id})


def _finalize_success(
    round_dir: Path,
    metrics: RoundMetrics,
    *,
    config: RunnerConfig,
) -> RoundMetrics:
    written = _write_metrics(round_dir, _with_round_attempt_id(metrics, config))
    write_artifact_manifest(
        round_dir,
        config=config,
        checkpoint_path=Path(written.checkpoint_path or round_dir / "checkpoint.pt"),
        metrics_path=round_dir / "metrics.json",
        train_log_path=Path(written.training_log_path or round_dir / "train.log"),
    )
    return written


def run_round(
    config: RunnerConfig,
    env_spec: EnvSpec,
    safety: RunnerSafety | None = None,
) -> RoundMetrics:
    if config.code_cwd is not None:
        raise RunnerCallPathError(
            "code_cwd is set — use autoqec.orchestration.subprocess_runner.run_round_in_subprocess "
            "instead of run_round (§15.8)"
        )
    safety = safety or RunnerSafety()
    _set_seed(config.seed)
    round_dir = Path(config.round_dir).resolve()
    round_dir.mkdir(parents=True, exist_ok=True)
    with (round_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.predecoder_config, f, sort_keys=False)
    train_log = round_dir / "train.log"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        artifacts = load_code_artifacts(env_spec)
        model = compile_predecoder(
            config.predecoder_config,
            n_var=artifacts.n_var,
            n_check=artifacts.n_check,
        )
    except Exception as exc:
        return _write_metrics(
            round_dir,
            _with_round_attempt_id(
                RoundMetrics(status="compile_error", status_reason=str(exc)),
                config,
            ),
        )

    n_params = int(sum(parameter.numel() for parameter in model.parameters()))
    profile = _profile_params(env_spec, config.training_profile)
    hidden_hint = (
        config.predecoder_config.get("gnn", {}).get("hidden_dim")
        or config.predecoder_config.get("neural_bp", {}).get("attention_heads")
        or 64
    )

    if safety.VRAM_PRE_CHECK and torch.cuda.is_available():
        vram_est = estimate_vram_gb(model, batch_size=64, hidden=hidden_hint)
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if vram_est > free_gb * 0.9:
            return _write_metrics(
                round_dir,
                _with_round_attempt_id(
                    RoundMetrics(
                        status="killed_by_safety",
                        status_reason=f"VRAM estimate {vram_est:.2f} > free {free_gb:.2f}",
                        n_params=n_params,
                    ),
                    config,
                ),
            )

    model = model.to(device)
    train_syndrome, train_target = sample_syndromes(
        env_spec,
        artifacts,
        env_spec.noise.seed_policy.train,
        profile["n_shots_train"],
    )
    train_syndrome = train_syndrome.to(device)
    train_target = train_target.to(device)

    ctx = {
        "edge_index": artifacts.edge_index.to(device),
        "n_var": artifacts.n_var,
        "n_check": artifacts.n_check,
        "prior_p": artifacts.prior_p.to(device),
    }
    if artifacts.parity_check_matrix is not None:
        ctx["parity_check_matrix"] = torch.from_numpy(artifacts.parity_check_matrix).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.predecoder_config["training"]["learning_rate"]),
    )
    batch_size = int(config.predecoder_config["training"]["batch_size"])
    epochs = min(int(config.predecoder_config["training"]["epochs"]), profile["epochs_cap"])
    losses: list[float] = []

    train_start = time.time()
    for _ in range(epochs):
        for start in range(0, train_syndrome.shape[0], batch_size):
            if time.time() - train_start > safety.WALL_CLOCK_HARD_CUTOFF_S:
                return _write_metrics(
                    round_dir,
                    _with_round_attempt_id(
                        RoundMetrics(
                            status="killed_by_safety",
                            status_reason="wall_clock_cutoff during training",
                            n_params=n_params,
                            train_wallclock_s=time.time() - train_start,
                        ),
                        config,
                    ),
                )
            batch_syndrome = train_syndrome[start : start + batch_size]
            if artifacts.code_type == "stim_circuit":
                target = batch_syndrome
            elif model.output_mode == "soft_priors":
                target = train_target[start : start + batch_size].float()
            else:
                target = batch_syndrome

            pred = model(batch_syndrome, ctx)
            if model.output_mode == "soft_priors":
                pred_slice = pred[:, : target.shape[1]].clamp(1e-5, 1 - 1e-5)
                loss = F.binary_cross_entropy(pred_slice, target.float())
            else:
                loss = F.mse_loss(pred.float(), target.float())

            if not torch.isfinite(loss):
                losses.append(float("nan"))
                if nan_rate(losses) > safety.MAX_NAN_RATE:
                    return _write_metrics(
                        round_dir,
                        _with_round_attempt_id(
                            RoundMetrics(
                                status="killed_by_safety",
                                status_reason=f"NaN rate {nan_rate(losses):.3f}",
                                n_params=n_params,
                                train_wallclock_s=time.time() - train_start,
                            ),
                            config,
                        ),
                    )
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

    train_wallclock = time.time() - train_start
    train_log.write_text(
        "\n".join(f"{idx}\t{loss:.6g}" for idx, loss in enumerate(losses)),
        encoding="utf-8",
    )

    eval_start = time.time()
    val_syndrome, val_target = sample_syndromes(
        env_spec,
        artifacts,
        env_spec.noise.seed_policy.val,
        profile["n_shots_val"],
    )
    syndrome_np = val_syndrome.numpy()
    target_np = val_target.numpy()

    if env_spec.classical_backend == "mwpm":
        baseline = PymatchingBaseline.from_circuit(artifacts.code_artifact)  # type: ignore[arg-type]
        plain_pred = baseline.decode_batch(syndrome_np.astype(bool))
    else:
        uniform = np.full((syndrome_np.shape[0], artifacts.n_var), float(env_spec.noise.p[0]))
        plain_pred = decode_with_predecoder(
            uniform,
            env_spec,
            syndrome_np,
            artifacts.code_artifact,
            "soft_priors",
        )
    ler_plain = _failure_rate(env_spec, plain_pred, target_np)

    model.eval()
    with torch.no_grad():
        pred_out = model(val_syndrome.to(device), ctx).cpu().numpy()
    pred_labels = decode_with_predecoder(
        pred_out,
        env_spec,
        syndrome_np,
        artifacts.code_artifact,
        model.output_mode,
    )
    ler_predecoder = _failure_rate(env_spec, pred_labels, target_np)
    delta_ler = ler_plain - ler_predecoder

    try:
        flops = estimate_flops(model, (val_syndrome[:1].to(device), ctx))
    except Exception:
        flops = 2 * n_params

    checkpoint_path = round_dir / "checkpoint.pt"
    torch.save(
        {
            "class_name": type(model).__name__,
            "state_dict": model.cpu().state_dict(),
            "output_mode": model.output_mode,
            "dsl_config": config.predecoder_config,
        },
        checkpoint_path,
    )
    metrics = RoundMetrics(
        status="ok",
        ler_plain_classical=ler_plain,
        ler_predecoder=ler_predecoder,
        delta_ler=delta_ler,
        flops_per_syndrome=int(flops),
        n_params=n_params,
        train_wallclock_s=train_wallclock,
        eval_wallclock_s=time.time() - eval_start,
        vram_peak_gb=float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0,
        checkpoint_path=str(checkpoint_path),
        training_log_path=str(train_log),
    )
    return _finalize_success(round_dir, metrics, config=config)
