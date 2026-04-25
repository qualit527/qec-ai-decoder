"""Pure-Python runner for the AutoQEC MVP."""

from __future__ import annotations

import math
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
from autoqec.eval.bootstrap import bootstrap_ci_mean
from autoqec.runner.artifact_manifest import write_artifact_manifest
from autoqec.runner.data import load_code_artifacts, sample_syndromes
from autoqec.runner.flops import estimate_flops
from autoqec.runner.safety import RunnerSafety, estimate_vram_gb, nan_rate
from autoqec.runner.schema import RoundMetrics, RunnerConfig


def _summarize_losses(losses: list[float], batches_per_epoch: int) -> dict[str, float | int | None]:
    """Derive the loss-telemetry fields surfaced on RoundMetrics.

    Finite-only: NaN/inf entries are dropped before summarizing so a single
    bad batch doesn't poison the rolling means. If no finite losses exist
    (e.g., compile_error before any step), every summary field is None.
    """
    finite = [x for x in losses if math.isfinite(x)]
    if not finite:
        return {
            "train_loss_initial": None,
            "train_loss_final": None,
            "train_loss_mean_last_epoch": None,
            "train_batches_total": len(losses),
        }
    tail = finite[-batches_per_epoch:] if batches_per_epoch > 0 else finite
    return {
        "train_loss_initial": float(finite[0]),
        "train_loss_final": float(finite[-1]),
        "train_loss_mean_last_epoch": float(sum(tail) / len(tail)),
        "train_batches_total": len(losses),
    }


class RunnerCallPathError(RuntimeError):
    """Raised when run_round is called in-process with cfg.code_cwd set.

    Worktree-path runs must go through autoqec.orchestration.subprocess_runner
    because Python's import cache would serve main's modules/*.py instead of
    the worktree's edited copies (§15.8).
    """


def _profile_params(env_spec: EnvSpec, profile: str) -> dict[str, int]:
    """Return per-profile caps on training shots/epochs.

    The val cap is decoupled from the train cap because Δ_LER is a
    binomial statistic whose stderr scales as ``1/sqrt(n_val)`` — at
    ``ler_plain ≈ 4e-3`` you need roughly 2000 val shots to resolve a
    delta of 0.3 %. Train shots are much cheaper per sample (no MWPM
    rebuild), so we can be conservative on train and spend the budget
    on val.
    """
    if profile == "dev":
        return {
            "n_shots_train": min(env_spec.eval_protocol.min_shots_train, 2048),
            "n_shots_val": min(env_spec.eval_protocol.min_shots_val, 2048),
            "epochs_cap": 3,
        }
    return {
        "n_shots_train": min(env_spec.eval_protocol.min_shots_train, 8192),
        "n_shots_val": min(env_spec.eval_protocol.min_shots_val, 8192),
        "epochs_cap": 10,
    }


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


def _per_shot_failures(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return a (B,) int32 indicator of "this shot logically failed".

    The paired-bootstrap below resamples shot indices with replacement;
    both decoders (plain & predecoder) must be scored on the *same*
    resampled indices so the paired variance cancels correctly.
    """
    return (predictions != targets).any(axis=1).astype(np.int32)


def _paired_delta_ci(
    plain_failures: np.ndarray,
    pred_failures: np.ndarray,
    *,
    ci: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over the per-shot Δ = plain - pred indicator.

    ``plain_failures[i]`` and ``pred_failures[i]`` must correspond to the
    *same* val shot i. Resampling the index jointly (rather than the two
    arrays independently) is what makes the CI narrow when the decoders
    mostly agree — the dominant source of variance is shot-to-shot, not
    decoder-to-decoder.
    """
    assert plain_failures.shape == pred_failures.shape
    delta_per_shot = (plain_failures.astype(np.int32) - pred_failures.astype(np.int32)).astype(np.float64)
    return bootstrap_ci_mean(delta_per_shot, n_resamples=n_resamples, ci=ci, seed=seed)


def _predict_model_outputs(
    model: torch.nn.Module,
    syndrome: torch.Tensor,
    ctx: dict,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    for start in range(0, syndrome.shape[0], batch_size):
        batch = syndrome[start : start + batch_size].to(device)
        outputs.append(model(batch, ctx).detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


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
    # config.seed is the per-round offset threaded by the orchestrator.
    # Round-0 sees seeds [train.start, train.start+8), round-1 sees
    # [train.start+8, train.start+16), etc. See `_select_seeds` for the
    # wrap policy.
    train_batch = sample_syndromes(
        env_spec,
        artifacts,
        env_spec.noise.seed_policy.train,
        profile["n_shots_train"],
        round_offset=config.seed,
    )
    train_syndrome = train_batch.syndrome.to(device)
    train_errors = train_batch.errors.to(device)

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
    batches_per_epoch = max(1, math.ceil(train_syndrome.shape[0] / batch_size))

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
                            **_summarize_losses(losses, batches_per_epoch),
                        ),
                        config,
                    ),
                )
            batch_syndrome = train_syndrome[start : start + batch_size]
            batch_errors = train_errors[start : start + batch_size]

            pred = model(batch_syndrome, ctx)
            if model.output_mode == "soft_priors":
                # Supervised per-mechanism error probability. Errors shape
                # (B, n_var) must match pred shape (B, n_var) — no slicing.
                assert pred.shape == batch_errors.shape, (
                    f"predecoder output {pred.shape} must match errors "
                    f"target {batch_errors.shape} in soft_priors mode"
                )
                pred_clamped = pred.clamp(1e-5, 1 - 1e-5)
                loss = F.binary_cross_entropy(pred_clamped, batch_errors.float())
            else:
                # hard_flip: model outputs a "cleaned syndrome" (B, n_check)
                # which is fed directly to the classical backend. We have
                # no ground-truth cleaned syndrome, so we train against the
                # syndrome that an ideal predecoder would produce: the
                # syndrome minus the syndrome of the supervised error
                # vector. For a correct decoder that output is all-zeros,
                # but operator-level XOR is non-differentiable so we use
                # the raw syndrome as a weak surrogate — see docs for
                # limitations. This is the best we can do without a
                # proper differentiable reconstruction loss; users wanting
                # meaningful deltaLER should prefer output_mode=soft_priors.
                target = batch_syndrome
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
                                **_summarize_losses(losses, batches_per_epoch),
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
    loss_summary = _summarize_losses(losses, batches_per_epoch)

    eval_start = time.time()
    val_batch = sample_syndromes(
        env_spec,
        artifacts,
        env_spec.noise.seed_policy.val,
        profile["n_shots_val"],
        round_offset=config.seed,
    )
    val_syndrome = val_batch.syndrome
    # LER is always computed against observables. For parity-check codes
    # observables aliases errors, so the comparison reduces to the
    # prior behaviour. For stim_circuit codes observables is the logical
    # outcome (shape (B, num_observables)), which is what both the MWPM
    # baseline and the reweighted path return from their decode calls.
    syndrome_np = val_syndrome.numpy()
    target_np = val_batch.observables.numpy()

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
    plain_failures = _per_shot_failures(plain_pred, target_np)
    ler_plain = float(plain_failures.mean())

    model.eval()
    with torch.no_grad():
        pred_out = _predict_model_outputs(
            model,
            val_syndrome,
            ctx,
            device=device,
            batch_size=batch_size,
        )
    pred_labels = decode_with_predecoder(
        pred_out,
        env_spec,
        syndrome_np,
        artifacts.code_artifact,
        model.output_mode,
    )
    pred_failures = _per_shot_failures(pred_labels, target_np)
    ler_predecoder = float(pred_failures.mean())
    delta_ler = ler_plain - ler_predecoder
    ci_level = float(env_spec.eval_protocol.bootstrap_ci)
    _, delta_ci_low, delta_ci_high = _paired_delta_ci(
        plain_failures, pred_failures, ci=ci_level, seed=config.seed,
    )

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
        delta_ler_ci_low=delta_ci_low,
        delta_ler_ci_high=delta_ci_high,
        flops_per_syndrome=int(flops),
        n_params=n_params,
        train_wallclock_s=train_wallclock,
        eval_wallclock_s=time.time() - eval_start,
        vram_peak_gb=float(torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0,
        checkpoint_path=str(checkpoint_path),
        training_log_path=str(train_log),
        **loss_summary,
    )
    return _finalize_success(round_dir, metrics, config=config)
