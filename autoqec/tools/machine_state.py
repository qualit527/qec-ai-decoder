"""Machine state probe (Task A2.2).

Surfaced to the Ideator subagent as `machine_state_hint` so it can
calibrate candidate size against observed GPU budget + round timings.

Everything except the GPU section is torch-free, so this module imports
cleanly in lean CI environments. Torch is imported lazily inside
`_gpu_snapshot` and any failure (missing module, driver error, no CUDA
device) returns an empty dict rather than raising.
"""
from __future__ import annotations

import json
from pathlib import Path


def _load_history(run_dir: Path) -> list[dict]:
    history_path = run_dir / "history.jsonl"
    if not history_path.exists():
        return []
    with history_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _gpu_snapshot() -> dict:
    """Probe CUDA without letting any failure leak out.

    The whole chain (import → is_available → mem_get_info → device
    properties) sits inside one `try` because a mis-configured driver
    can raise at *any* of those steps, and the Ideator prompt assembly
    must not crash when a GPU probe fails.
    """
    try:
        import torch  # noqa: PLC0415 — intentional lazy import

        if not torch.cuda.is_available():
            return {}
        free, _total = torch.cuda.mem_get_info()
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "vram_free_gb": free / 1e9,
        }
    except Exception:
        return {}


def machine_state(
    run_dir: str | Path,
    *,
    total_wallclock_s_budget: float | None = None,
) -> dict:
    """Compose a snapshot for the Ideator's L3 context.

    `total_wallclock_s_budget` is the outer budget the orchestrator
    enforces; `total_wallclock_s_remaining` is derived from it minus spent.
    """
    run_dir = Path(run_dir)
    history = _load_history(run_dir)
    timings = [
        float(r.get("train_wallclock_s", 0) or 0) + float(r.get("eval_wallclock_s", 0) or 0)
        for r in history
    ]
    killed = sum(1 for r in history if r.get("status") == "killed_by_safety")
    n = len(timings)
    wall_mean = (sum(timings) / n) if n else 0
    wall_p95 = sorted(timings)[min(int(0.95 * n), max(n - 1, 0))] if n else 0
    params_vs_time = [(int(r.get("n_params", 0) or 0), t) for r, t in zip(history, timings)]
    loss_trajectory: list[dict] = []
    delta_trajectory: list[dict] = []
    for r in history:
        # Skip non-training statuses — their loss fields are absent and
        # delta is meaningless. This keeps the Ideator focused on the
        # rounds that actually ran training.
        if r.get("status") != "ok":
            continue
        round_idx = r.get("round")
        if r.get("train_loss_initial") is not None:
            loss_trajectory.append({
                "round": round_idx,
                "initial": r.get("train_loss_initial"),
                "final": r.get("train_loss_final"),
                "mean_last_epoch": r.get("train_loss_mean_last_epoch"),
            })
        if r.get("delta_ler") is not None:
            ci_lo = r.get("delta_ler_ci_low")
            ci_hi = r.get("delta_ler_ci_high")
            delta_trajectory.append({
                "round": round_idx,
                "delta_ler": r.get("delta_ler"),
                "ci": [ci_lo, ci_hi] if (ci_lo is not None or ci_hi is not None) else None,
            })
    spent = sum(timings)
    remaining = (total_wallclock_s_budget - spent) if total_wallclock_s_budget is not None else None

    return {
        "gpu": _gpu_snapshot(),
        "history_timings": {
            "rounds_so_far": n,
            "wall_clock_mean_s": wall_mean,
            "wall_clock_p95_s": wall_p95,
            "params_vs_time": params_vs_time,
            "killed_by_safety_count": killed,
            # Per-round training-loss + Δ_LER so the Ideator can detect
            # "loss drops but Δ_LER flat across every round" (harness /
            # loss-alignment bug) vs "this architecture doesn't work"
            # (science). Entries are in history order; rounds without
            # a training run (status != ok) are omitted.
            "loss_trajectory": loss_trajectory,
            "delta_ler_trajectory": delta_trajectory,
        },
        "budget": {
            "total_wallclock_s_spent": spent,
            "total_wallclock_s_remaining": remaining,
        },
    }
