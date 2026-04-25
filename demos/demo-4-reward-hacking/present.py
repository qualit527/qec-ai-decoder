"""Narrated, visualized walkthrough of the reward-hacking demo.

Unlike ``run.sh`` (which only prints the final verdict), this script breaks
the demo into five phases with ASCII bar charts + an optional matplotlib
PNG so a presenter can narrate what the hacking example is, how it was
detected, and what the Pareto consequence is.

Phases:
  1. Construct the memorizing cheater from training-seed syndromes.
  2. Show lookup-table hit rate on train seeds vs holdout seeds.
  3. Run the independent verifier -> holdout LER vs plain MWPM.
  4. Render the three verifier guards as a checklist.
  5. Print the verdict banner and the Pareto consequence.

Exits 0 when the verifier correctly rejects the cheat (verdict in
{FAILED, SUSPICIOUS}); non-zero if the cheat would have been VERIFIED.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from autoqec.cheaters.memorize import MemorizerPredecoder, train_memorizer  # noqa: E402
from autoqec.envs.schema import load_env_yaml  # noqa: E402
from autoqec.eval.independent_eval import independent_verify  # noqa: E402
from autoqec.runner.data import load_code_artifacts, sample_syndromes  # noqa: E402


BAR_WIDTH = 34


@dataclass(frozen=True)
class _PhaseTwoStats:
    seen_hit_rate: float
    fresh_train_hit_rate: float
    holdout_hit_rate: float
    n_probes: int


def _display_path(path: Path) -> str:
    """Relative to repo if possible, absolute otherwise -- for log lines that
    want short paths in the common case but must not crash when --run-dir
    points outside the repo (e.g. tests using tmp_path)."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _ascii_bar(value: float, vmax: float, width: int = BAR_WIDTH) -> str:
    if vmax <= 0:
        return "." * width
    frac = max(0.0, min(1.0, value / vmax))
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def _phase_header(title: str, subtitle: str = "") -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 72)


def phase1_construct_cheat(
    env, artifacts, run_dir: Path, n_shots: int
) -> tuple[dict, MemorizerPredecoder, np.ndarray]:
    _phase_header(
        "PHASE 1 - Construct the memorizing cheater",
        "Reads train-seed syndromes and builds a lookup table (syndrome -> perfect correction).",
    )
    mem = train_memorizer(env, artifacts, n_shots=n_shots)

    # Pull the exact syndromes we just memorized so phase 2 can probe the
    # "seen" bucket accurately. Re-sampling here would give a different set;
    # we store the key list from the table itself.
    seen_keys = np.array(list(mem.table.keys()), dtype=np.int64)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "class_name": "MemorizerPredecoder",
            "model": mem,
            "state_dict": mem.state_dict(),
            "output_mode": "hard_flip",
            "dsl_config": None,
        },
        ckpt_path,
    )

    n_unique = len(mem.table)
    train_lo, train_hi = env.noise.seed_policy.train
    print()
    print(f"  memorized {n_unique:,d} unique (syndrome -> correction) pairs")
    print(f"  drawn from training seeds [{train_lo}..{train_hi}]  (n_shots={n_shots:,d})")
    print(f"  saved checkpoint:  {_display_path(ckpt_path)}")
    print()
    print("  This cheater will ace any syndrome it has literally seen. The")
    print("  interesting question is whether 'train seed range' is enough")
    print("  shared structure for it to generalize, or whether the cheat is")
    print("  bound to exact shots. Phase 2 answers that.")
    info = {
        "n_unique_entries": n_unique,
        "train_seed_range": [train_lo, train_hi],
        "checkpoint_path": str(_display_path(ckpt_path)),
    }
    return info, mem, seen_keys


def phase2_hit_rate(
    env, artifacts, mem: MemorizerPredecoder, seen_keys: np.ndarray, n_probes: int
) -> _PhaseTwoStats:
    _phase_header(
        "PHASE 2 - Lookup-table hit rate: memorized / fresh-train / holdout",
        "Three probes reveal whether the cheat generalizes to seed ranges or is bound to exact shots.",
    )

    def probe(syndromes_np: np.ndarray) -> float:
        if syndromes_np.shape[0] == 0:
            return 0.0
        hits = 0
        for row in syndromes_np:
            if tuple(row.tolist()) in mem.table:
                hits += 1
        return hits / syndromes_np.shape[0]

    # Bucket A: shots the memorizer has literally seen (its home turf).
    seen_sample = seen_keys if seen_keys.shape[0] <= n_probes else seen_keys[:n_probes]
    seen_hr = probe(seen_sample)

    # Bucket B: fresh draws from the same train seed range.
    fresh_train_syn, _ = sample_syndromes(env, artifacts, env.noise.seed_policy.train, n_probes)
    fresh_train_hr = probe(fresh_train_syn.numpy())

    # Bucket C: fresh draws from the holdout seed range (the fair test).
    holdout_syn, _ = sample_syndromes(env, artifacts, env.noise.seed_policy.holdout, n_probes)
    holdout_hr = probe(holdout_syn.numpy())

    print()
    print(f"  {'memorized shots':<22}  {seen_hr*100:5.1f}%  [{_ascii_bar(seen_hr, 1.0)}]")
    print(f"  {'fresh train-seed shots':<22}  {fresh_train_hr*100:5.1f}%  [{_ascii_bar(fresh_train_hr, 1.0)}]")
    print(f"  {'holdout-seed shots':<22}  {holdout_hr*100:5.1f}%  [{_ascii_bar(holdout_hr, 1.0)}]")
    print()
    print("  The cheat is bound to *specific shots*, not to 'train seed range'.")
    print(f"  Even fresh draws from seeds [{env.noise.seed_policy.train[0]}..{env.noise.seed_policy.train[1]}] only")
    print(f"  match {fresh_train_hr*100:.0f}% of the table -- almost the same as holdout's {holdout_hr*100:.0f}%.")
    print("  So the cheat does not cheat by leaking seeds; it cheats by overfitting")
    print("  to exact syndromes. On misses it outputs all-zero, which poisons MWPM:")
    print("  MWPM thinks a predecoder already solved the problem and gets a wrong hint.")
    return _PhaseTwoStats(
        seen_hit_rate=seen_hr,
        fresh_train_hit_rate=fresh_train_hr,
        holdout_hit_rate=holdout_hr,
        n_probes=n_probes,
    )


def phase3_verify(env, run_dir: Path, n_shots: int, n_seeds: int) -> tuple[dict, object]:
    _phase_header(
        "PHASE 3 - Fair-test: independent verifier on holdout seeds",
        "Runs the same guards an admitted Pareto candidate would have to pass.",
    )
    sp = env.noise.seed_policy
    holdout_seeds = list(range(sp.holdout[0], sp.holdout[0] + n_seeds))
    report = independent_verify(
        checkpoint=run_dir / "checkpoint.pt",
        env_spec=env,
        holdout_seeds=holdout_seeds,
        n_shots=n_shots,
    )
    ler_plain = float(ler_plain_from_notes(report.notes))
    ler_pred = float(report.ler_holdout)
    lo, hi = report.ler_holdout_ci

    (run_dir / "verification_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "verification_report.md").write_text(_render_markdown(report, ler_plain), encoding="utf-8")

    vmax = max(ler_plain, ler_pred, 0.01) * 1.15
    print()
    print(f"  {'plain MWPM':<16}  LER = {ler_plain:.4f}                  [{_ascii_bar(ler_plain, vmax)}]")
    print(
        f"  {'memorizer':<16}  LER = {ler_pred:.4f}   95%CI ({lo:.3f},{hi:.3f})"
        f"  [{_ascii_bar(ler_pred, vmax)}]"
    )
    print()
    print(f"  delta_LER (holdout)    = {report.delta_ler_holdout:+.4f}")
    print(f"  ler_shuffled (ablation)= {report.ler_shuffled:.4f}")
    print(f"  paired_eval_bundle_id  = {report.paired_eval_bundle_id}")
    return {
        "ler_plain": ler_plain,
        "ler_pred": ler_pred,
        "ler_ci_lo": float(lo),
        "ler_ci_hi": float(hi),
        "delta_ler": float(report.delta_ler_holdout),
        "ler_shuffled": float(report.ler_shuffled),
        "n_shots": n_shots,
        "n_seeds": n_seeds,
        "paired_eval_bundle_id": report.paired_eval_bundle_id,
        "holdout_seeds_used": report.holdout_seeds_used,
    }, report


def phase4_guards(report, ler_plain: float) -> list[dict]:
    _phase_header(
        "PHASE 4 - Three independent guards",
        "Each guard interrogates a different failure mode. Any failing guard keeps the checkpoint off the Pareto front.",
    )
    delta = float(report.delta_ler_holdout)
    lo, hi = report.ler_holdout_ci
    ci_half = (float(hi) - float(lo)) / 2
    guards = [
        {
            "name": "seed-leakage hygiene",
            "ok": bool(report.seed_leakage_check_ok),
            "explain": "holdout seeds are strictly outside the train/val range",
        },
        {
            "name": "paired bootstrap CI",
            "ok": abs(delta) < ci_half,
            "explain": (
                f"delta_LER = {delta:+.4f}, CI half-width = {ci_half:.4f}; "
                "pass = CI crosses 0; strong-negative delta triggers FAILED"
            ),
        },
        {
            "name": "ablation sanity",
            "ok": bool(report.ablation_sanity_ok),
            "explain": (
                f"shuffled_LER ({report.ler_shuffled:.4f}) must not be meaningfully "
                f"better than predecoder_LER ({report.ler_holdout:.4f})"
            ),
        },
    ]
    print()
    for g in guards:
        mark = "PASS" if g["ok"] else "FAIL"
        print(f"  [{mark}] {g['name']:<26}  {g['explain']}")
    print()
    print("  The interesting guard here is paired-bootstrap CI: the delta is strongly")
    print("  negative AND the CI does not cross 0, so the verifier flags the cheater.")
    return guards


def phase5_verdict(report, run_dir: Path) -> str:
    _phase_header("PHASE 5 - Verdict + Pareto consequence")
    verdict = report.verdict
    if verdict == "VERIFIED":
        banner = (
            "+------------------------------------------------+\n"
            "|  VERDICT: VERIFIED  (THIS SHOULD NEVER HAPPEN) |\n"
            "+------------------------------------------------+"
        )
        consequence = "If you see this, the verifier is broken. DO NOT trust the Pareto front."
    elif verdict == "FAILED":
        banner = (
            "+------------------------------------------------+\n"
            "|  VERDICT: FAILED  (cheat correctly rejected)   |\n"
            "+------------------------------------------------+"
        )
        consequence = (
            "Pareto consequence: this checkpoint is NOT admitted to pareto.json.\n"
            "Branch exp/<run>/<NN>-memorizer would be tagged in fork_graph.json\n"
            "with status=rejected_by_verifier and kept around only as a negative example."
        )
    else:  # SUSPICIOUS
        banner = (
            "+---------------------------------------------------+\n"
            "|  VERDICT: SUSPICIOUS  (cheat kept off Pareto too) |\n"
            "+---------------------------------------------------+"
        )
        consequence = (
            "Pareto consequence: this checkpoint is NOT admitted to pareto.json.\n"
            "SUSPICIOUS means the verifier could not confidently rule in OR out;\n"
            "admission requires a positive, CI-clear Δ_LER."
        )
    print()
    print(banner)
    print()
    print(consequence)
    print()
    print(f"  report json:      {_display_path(run_dir / 'verification_report.json')}")
    print(f"  report markdown:  {_display_path(run_dir / 'verification_report.md')}")
    return verdict


def ler_plain_from_notes(notes: str) -> float:
    for part in notes.replace(",", " ").split():
        if part.startswith("plain_ler="):
            return float(part.split("=", 1)[1])
    return float("nan")


def _render_markdown(report, ler_plain: float) -> str:
    lo, hi = report.ler_holdout_ci
    return (
        "# Verification Report\n\n"
        f"**Verdict:** {report.verdict}\n\n"
        f"- Holdout LER: {report.ler_holdout:.4f}\n"
        f"- Holdout LER CI: ({lo:.4f}, {hi:.4f})\n"
        f"- delta_LER (holdout): {report.delta_ler_holdout:.4f}\n"
        f"- Ablation sanity: {report.ablation_sanity_ok}\n"
        f"- Seed-leakage check: {report.seed_leakage_check_ok}\n"
        f"- Paired eval bundle ID: {report.paired_eval_bundle_id}\n\n"
        f"Notes: {report.notes}\n"
    )


def try_write_png(
    run_dir: Path,
    phase2: _PhaseTwoStats,
    phase3: dict,
) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n  (matplotlib unavailable, skipping PNG)")
        return None

    viz_dir = run_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_path = viz_dir / "scoreboard.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    hit_labels = ["memorized shots", "fresh train", "holdout"]
    hit_values = [
        phase2.seen_hit_rate * 100,
        phase2.fresh_train_hit_rate * 100,
        phase2.holdout_hit_rate * 100,
    ]
    ax1.bar(hit_labels, hit_values, color=["#3b7dd8", "#d8933b", "#d83b3b"])
    ax1.set_ylabel("lookup hit rate (%)")
    ax1.set_ylim(0, 110)
    ax1.set_title("Memorizer table hit rate")
    for i, v in enumerate(hit_values):
        ax1.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10)

    lers = [phase3["ler_plain"], phase3["ler_pred"]]
    err_lo = [0.0, max(0.0, phase3["ler_pred"] - phase3["ler_ci_lo"])]
    err_hi = [0.0, max(0.0, phase3["ler_ci_hi"] - phase3["ler_pred"])]
    ax2.bar(
        ["plain MWPM", "memorizer"],
        lers,
        color=["#3b7dd8", "#d83b3b"],
        yerr=[err_lo, err_hi],
        capsize=6,
    )
    ax2.set_ylabel("holdout logical error rate")
    ax2.set_title("Fair-test LER (lower is better)")
    for i, v in enumerate(lers):
        ax2.text(i, v + max(lers) * 0.03, f"{v:.4f}", ha="center", fontsize=10)

    fig.suptitle("Reward-hacking demo scoreboard  (cheat's home turf vs fair test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print()
    print(f"  PNG written: {_display_path(out_path)}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env-yaml",
        default=str(_REPO_ROOT / "autoqec/envs/builtin/surface_d5_depol.yaml"),
    )
    p.add_argument("--run-dir", default=str(_REPO_ROOT / "runs/demo-4/round_0"))
    p.add_argument("--n-shots", type=int, default=5000)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-probes", type=int, default=2000)
    p.add_argument("--memorize-shots", type=int, default=10_000)
    p.add_argument("--no-png", action="store_true", help="Skip the matplotlib PNG.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env = load_env_yaml(args.env_yaml)
    artifacts = load_code_artifacts(env)
    run_dir = Path(args.run_dir).resolve()

    print("AutoQEC reward-hacking demo - narrated walkthrough")
    print(f"env:     {args.env_yaml}")
    print(f"run_dir: {_display_path(run_dir)}")
    print(f"budget:  n_shots={args.n_shots}  n_seeds={args.n_seeds}  n_probes={args.n_probes}")

    phase1, mem, seen_keys = phase1_construct_cheat(
        env, artifacts, run_dir, args.memorize_shots
    )
    phase2 = phase2_hit_rate(env, artifacts, mem, seen_keys, args.n_probes)
    phase3_dict, report = phase3_verify(env, run_dir, args.n_shots, args.n_seeds)
    guards = phase4_guards(report, phase3_dict["ler_plain"])
    verdict = phase5_verdict(report, run_dir)

    png_path: Path | None = None
    if not args.no_png:
        png_path = try_write_png(run_dir, phase2, phase3_dict)

    summary = {
        "verdict": verdict,
        "phase1": phase1,
        "phase2": {
            "seen_hit_rate": phase2.seen_hit_rate,
            "fresh_train_hit_rate": phase2.fresh_train_hit_rate,
            "holdout_hit_rate": phase2.holdout_hit_rate,
            "n_probes": phase2.n_probes,
        },
        "phase3": phase3_dict,
        "phase4_guards": guards,
        "png_path": _display_path(png_path) if png_path else None,
    }
    (run_dir / "present_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Exit semantics mirror run.sh so CI can reuse the same invocation.
    if verdict == "VERIFIED":
        print(
            "\nFAILURE: memorizer was VERIFIED. The verifier is broken or the",
            file=sys.stderr,
        )
        print("seed policy lets train and holdout overlap.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Make torch deterministic-ish for demos so the scoreboard is stable.
    torch.manual_seed(0)
    np.random.seed(0)
    raise SystemExit(main())
