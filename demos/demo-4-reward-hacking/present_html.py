"""Generate a self-contained HTML visualization of the reward-hacking demo.

Reads the verifier's JSON report from a demo-4 run directory (and optionally
the richer phase-by-phase summary produced by ``present.py``), then writes a
static HTML file with inline SVG bar charts and opens it in the default
browser. Designed to be invoked at the tail of ``run.sh`` / ``present.sh``.

Required input
  runs/demo-4/round_0/verification_report.json

Optional input
  runs/demo-4/round_0/present_summary.json   adds the Phase 2 hit-rate card

Output
  runs/demo-4/round_0/visualizations/index.html   (self-contained, file:// safe)

Env
  AUTOQEC_NO_OPEN=1   never auto-open the browser (use in CI / headless runs)
  CI=true             treated the same as AUTOQEC_NO_OPEN=1
"""
from __future__ import annotations

import argparse
import html
import json
import os
import webbrowser
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Guard:
    name: str
    ok: bool
    explain: str


def _load_required_report(run_dir: Path) -> dict:
    p = run_dir / "verification_report.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"verification_report.json not found at {p}. "
            "Run demos/demo-4-reward-hacking/run.sh first."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def _load_optional_summary(run_dir: Path) -> dict | None:
    p = run_dir / "present_summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _plain_ler_from_notes(notes: str) -> float | None:
    for part in notes.replace(",", " ").split():
        if part.startswith("plain_ler="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _derive_guards(report: dict) -> list[_Guard]:
    delta = float(report.get("delta_ler_holdout", 0.0))
    lo, hi = report.get("ler_holdout_ci", [0.0, 0.0])
    ci_half = (float(hi) - float(lo)) / 2.0
    return [
        _Guard(
            name="seed-leakage hygiene",
            ok=bool(report.get("seed_leakage_check_ok", True)),
            explain="holdout seeds are strictly outside the train/val range",
        ),
        _Guard(
            name="paired bootstrap CI",
            ok=abs(delta) < ci_half,
            explain=(
                f"Δ_LER = {delta:+.4f}, CI half-width = {ci_half:.4f}; "
                "pass = CI crosses 0; strong-negative Δ triggers FAILED"
            ),
        ),
        _Guard(
            name="ablation sanity",
            ok=bool(report.get("ablation_sanity_ok", True)),
            explain=(
                f"shuffled_LER ({float(report.get('ler_shuffled', 0.0)):.4f}) must not be "
                f"meaningfully better than predecoder_LER "
                f"({float(report.get('ler_holdout', 0.0)):.4f})"
            ),
        ),
    ]


_CSS = """
:root{color-scheme:dark;--bg:#0f172a;--panel:#1e293b;--border:#334155;
--text:#e2e8f0;--muted:#94a3b8;--plain:#3b7dd8;--train:#d8933b;--cheat:#d83b3b;
--ok:#22c55e;--warn:#eab308;--bad:#b91c1c}
*{box-sizing:border-box}
body{margin:0;font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;
background:var(--bg);color:var(--text);line-height:1.5}
.container{max-width:1080px;margin:0 auto;padding:40px 24px 80px}
header{margin-bottom:28px}
h1{margin:0 0 6px;font-size:28px;font-weight:700}
.sub{color:var(--muted);font-size:14px}
.sub code{background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px}
.banner{margin-top:20px;padding:16px 22px;border-radius:12px;
display:flex;align-items:center;gap:18px;font-size:16px;flex-wrap:wrap}
/* banner color tracks DEMO OUTCOME, not the raw verdict label:
   FAILED = cheat rejected = GOOD (green); VERIFIED = cheat admitted = verifier broken (red). */
.banner.FAILED{background:rgba(34,197,94,0.18);border:1px solid var(--ok)}
.banner.SUSPICIOUS{background:rgba(234,179,8,0.18);border:1px solid var(--warn)}
.banner.VERIFIED{background:rgba(185,28,28,0.22);border:1px solid var(--bad)}
.pill{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;
font-weight:700;letter-spacing:0.04em;padding:4px 10px;border-radius:999px;
background:rgba(0,0,0,0.35);text-transform:uppercase}
.banner.FAILED .pill{color:#bbf7d0}
.banner.SUSPICIOUS .pill{color:#fde68a}
.banner.VERIFIED .pill{color:#fecaca}
.grid{display:grid;gap:18px;grid-template-columns:repeat(2,1fr)}
.grid.full{grid-template-columns:1fr;margin-top:18px}
@media(max-width:720px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:12px;
padding:20px 22px}
.card h2{margin:0 0 4px;font-size:13px;font-weight:700;text-transform:uppercase;
letter-spacing:0.08em;color:var(--muted)}
.card h3{margin:0 0 12px;font-size:18px;font-weight:600}
.card p{margin:0 0 8px;font-size:14px}
.card p:last-child{margin-bottom:0}
.card .note{color:var(--muted);font-size:13px;font-style:italic}
.card code{font-family:ui-monospace,Menlo,Consolas,monospace;
background:rgba(255,255,255,0.06);padding:1px 5px;border-radius:4px;font-size:12.5px}
.bar-row{display:grid;grid-template-columns:150px 1fr 210px;align-items:center;
gap:12px;margin-bottom:8px}
.bar-label{font-size:13px;color:var(--muted)}
.bar-value{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:13px;
text-align:right}
.bar-track{background:rgba(255,255,255,0.06);border-radius:6px;
position:relative;height:22px}
.bar-fill{height:22px;border-radius:6px;display:block}
.bar-fill.plain{background:var(--plain)}
.bar-fill.train{background:var(--train)}
.bar-fill.cheat{background:var(--cheat)}
.bar-ci{position:absolute;top:3px;bottom:3px;pointer-events:none;
border-left:2px solid rgba(255,255,255,0.85);
border-right:2px solid rgba(255,255,255,0.85)}
.bar-ci::before{content:"";position:absolute;top:50%;left:0;right:0;
border-top:1px solid rgba(255,255,255,0.85)}
.guards{display:flex;flex-direction:column;gap:10px}
.guard{display:grid;grid-template-columns:68px 1fr;gap:14px;padding:10px 12px;
border-radius:8px;background:rgba(0,0,0,0.25)}
.guard .tag{font-family:ui-monospace,Menlo,Consolas,monospace;font-weight:700;
font-size:12px;text-align:center;padding:5px 0;border-radius:4px;align-self:start}
.guard .tag.pass{background:rgba(34,197,94,0.18);color:#86efac}
.guard .tag.fail{background:rgba(185,28,28,0.22);color:#fca5a5}
.guard .name{font-weight:600;font-size:14px}
.guard .explain{color:var(--muted);font-size:12.5px;margin-top:2px;
font-family:ui-monospace,Menlo,Consolas,monospace}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:13px;
margin-top:14px}
.kv dt{color:var(--muted)}
.kv dd{margin:0;font-family:ui-monospace,Menlo,Consolas,monospace;
word-break:break-all}
footer{margin-top:36px;color:var(--muted);font-size:12px;text-align:center}
footer code{font-family:ui-monospace,Menlo,Consolas,monospace}
footer a{color:var(--muted);text-decoration:underline dotted}
footer .row{margin-top:6px}
"""


def _bar_row(label: str, value: float, vmax: float, color_cls: str, rhs: str) -> str:
    frac = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{html.escape(label)}</div>'
        f'<div class="bar-track"><span class="bar-fill {color_cls}" '
        f'style="width:{frac * 100:.2f}%"></span></div>'
        f'<div class="bar-value">{html.escape(rhs)}</div>'
        f"</div>"
    )


def _bar_row_with_ci(
    label: str, value: float, ci_lo: float, ci_hi: float, vmax: float,
    color_cls: str, rhs: str,
) -> str:
    def pct(v: float) -> float:
        return 0.0 if vmax <= 0 else max(0.0, min(1.0, v / vmax)) * 100
    w, lo, hi = pct(value), pct(ci_lo), pct(ci_hi)
    ci_w = max(0.0, hi - lo)
    return (
        f'<div class="bar-row">'
        f'<div class="bar-label">{html.escape(label)}</div>'
        f'<div class="bar-track">'
        f'<span class="bar-fill {color_cls}" style="width:{w:.2f}%"></span>'
        f'<span class="bar-ci" style="left:{lo:.2f}%;width:{ci_w:.2f}%"></span>'
        f"</div>"
        f'<div class="bar-value">{html.escape(rhs)}</div>'
        f"</div>"
    )


def _intro_card() -> str:
    return """
    <section class="card">
      <h2>Phase 1 · what is reward hacking here?</h2>
      <h3>Memorize training syndromes. Ace the train set. Fail the fair test.</h3>
      <p>The <strong>MemorizerPredecoder</strong> cheat reads syndromes from the
      <em>training seed range</em>, builds a lookup table
      <code>(syndrome &rarr; perfect correction)</code>, and returns the stored
      correction at inference time. On its home turf it looks flawless.</p>
      <p>On holdout it misses every lookup and returns all-zero, which gives the
      downstream MWPM a broken hint and decodes strictly worse than plain MWPM
      alone. That is the gap the independent verifier is built to catch.</p>
    </section>"""


def _pareto_card(verdict: str) -> str:
    if verdict == "VERIFIED":
        body = (
            "If you see this banner the verifier is broken &mdash; the memorizer "
            "has no fair-test signal and must never be admitted. Investigate "
            "<code>autoqec/eval/independent_eval.py</code> before shipping."
        )
    elif verdict == "FAILED":
        body = (
            "This checkpoint is <strong>not</strong> admitted to "
            "<code>pareto.json</code>. The corresponding experiment branch would be "
            "tagged <code>status=rejected_by_verifier</code> in "
            "<code>fork_graph.json</code> and kept around only as a negative "
            "example for the Ideator."
        )
    else:
        body = (
            "This checkpoint is <strong>not</strong> admitted to "
            "<code>pareto.json</code>. SUSPICIOUS means the verifier could not "
            "confidently rule the cheat in or out &mdash; admission to the Pareto "
            "archive requires a positive, CI-clear Δ_LER."
        )
    return f"""
    <section class="card">
      <h2>Phase 5 · Pareto consequence</h2>
      <h3>What happens to this branch in the archive?</h3>
      <p>{body}</p>
    </section>"""


def _hit_rate_card(phase2: dict) -> str:
    seen = float(phase2["seen_hit_rate"]) * 100
    fresh = float(phase2["fresh_train_hit_rate"]) * 100
    hold = float(phase2["holdout_hit_rate"]) * 100
    n = int(phase2.get("n_probes", 0))
    return f"""
    <section class="card">
      <h2>Phase 2 · lookup-table hit rate</h2>
      <h3>The cheat is bound to <em>specific shots</em>, not to the train seed range.</h3>
      {_bar_row("memorized shots", seen, 100.0, "cheat", f"{seen:.1f}%")}
      {_bar_row("fresh train-seed", fresh, 100.0, "train", f"{fresh:.1f}%")}
      {_bar_row("holdout-seed", hold, 100.0, "plain", f"{hold:.1f}%")}
      <p class="note">Fresh draws from the train seed range hit the table at roughly
      the same rate as holdout draws &mdash; so the memorizer is not leaking seeds, it
      is overfitting to exact syndromes. On a miss it outputs zero, which poisons the
      downstream MWPM solver (over <strong>{n:,d}</strong> probes per bucket).</p>
    </section>"""


def _hit_rate_placeholder() -> str:
    return """
    <section class="card">
      <h2>Phase 2 · lookup-table hit rate</h2>
      <h3>not computed in this run</h3>
      <p class="note">Run <code>bash demos/demo-4-reward-hacking/present.sh</code>
      (the narrated mode) to populate <code>present_summary.json</code> with
      memorized / fresh-train / holdout hit-rate probes for this card.</p>
    </section>"""


def _ler_card(report: dict, plain_ler: float) -> str:
    pred = float(report["ler_holdout"])
    lo, hi = report["ler_holdout_ci"]
    delta = float(report["delta_ler_holdout"])
    vmax = max(plain_ler, pred, float(hi), 0.01) * 1.15
    return f"""
    <section class="card">
      <h2>Phase 3 · fair-test LER on holdout seeds</h2>
      <h3>plain MWPM vs memorizer &mdash; lower is better</h3>
      {_bar_row("plain MWPM", plain_ler, vmax, "plain", f"{plain_ler:.4f}")}
      {_bar_row_with_ci(
          "memorizer", pred, float(lo), float(hi), vmax, "cheat",
          f"{pred:.4f}   95%CI [{float(lo):.3f}, {float(hi):.3f}]",
      )}
      <dl class="kv">
        <dt>Δ_LER (holdout)</dt><dd>{delta:+.4f}</dd>
        <dt>ler_shuffled (ablation)</dt><dd>{float(report.get("ler_shuffled", 0.0)):.4f}</dd>
        <dt>paired bundle id</dt><dd>{html.escape(str(report.get("paired_eval_bundle_id", "—")))}</dd>
      </dl>
    </section>"""


def _ler_placeholder() -> str:
    return """
    <section class="card">
      <h2>Phase 3 · fair-test LER</h2>
      <h3>plain_ler missing from notes</h3>
      <p class="note">Expected a <code>plain_ler=</code> token in
      <code>verification_report.json.notes</code>. Regenerate the report with
      a newer verifier to render this card.</p>
    </section>"""


def _guards_card(guards: list[_Guard]) -> str:
    rows = []
    for g in guards:
        tag = (
            '<span class="tag pass">PASS</span>' if g.ok
            else '<span class="tag fail">FAIL</span>'
        )
        rows.append(
            f'<div class="guard">{tag}'
            f'<div><div class="name">{html.escape(g.name)}</div>'
            f'<div class="explain">{html.escape(g.explain)}</div></div></div>'
        )
    return f"""
    <section class="card">
      <h2>Phase 4 · three independent verifier guards</h2>
      <h3>Any failing guard keeps the checkpoint off the Pareto front.</h3>
      <div class="guards">{''.join(rows)}</div>
    </section>"""


def render_html(
    report: dict, summary: dict | None, plain_ler: float | None, run_dir: Path,
) -> str:
    verdict = str(report.get("verdict", "UNKNOWN")).upper()
    phrase = {
        "FAILED": "cheat correctly rejected",
        "SUSPICIOUS": "cheat kept off the Pareto front",
        "VERIFIED": "THIS SHOULD NEVER HAPPEN &mdash; the verifier is broken",
    }.get(verdict, "unknown verdict")

    try:
        run_dir_display = str(run_dir.relative_to(_REPO_ROOT))
    except ValueError:
        run_dir_display = str(run_dir)

    hit_rate_html = (
        _hit_rate_card(summary["phase2"])
        if summary and isinstance(summary.get("phase2"), dict)
        else _hit_rate_placeholder()
    )
    ler_html = _ler_card(report, plain_ler) if plain_ler is not None else _ler_placeholder()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AutoQEC · Demo 4 · Reward-hacking detection</title>
<style>{_CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Reward-hacking detection</h1>
    <div class="sub">AutoQEC &middot; Demo 4 &middot; run dir <code>{html.escape(run_dir_display)}</code></div>
    <div class="banner {html.escape(verdict)}">
      <span class="pill">verdict: {html.escape(verdict)}</span>
      <span>{phrase}</span>
    </div>
  </header>

  <div class="grid">
    {_intro_card()}
    {_pareto_card(verdict)}
  </div>

  <div class="grid full">
    {hit_rate_html}
    {ler_html}
    {_guards_card(_derive_guards(report))}
  </div>

  <footer>
    <div class="row"><code>holdout seeds: {html.escape(str(report.get("holdout_seeds_used") or []))}</code></div>
    <div class="row">{html.escape(str(report.get("notes", "")))}</div>
    <div class="row">
      <a href="../verification_report.md">verification_report.md</a>
      &nbsp;&middot;&nbsp;
      <a href="../verification_report.json">verification_report.json</a>
    </div>
  </footer>
</div>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default=str(_REPO_ROOT / "runs/demo-4/round_0"),
        help="directory containing verification_report.json",
    )
    p.add_argument("--out", default=None, help="override output HTML path")
    p.add_argument(
        "--no-open", action="store_true",
        help="skip browser auto-open (also honors AUTOQEC_NO_OPEN / CI env vars)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    report = _load_required_report(run_dir)
    summary = _load_optional_summary(run_dir)
    plain_ler = _plain_ler_from_notes(str(report.get("notes", "")))

    out_path = (
        Path(args.out).resolve() if args.out else run_dir / "visualizations" / "index.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_html(report, summary, plain_ler, run_dir), encoding="utf-8",
    )
    print(f"  HTML written: {out_path}")

    should_open = (
        not args.no_open
        and not os.environ.get("AUTOQEC_NO_OPEN")
        and not os.environ.get("CI")
    )
    if should_open:
        try:
            webbrowser.open(out_path.as_uri())
            print("  opened in default browser (set AUTOQEC_NO_OPEN=1 to skip)")
        except Exception as exc:
            print(f"  could not auto-open browser: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
