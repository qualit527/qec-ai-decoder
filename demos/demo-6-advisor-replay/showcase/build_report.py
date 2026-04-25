from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any

from autoqec.eval.schema import VerifyReport

FLOAT_ABS_TOL = 1e-6
FLOAT_REL_TOL = 1e-6


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(float(expected), float(actual), rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(_same(left, right) for left, right in zip(expected, actual, strict=True))
    return expected == actual


def _comparison_fields() -> list[str]:
    return list(VerifyReport.model_fields)


def _comparison(original: dict[str, Any] | None, replay: dict[str, Any] | None) -> tuple[bool, list[dict[str, Any]]]:
    if original is None or replay is None:
        return False, []
    rows = [
        {"field": field, "original": original.get(field), "replay": replay.get(field), "match": _same(original.get(field), replay.get(field))}
        for field in _comparison_fields()
    ]
    return all(row["match"] for row in rows), rows


def _package_for_round(root: Path, round_dir: Path | None) -> Path | None:
    if round_dir is None:
        return None
    run_id = round_dir.parent.name
    package_path = root / "runs" / f"{run_id}.tar.gz"
    return package_path if package_path.exists() else None


def _collect_replay(root: Path) -> dict[str, Any]:
    replay_root = root / "runs" / "demo-6-replay"
    original_path = _latest(list(replay_root.glob("**/verification_report.original.json")))
    replay_path = original_path.with_name("verification_report.json") if original_path is not None else None
    round_dir = replay_path.parent if replay_path is not None else None
    manifest_path = round_dir / "artifact_manifest.json" if round_dir is not None else None
    package_path = _package_for_round(root, round_dir)
    original = _read_json(original_path)
    replay = _read_json(replay_path)
    reports_match, rows = _comparison(original, replay)
    return {
        "reports_match": reports_match,
        "comparison": rows,
        "comparison_matches": sum(1 for row in rows if row["match"]),
        "comparison_total": len(rows),
        "manifest": _rel(manifest_path, root),
        "manifest_exists": manifest_path.exists() if manifest_path is not None else False,
        "package": _rel(package_path, root),
        "package_exists": package_path.exists() if package_path is not None else False,
        "original_report": _rel(original_path, root),
        "original_report_exists": original_path.exists() if original_path is not None else False,
        "replay_report": _rel(replay_path, root),
        "replay_report_exists": replay_path.exists() if replay_path is not None else False,
        "round_dir": _rel(round_dir, root),
        "verdict": replay.get("verdict") if replay else None,
    }


def build_summary(root: Path, output_dir: Path, phases: list[dict[str, Any]]) -> dict[str, Any]:
    replay = _collect_replay(root)
    failed_phases = [str(phase["name"]) for phase in phases if int(phase.get("exit_code", 1)) != 0]
    if not replay["reports_match"]:
        failed_phases.append("verification-report-drift")
    for key in ("manifest_exists", "package_exists", "original_report_exists", "replay_report_exists"):
        if not replay[key]:
            failed_phases.append(key.removesuffix("_exists"))
    status = "pass" if not failed_phases else "fail"
    return {
        "status": status,
        "healthy_line": "demo6 showcase healthy" if status == "pass" else f"needs triage: {', '.join(failed_phases)}",
        "output_dir": _rel(output_dir, root),
        "git_sha": _git_sha(root),
        "python": platform.python_version(),
        "phases": phases,
        "failed_phases": failed_phases,
        **replay,
    }


def _href(path: str | None) -> str:
    if not path:
        return "#"
    return f"../../{html.escape(path)}"


def _artifact_card(label: str, path: str | None, exists: bool) -> str:
    status_class = "present" if exists else "absent"
    safe_path = html.escape(path or "path=null")
    status_text = f"exists={str(exists).lower()}"
    return f"<a class=\"artifact-card\" href=\"{_href(path)}\"><span>{html.escape(label)}</span><code>{safe_path}</code><b class=\"chip {status_class}\">{status_text}</b></a>"


def _render_diff(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary["comparison"]:
        state = "match" if item["match"] else "mismatch"
        status = f"match={str(item['match']).lower()}"
        rows.append(
            f"<tr class=\"{state}\"><td>{html.escape(item['field'])}</td><td>{html.escape(json.dumps(item['original'], ensure_ascii=False))}</td><td>{html.escape(json.dumps(item['replay'], ensure_ascii=False))}</td><td>{status}</td></tr>"
        )
    return "\n".join(rows)


def _comparison_line(summary: dict[str, Any]) -> str:
    return (
        f"Compared {summary['comparison_total']} verifier fields from "
        "verification_report.original.json and verification_report.json."
    )


def _render_html(summary: dict[str, Any]) -> str:
    status_class = "pass" if summary["status"] == "pass" else "fail"
    comparison_score = f"{summary['comparison_matches']} / {summary['comparison_total']} fields match"
    reports_match = f"reports_match={str(summary['reports_match']).lower()}"
    round_dir = html.escape(str(summary.get("round_dir") or "round_dir=null"))
    original_report = html.escape(str(summary.get("original_report") or "original_report=null"))
    replay_report = html.escape(str(summary.get("replay_report") or "replay_report=null"))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Demo 6 Advisor Replay Showcase</title>
  <style>
    :root {{ color-scheme: dark; --bg:#070d1a; --panel:#111a2f; --line:#2d4568; --text:#edf6ff; --muted:#9fb4d0; --good:#8ff0bf; --bad:#ff9c9c; --blue:#9ec5ff; }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at top left,#1b6eea44,#070d1a 48%); color:var(--text); font:16px/1.55 Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width:1200px; margin:0 auto; padding:36px; }} a {{ color:var(--blue); }} a:focus {{ outline:3px solid var(--good); outline-offset:3px; }}
    .hero,.panel {{ border:1px solid var(--line); border-radius:28px; background:rgba(17,26,47,.94); box-shadow:0 24px 80px #0008; }} .hero {{ padding:34px; }}
    .badge {{ min-height:44px; display:inline-flex; align-items:center; border-radius:999px; padding:0 16px; font-weight:900; text-transform:uppercase; letter-spacing:.08em; }} .badge.pass {{ background:#103d2b; color:var(--good); }} .badge.fail {{ background:#4c1d22; color:var(--bad); }}
    h1 {{ margin:18px 0 8px; font-size:clamp(34px,6vw,58px); line-height:1.02; }} h2 {{ margin:42px 0 16px; font-size:28px; }} .sub {{ color:var(--muted); max-width:780px; }} code {{ color:#b9f6ca; overflow-wrap:anywhere; }}
    .stats,.artifacts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:22px; }} .stat,.artifact-card {{ border:1px solid var(--line); border-radius:18px; background:#0d1728; padding:16px; min-height:96px; }} .stat b {{ display:block; font-size:24px; }} .stat span,.artifact-card span {{ color:var(--muted); }}
    .pipeline {{ width:100%; min-height:150px; }} .pipeline rect {{ fill:#17294a; stroke:#62a0ff; stroke-width:2; rx:14; }} .pipeline text {{ fill:#edf6ff; font-weight:800; font-size:13px; }} .pipeline path {{ stroke:var(--good); stroke-width:3; fill:none; marker-end:url(#arrow); }}
    .artifact-card {{ display:grid; gap:8px; text-decoration:none; min-height:128px; }} .chip {{ width:max-content; border-radius:999px; padding:4px 9px; font-size:12px; font-weight:900; }} .chip.present {{ background:#103d2b; color:var(--good); }} .chip.absent {{ background:#4c1d22; color:var(--bad); }}
    .panel {{ padding:22px; }} .evidence-note {{ display:grid; gap:10px; border-left:5px solid var(--good); padding:14px 18px; background:#0b1627; border-radius:14px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:14px; text-align:left; vertical-align:top; }} th {{ color:var(--blue); }} tr.match td:last-child {{ color:var(--good); font-weight:900; }} tr.mismatch td:last-child {{ color:var(--bad); font-weight:900; }}
    .view-box {{ margin-top:20px; border:1px dashed #47688f; border-radius:18px; padding:16px; background:#091524; }}
    @media (max-width:700px) {{ main {{ padding:18px; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <span class="badge {status_class}">{html.escape(summary['status'])}</span>
    <h1>Demo 6 Advisor Replay Showcase</h1>
    <p class="sub">Loaded replay round <code>{round_dir}</code>. {_comparison_line(summary)}</p>
    <div class="stats">
      <div class="stat"><span>Result</span><b>{html.escape(summary['healthy_line'])}</b></div>
      <div class="stat"><span>Replay verdict</span><b>{html.escape(str(summary.get('verdict')))}</b></div>
      <div class="stat"><span>Comparison</span><b>{html.escape(comparison_score)}</b><code>{html.escape(reports_match)}</code></div>
      <div class="stat"><span>Git SHA</span><b>{html.escape(summary['git_sha'])}</b></div>
    </div>
    <div class="view-box"><strong>How to view this report</strong><p>For working artifact links, serve from repo root: <code>python -m http.server 8768 --bind 127.0.0.1</code>, then open <code>http://127.0.0.1:8768/runs/demo-6-showcase/report.html</code>.</p></div>
  </section>

  <h2>No-LLM Run → Verify → Package → Offline Replay</h2>
  <section class="panel">
    <svg class="pipeline" viewBox="0 0 980 150" role="img" aria-label="No-LLM Run to Verify to Package to Offline Replay">
      <defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#8ff0bf"/></marker></defs>
      <rect x="10" y="42" width="145" height="64"/><text x="38" y="79">No-LLM Run</text><path d="M155 74 H215"/>
      <rect x="215" y="42" width="120" height="64"/><text x="252" y="79">Verify</text><path d="M335 74 H395"/>
      <rect x="395" y="42" width="130" height="64"/><text x="430" y="79">Package</text><path d="M525 74 H585"/>
      <rect x="585" y="42" width="155" height="64"/><text x="615" y="79">Offline Replay</text><path d="M740 74 H800"/>
      <rect x="800" y="42" width="160" height="64"/><text x="842" y="79">Compare</text>
    </svg>
    <div class="evidence-note"><strong>Replay round</strong><code>{round_dir}</code><strong>Comparison sources</strong><code>{original_report}</code><code>{replay_report}</code></div>
  </section>

  <h2>Evidence Artifacts</h2>
  <section class="artifacts">
    {_artifact_card('artifact_manifest.json', summary.get('manifest'), bool(summary.get('manifest_exists')))}
    {_artifact_card('run package tarball', summary.get('package'), bool(summary.get('package_exists')))}
    {_artifact_card('verification_report.original.json', summary.get('original_report'), bool(summary.get('original_report_exists')))}
    {_artifact_card('verification_report.json', summary.get('replay_report'), bool(summary.get('replay_report_exists')))}
  </section>

  <h2>Original vs Replay</h2>
  <section class="panel"><table><thead><tr><th>Field</th><th>Original</th><th>Replay</th><th>Status</th></tr></thead><tbody>{_render_diff(summary)}</tbody></table></section>
</main>
</body>
</html>
"""


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Demo 6 Advisor Replay Showcase",
        "",
        f"Status: **{summary['status']}**",
        f"Result: `{summary['healthy_line']}`",
        f"reports_match: `{summary['reports_match']}`",
        f"Comparison fields: `{summary['comparison_matches']} / {summary['comparison_total']}`",
        "",
        "## Evidence",
        "",
        f"- Manifest: `{summary.get('manifest')}`",
        f"- Package: `{summary.get('package')}`",
        f"- Original report: `{summary.get('original_report')}`",
        f"- Replay report: `{summary.get('replay_report')}`",
        "",
        summary["healthy_line"],
        "",
    ]
    return "\n".join(lines)


def write_reports(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "report.md"
    html_path = output_dir / "report.html"
    summary_path = output_dir / "summary.json"
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    html_path.write_text(_render_html(summary), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return markdown_path, html_path, summary_path


def _load_phases(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("phase file must contain a list")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Demo 6 visual showcase report.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="runs/demo-6-showcase")
    parser.add_argument("--phase-file")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    summary = build_summary(root, output_dir, _load_phases(Path(args.phase_file) if args.phase_file else None))
    markdown_path, html_path, summary_path = write_reports(summary, output_dir)
    print(json.dumps({"status": summary["status"], "report_md": str(markdown_path), "report_html": str(html_path), "summary_json": str(summary_path), "file_url": html_path.resolve().as_uri()}, ensure_ascii=False))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
