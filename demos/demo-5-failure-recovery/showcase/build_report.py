from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import platform
import subprocess
from typing import Any


REQUIRED_CAUSES = {"compile_error", "nan_loss", "oom"}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_text(path: Path, limit: int = 3000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _rel(path: Path, root: Path) -> str:
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


def _classify(metrics: dict[str, Any] | None, train_log: str) -> str:
    status = str((metrics or {}).get("status") or "").lower()
    reason = str((metrics or {}).get("status_reason") or "").lower()
    log = train_log.lower()
    if status == "compile_error" or "validation failed" in reason:
        return "compile_error"
    if status == "killed_by_safety" and "nan" in reason:
        return "nan_loss"
    if "out of memory" in reason or "out of memory" in log:
        return "oom"
    return "unknown"


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {"path": _rel(path, root), "exists": path.exists()}


def _extract_markdown_section(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    capture = False
    section_lines: list[str] = []
    target = heading.strip().lower()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            current = stripped.removeprefix("## ").strip().lower()
            if capture:
                break
            capture = current == target or current.startswith(f"{target} ")
            continue
        if capture:
            section_lines.append(line)
    return "\n".join(section_lines).strip()


def _diagnosis_sections(diagnosis_text: str) -> dict[str, str]:
    recommended_fix = _extract_markdown_section(diagnosis_text, "Recommended Fix")
    recommended_fix = recommended_fix.split("\n**Note:**", 1)[0].strip()
    return {
        "root_cause": _extract_markdown_section(diagnosis_text, "Root Cause"),
        "evidence": _extract_markdown_section(diagnosis_text, "Evidence"),
        "recommended_fix": recommended_fix,
    }


def _collect_rounds(root: Path) -> list[dict[str, Any]]:
    run_dir = root / "runs" / "demo-5"
    rounds = []
    for round_dir in sorted(run_dir.glob("round_*"), key=lambda item: item.name):
        metrics_path = round_dir / "metrics.json"
        train_log_path = round_dir / "train.log"
        config_path = round_dir / "config.yaml"
        diagnosis_path = round_dir / "diagnosis.md"
        metrics = _read_json(metrics_path)
        train_log = _read_text(train_log_path)
        diagnosis_text = _read_text(diagnosis_path, 3000)
        root_cause = _classify(metrics, train_log)
        rounds.append(
            {
                "round": round_dir.name,
                "root_cause": root_cause,
                "status": (metrics or {}).get("status", "missing_metrics"),
                "status_reason": (metrics or {}).get("status_reason", ""),
                "train_log_excerpt": train_log[-500:],
                "diagnosis_excerpt": diagnosis_text[-1000:],
                "diagnosis": _diagnosis_sections(diagnosis_text),
                "artifacts": {
                    "metrics.json": _artifact(metrics_path, root),
                    "train.log": _artifact(train_log_path, root),
                    "config.yaml": _artifact(config_path, root),
                    "diagnosis.md": _artifact(diagnosis_path, root),
                },
            }
        )
    return rounds


def build_summary(root: Path, output_dir: Path, phases: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = _collect_rounds(root)
    causes = {str(item["root_cause"]) for item in rounds}
    failed_phases = [str(phase["name"]) for phase in phases if int(phase.get("exit_code", 1)) != 0]
    if not REQUIRED_CAUSES.issubset(causes):
        failed_phases.append("missing-diagnosis-evidence")
    status = "pass" if not failed_phases else "fail"
    return {
        "status": status,
        "healthy_line": "demo5 showcase healthy" if status == "pass" else f"needs triage: {', '.join(failed_phases)}",
        "output_dir": _rel(output_dir, root),
        "git_sha": _git_sha(root),
        "python": platform.python_version(),
        "phases": phases,
        "failed_phases": failed_phases,
        "rounds": rounds,
    }


def _href(path: str) -> str:
    safe = html.escape(path)
    return f"../../{safe}"


def _artifact_link(name: str, artifact: dict[str, Any]) -> str:
    path = str(artifact["path"])
    chip = "present" if artifact["exists"] else "missing"
    return (
        f"<a class=\"artifact-link\" href=\"{_href(path)}\">"
        f"<span>{html.escape(name)}</span><code>{html.escape(path)}</code><b class=\"chip {chip}\">{chip}</b></a>"
    )


def _render_diagnosis(item: dict[str, Any]) -> str:
    diagnosis = item.get("diagnosis") or {}
    rows = [
        ("Root cause", str(diagnosis.get("root_cause") or "")),
        ("Evidence", str(diagnosis.get("evidence") or "")),
        ("Recommended fix", str(diagnosis.get("recommended_fix") or "")),
    ]
    rendered_rows = "".join(
        f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd>"
        for label, value in rows
        if value.strip()
    )
    if not rendered_rows:
        rendered_rows = "<dt>Diagnosis artifact</dt><dd>See diagnosis.md for details.</dd>"
    return f"""
      <section class="diagnosis-box" aria-label="Agent Diagnosis">
        <h4>Agent Diagnosis</h4>
        <dl>{rendered_rows}</dl>
      </section>
    """


def _render_round_cards(summary: dict[str, Any]) -> str:
    cards = []
    for item in summary["rounds"]:
        links = "".join(_artifact_link(name, artifact) for name, artifact in item["artifacts"].items())
        cards.append(
            f"""
            <article class="round-card">
              <div class="card-topline"><span>{html.escape(item['round'])}</span><b>metrics.status={html.escape(str(item['status']))}</b></div>
              <h3>{html.escape(item['root_cause'])}</h3>
              <p class="reason">{html.escape(str(item['status_reason']))}</p>
              <div class="journey" aria-label="Broken Round to Root Cause to Safe Fix">
                <span>Broken Round</span><i>→</i><span>Root Cause</span><i>→</i><span>Safe Fix</span>
              </div>
              {_render_diagnosis(item)}
              <div class="artifact-grid">{links}</div>
            </article>
            """
        )
    return "\n".join(cards) or "<p class=\"missing\">No Demo 5 rounds were found.</p>"


def _render_matrix(summary: dict[str, Any]) -> str:
    rows = []
    for item in summary["rounds"]:
        evidence = ", ".join(name for name, artifact in item["artifacts"].items() if artifact["exists"])
        rows.append(
            f"<tr><td>{html.escape(item['root_cause'])}</td><td>{html.escape(evidence)}</td><td>Shows AutoQEC can classify this failure and cite local artifacts.</td></tr>"
        )
    return "\n".join(rows)


def _render_html(summary: dict[str, Any]) -> str:
    status_class = "pass" if summary["status"] == "pass" else "fail"
    causes = ", ".join(str(item["root_cause"]) for item in summary["rounds"]) or "none"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Demo 5 Failure Diagnosis Showcase</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#111d31; --line:#284363; --text:#edf6ff; --muted:#9fb4d0; --good:#8ff0bf; --bad:#ff9c9c; --blue:#91c4ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:linear-gradient(135deg,#07111f,#101827); color:var(--text); font:16px/1.55 Inter, ui-sans-serif, system-ui, sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:36px; }}
    a {{ color:var(--blue); }} a:focus {{ outline:3px solid var(--good); outline-offset:3px; }}
    .hero,.panel,.round-card {{ border:1px solid var(--line); border-radius:28px; background:rgba(17,29,49,.92); box-shadow:0 24px 80px #0008; }}
    .hero {{ padding:34px; background:radial-gradient(circle at top left,#1b6eea55,#111d31 48%,#0b1424); }}
    .badge {{ display:inline-flex; min-height:44px; align-items:center; border-radius:999px; padding:0 16px; font-weight:900; text-transform:uppercase; letter-spacing:.08em; }}
    .badge.pass {{ background:#103d2b; color:var(--good); }} .badge.fail {{ background:#4c1d22; color:var(--bad); }}
    h1 {{ margin:18px 0 8px; font-size:clamp(34px,6vw,58px); line-height:1.02; }} h2 {{ margin:42px 0 16px; font-size:28px; }}
    .sub {{ color:var(--muted); max-width:760px; }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-top:24px; }}
    .stat {{ border:1px solid var(--line); border-radius:18px; padding:16px; background:#0c1728; min-height:88px; }} .stat b {{ display:block; font-size:24px; }} .stat span {{ color:var(--muted); }}
    .view-box {{ margin-top:20px; border:1px dashed #47688f; border-radius:18px; padding:16px; background:#091524; }} code {{ color:#b9f6ca; overflow-wrap:anywhere; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:18px; }} .round-card {{ padding:22px; }}
    .card-topline {{ display:flex; gap:10px; justify-content:space-between; align-items:center; color:var(--good); font-size:13px; text-transform:uppercase; font-weight:900; }}
    .round-card h3 {{ font-size:28px; margin:14px 0 4px; }} .reason {{ color:#dce8ff; min-height:48px; }}
    .journey {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:16px 0; }} .journey span {{ min-height:44px; display:inline-flex; align-items:center; border-radius:14px; padding:0 12px; background:#172a49; border:1px solid #385d8d; }} .journey i {{ color:var(--good); font-style:normal; font-weight:900; }}
    .diagnosis-box {{ margin:16px 0; border:1px solid #36577f; border-radius:16px; background:#0a1628; padding:14px; }} .diagnosis-box h4 {{ margin:0 0 10px; color:var(--blue); }} .diagnosis-box dl {{ margin:0; display:grid; gap:10px; }} .diagnosis-box dt {{ color:var(--muted); font-weight:800; }} .diagnosis-box dd {{ margin:0; white-space:pre-wrap; }} .artifact-grid {{ display:grid; gap:10px; }}
    .artifact-link {{ min-height:54px; display:grid; grid-template-columns:110px 1fr auto; gap:10px; align-items:center; padding:10px 12px; border-radius:14px; background:#0b1627; border:1px solid #253b5c; text-decoration:none; }}
    .chip {{ border-radius:999px; padding:4px 9px; font-size:12px; }} .chip.present {{ background:#103d2b; color:var(--good); }} .chip.missing {{ background:#4c1d22; color:var(--bad); }}
    .panel {{ padding:22px; }} table {{ width:100%; border-collapse:collapse; }} th,td {{ border-bottom:1px solid var(--line); padding:14px; text-align:left; vertical-align:top; }} th {{ color:var(--blue); }}
    @media (max-width:700px) {{ main {{ padding:18px; }} .artifact-link {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <span class="badge {status_class}">{html.escape(summary['status'])}</span>
    <h1>Demo 5 Failure Diagnosis Showcase</h1>
    <p class="sub">Loaded {len(summary['rounds'])} diagnosis.md artifacts from <code>runs/demo-5</code>. Root causes shown in this report: {html.escape(causes)}.</p>
    <div class="stats">
      <div class="stat"><span>Result</span><b>{html.escape(summary['healthy_line'])}</b></div>
      <div class="stat"><span>Git SHA</span><b>{html.escape(summary['git_sha'])}</b></div>
      <div class="stat"><span>Python</span><b>{html.escape(summary['python'])}</b></div>
    </div>
    <div class="view-box">
      <strong>How to view this report</strong>
      <p>For working artifact links, serve from the repo root: <code>python -m http.server 8767 --bind 127.0.0.1</code>, then open <code>http://127.0.0.1:8767/runs/demo-5-showcase/report.html</code>.</p>
    </div>
  </section>
  <h2>Broken Round → Root Cause → Safe Fix</h2>
  <section class="cards">{_render_round_cards(summary)}</section>
  <h2>Claim → Evidence → Why it matters</h2>
  <section class="panel"><table><thead><tr><th>Claim</th><th>Evidence</th><th>Why it matters</th></tr></thead><tbody>{_render_matrix(summary)}</tbody></table></section>
</main>
</body>
</html>
"""


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Demo 5 Failure Diagnosis Showcase",
        "",
        f"Status: **{summary['status']}**",
        f"Result: `{summary['healthy_line']}`",
        "",
        "## Diagnosed Rounds",
        "",
    ]
    for item in summary["rounds"]:
        lines.append(f"- `{item['round']}`: `{item['root_cause']}` — {item['status_reason']}")
    lines.extend(["", summary["healthy_line"], ""])
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
    parser = argparse.ArgumentParser(description="Build the Demo 5 visual showcase report.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="runs/demo-5-showcase")
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
