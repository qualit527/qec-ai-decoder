"""Generate diagnosis.md for each failed round in a demo run directory."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _line_number(text: str, needle: str) -> int | None:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return None


def generate(round_dir: Path) -> None:
    metrics_path = round_dir / "metrics.json"
    if not metrics_path.exists():
        return
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    log = (round_dir / "train.log").read_text(encoding="utf-8").strip()
    cfg_text = (round_dir / "config.yaml").read_text(encoding="utf-8")

    status = metrics.get("status", "unknown")
    reason = metrics.get("status_reason", "")

    root = status
    evidence: list[str] = []
    fix = ""

    if "nan" in reason.lower() or "nan" in log.lower():
        root = "nan_loss"
        line_no = _line_number(log, "nan")
        if line_no is not None:
            evidence.append(f"train.log:{line_no}: NaN loss values observed")
        evidence.append(f"metrics.json: status_reason={reason}")
        fix = "training:\n  learning_rate: 1.0e-3   # was 10.0\n  loss: focal"
    elif "out of memory" in reason.lower() or "oom" in reason.lower():
        root = "oom"
        line_no = _line_number(log, "out of memory")
        if line_no is not None:
            evidence.append(f"train.log:{line_no}: CUDA out-of-memory traceback")
        evidence.append(f"metrics.json: vram_peak_gb={metrics.get('vram_peak_gb', '?')}, status_reason={reason}")
        fix = "training:\n  batch_size: 16   # was 64\n  profile: dev\ngnn:\n  hidden_dim: 16   # was 32"
    elif "compile_error" in status or "validation" in reason.lower():
        root = "compile_error"
        line_no = _line_number(cfg_text, "hidden_dim: -1")
        if line_no is not None:
            evidence.append(f"config.yaml:{line_no}: hidden_dim: -1")
        evidence.append(f"metrics.json: status_reason={reason}")
        fix = "gnn:\n  hidden_dim: 32   # was -1"

    lines = [
        f"# Diagnosis: {round_dir.name}",
        "",
        "Generated to match the `/diagnose-failure` skill contract.",
        "",
        "## Root Cause",
        f"{root}: {reason}",
        "",
        "## Evidence",
    ]
    for e in evidence:
        lines.append(f"- {e}")
    lines += [
        "",
        "## Recommended Fix (not applied)",
        "```yaml",
        fix,
        "```",
        "",
        "**Note:** The system does not apply fixes automatically. Apply the suggested patch manually.",
    ]

    (round_dir / "diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Wrote {round_dir.name}/diagnosis.md")


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    for rd in sorted(run_dir.glob("round_*")):
        generate(rd)
