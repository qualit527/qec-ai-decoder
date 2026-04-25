from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
SPEAK_DEMO_PATH = Path(__file__).with_name("speak_demo.py")
DEFAULT_RUN_ROOT = Path("runs/demo-presenter-live")
DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "coral"
DEFAULT_INSTRUCTIONS = (
    "Speak like a warm, precise research demo host. Keep each segment concise."
)
# Non-blocking audio players, in preference order. afplay is macOS-only.
ASYNC_PLAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ffplay", ("-nodisp", "-autoexit", "-loglevel", "quiet")),
    ("afplay", ()),
    ("mpv", ("--no-video", "--really-quiet")),
    ("mpg123", ("-q",)),
    ("paplay", ()),
    ("aplay", ("-q",)),
)


@dataclass(frozen=True)
class DemoStep:
    demo_id: str
    name: str
    status: str
    command: list[str] | None
    pre_narration: str
    post_success: str
    post_failure: str
    artifacts: list[str]


def load_speak_demo():
    spec = importlib.util.spec_from_file_location("speak_demo", SPEAK_DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SPEAK_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_demo_steps(include_pr_worktree: bool = False) -> list[DemoStep]:
    worktree_command = None
    if include_pr_worktree and (REPO_ROOT / "demos/demo-3-worktree-provenance/run.sh").exists():
        worktree_command = ["bash", "demos/demo-3-worktree-provenance/run.sh"]

    return [
        DemoStep(
            demo_id="1",
            name="surface_d5 end-to-end loop",
            status="merged on main",
            command=["bash", "demos/demo-1-surface-d5/run_quick.sh"],
            pre_narration=(
                "Demo 1 starts with the surface-code path. The important point is that "
                "AutoQEC is not just printing a config; it will compile a decoder DSL, "
                "train it, evaluate it, and write auditable artifacts."
            ),
            post_success=(
                "Demo 1 finished. The correctness evidence is structural: the run wrote "
                "history, Pareto candidate data, a trained checkpoint, logs, and metrics "
                "for the surface-code backend."
            ),
            post_failure=(
                "Demo 1 did not complete in this environment. Keep the failure visible; "
                "the surface-code end-to-end claim is not freshly supported by this run."
            ),
            artifacts=[
                "runs/<latest>/history.jsonl",
                "runs/<latest>/candidate_pareto.json",
                "runs/<latest>/round_1/metrics.json",
            ],
        ),
        DemoStep(
            demo_id="2",
            name="BB72 qLDPC backend switch",
            status="merged on main",
            command=["bash", "demos/demo-2-bb72/run.sh"],
            pre_narration=(
                "Demo 2 switches code families. We keep the same harness but move from "
                "surface-code MWPM to BB72 qLDPC with OSD through environment config."
            ),
            post_success=(
                "Demo 2 finished. The key evidence is that the qLDPC run produces the "
                "same artifact shape while using the OSD classical backend."
            ),
            post_failure=(
                "Demo 2 failed here. The generic-backend claim needs triage before using "
                "this as fresh live evidence."
            ),
            artifacts=["runs/<latest>/round_1/metrics.json", "autoqec/envs/builtin/bb72_depol.yaml"],
        ),
        DemoStep(
            demo_id="3",
            name="worktree branches-as-Pareto provenance",
            status="planned / PR-only until merged",
            command=worktree_command,
            pre_narration=(
                "Demo 3 is the provenance story. It is currently PR-only unless the demo "
                "directory exists on main. The claim is that research candidates become "
                "git branches with pointer JSON, compose merges, and recorded conflicts."
            ),
            post_success=(
                "The worktree provenance demo finished. Branch names, commit SHAs, "
                "pointer JSON, and merge graph output make candidate history auditable."
            ),
            post_failure=(
                "The worktree demo was not run successfully from main. Present it as "
                "planned or PR-only evidence, not as a merged live demo."
            ),
            artifacts=[
                "origin/feat/issue-38-worktree-demo:demos/demo-3-worktree-provenance/README.md",
                "origin/feat/issue-38-worktree-demo:demos/demo-3-worktree-provenance/expected_output/run_demo.stdout.txt",
            ],
        ),
        DemoStep(
            demo_id="4",
            name="reward-hacking rejection (narrated)",
            status="merged on main",
            command=["bash", "demos/demo-4-reward-hacking/present.sh"],
            pre_narration=(
                "Demo 4 is the trust test, and you will see it unfold in five labeled "
                "phases. Phase 1 builds a memorizing cheater from training-seed "
                "syndromes. Phase 2 shows the table hit rate on memorized shots, fresh "
                "train shots, and holdout shots; that comparison reveals the cheat is "
                "bound to specific shots, not to seed ranges. Phase 3 runs the "
                "independent verifier on holdout. Phase 4 checks the three guards -- "
                "seed leakage, paired bootstrap CI, ablation sanity. Phase 5 reads the "
                "verdict and states the Pareto consequence."
            ),
            post_success=(
                "Demo 4 finished. The memorizer was rejected, the scoreboard PNG was "
                "written under visualizations, and present_summary.json records the "
                "phase-by-phase numbers. A memorizer should be SUSPICIOUS or FAILED, "
                "never VERIFIED."
            ),
            post_failure=(
                "Demo 4 failed here. Do not claim the verifier rejected the cheater in "
                "this live session until the report exists and says so."
            ),
            artifacts=[
                "runs/demo-4/round_0/verification_report.json",
                "runs/demo-4/round_0/present_summary.json",
                "runs/demo-4/round_0/visualizations/scoreboard.png",
            ],
        ),
        DemoStep(
            demo_id="5",
            name="failure root-cause diagnosis",
            status="merged on main",
            command=["bash", "demos/demo-5-failure-recovery/run.sh"],
            pre_narration=(
                "Demo 5 shows operational recovery. A research harness needs useful "
                "failure states, not just green-path metrics."
            ),
            post_success=(
                "Demo 5 finished. The machine-readable compile_error and reason field "
                "are what make diagnosis automatable."
            ),
            post_failure=(
                "Demo 5 failed here, so the diagnostic claim is not freshly demonstrated "
                "by this run."
            ),
            artifacts=["runs/demo-5/", "stdout JSON status_reason"],
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AutoQEC demos while generating and optionally playing narration."
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not run demo commands.")
    parser.add_argument("--no-audio", action="store_true", help="Write narration text only.")
    parser.add_argument("--no-playback", action="store_true", help="Generate audio but do not play it.")
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Play pre-narration to completion before starting each demo command.",
    )
    parser.add_argument(
        "--include-pr-worktree",
        action="store_true",
        help="Run Demo 3 only if its PR-only directory exists locally.",
    )
    parser.add_argument(
        "--skip-demo",
        action="append",
        default=[],
        help="Demo id to skip. Can be repeated, for example --skip-demo 4.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    return parser


def timestamped_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RUN_ROOT / stamp


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(proc.stdout)
    return proc


def _resolve_async_player(path: Path) -> list[str] | None:
    system = platform.system()
    for name, args in ASYNC_PLAYERS:
        if name == "afplay" and system != "Darwin":
            continue
        exe = shutil.which(name)
        if exe:
            return [exe, *args, str(path)]
    return None


def start_playback(path: Path) -> subprocess.Popen | None:
    """Start non-blocking audio playback. Returns Popen or None when no suitable
    async player is available. Callers must wait() to avoid leaving the child
    running past the step."""
    if not path.exists() or path.suffix.lower() == ".txt":
        return None
    cmd = _resolve_async_player(path)
    if cmd is None:
        return None
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None


def play_audio(path: Path) -> None:
    """Blocking audio playback. Prefers the non-blocking player family so there
    is no GUI popup; falls back to the platform default if none is installed."""
    popen = start_playback(path)
    if popen is not None:
        popen.wait()
        return
    system = platform.system()
    if system == "Windows":
        subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"Start-Process -FilePath '{path}' -Wait",
            ],
            check=False,
        )
        return
    if system == "Darwin":
        subprocess.run(["afplay", str(path)], check=False)
        return
    subprocess.run(["xdg-open", str(path)], check=False)


def synthesize_narration(
    *,
    text: str,
    output_path: Path,
    audio: bool,
    base_url: str | None,
    model: str,
    voice: str,
    instructions: str,
) -> Path:
    text_path = output_path.with_suffix(".txt")
    write_text(text_path, text)
    if not audio:
        return text_path

    speak_demo = load_speak_demo()
    args = speak_demo.build_parser().parse_args(
        [
            str(text_path),
            "--output",
            str(output_path),
            "--model",
            model,
            "--voice",
            voice,
            "--instructions",
            instructions,
            *(["--base-url", base_url] if base_url else []),
        ]
    )
    speak_demo.synthesize_script(args)
    return output_path


def narrate(
    *,
    text: str,
    output_path: Path,
    audio: bool,
    playback: bool,
    base_url: str | None,
    model: str,
    voice: str,
    instructions: str,
) -> Path:
    path = synthesize_narration(
        text=text,
        output_path=output_path,
        audio=audio,
        base_url=base_url,
        model=model,
        voice=voice,
        instructions=instructions,
    )
    if audio and playback:
        play_audio(path)
    return path


def _run_step_with_overlap(
    *,
    step: DemoStep,
    pre_audio_path: Path,
    log_path: Path,
    dry_run: bool,
    audio: bool,
    playback: bool,
    overlap: bool,
) -> tuple[str, int | None]:
    """Start the pre-narration playback and the demo command concurrently, then
    wait for both. If no async player is available, fall back to sequential
    pre-narration so the listener still hears the context."""
    want_audio = audio and playback and pre_audio_path.suffix.lower() != ".txt"
    audio_proc: subprocess.Popen | None = None

    if want_audio:
        if overlap:
            audio_proc = start_playback(pre_audio_path)
            if audio_proc is None:
                # No async player — preserve narration by playing blocking.
                play_audio(pre_audio_path)
        else:
            play_audio(pre_audio_path)

    try:
        if dry_run:
            write_text(log_path, "dry-run: command not executed")
            return "dry-run", None
        if step.command is None:
            write_text(log_path, "skipped: demo is not runnable from this checkout")
            return "skipped", None
        proc = run_command(step.command, log_path)
        result = "passed" if proc.returncode == 0 else "failed"
        return result, proc.returncode
    finally:
        if audio_proc is not None:
            audio_proc.wait()


def execute_demo_steps(
    *,
    steps: list[DemoStep],
    output_dir: Path,
    dry_run: bool,
    audio: bool,
    playback: bool,
    base_url: str | None,
    model: str,
    voice: str,
    instructions: str,
    overlap: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "audio": audio,
        "playback": playback,
        "overlap": overlap,
        "steps": [],
    }

    opening = (
        "This is an AI-generated narration. I will run the AutoQEC demos in order, "
        "explain why each one matters, and keep failures visible."
    )
    narrate(
        text=opening,
        output_path=output_dir / "00-opening.mp3",
        audio=audio,
        playback=playback,
        base_url=base_url,
        model=model,
        voice=voice,
        instructions=instructions,
    )

    for step in steps:
        prefix = f"demo-{step.demo_id}"
        pre_audio_path = synthesize_narration(
            text=step.pre_narration,
            output_path=output_dir / f"{prefix}-before.mp3",
            audio=audio,
            base_url=base_url,
            model=model,
            voice=voice,
            instructions=instructions,
        )

        log_path = output_dir / f"{prefix}.log"
        result, returncode = _run_step_with_overlap(
            step=step,
            pre_audio_path=pre_audio_path,
            log_path=log_path,
            dry_run=dry_run,
            audio=audio,
            playback=playback,
            overlap=overlap,
        )

        post_text = step.post_success if result == "passed" else step.post_failure
        narrate(
            text=post_text,
            output_path=output_dir / f"{prefix}-after.mp3",
            audio=audio,
            playback=playback,
            base_url=base_url,
            model=model,
            voice=voice,
            instructions=instructions,
        )

        report["steps"].append(
            {
                **asdict(step),
                "result": result,
                "returncode": returncode,
                "log_path": str(log_path),
            }
        )

    closing = (
        "The live walkthrough is complete. Use the manifest and logs as the audit "
        "trail for which claims were demonstrated live and which were only planned "
        "or skipped."
    )
    narrate(
        text=closing,
        output_path=output_dir / "99-closing.mp3",
        audio=audio,
        playback=playback,
        base_url=base_url,
        model=model,
        voice=voice,
        instructions=instructions,
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = args.output_dir or timestamped_output_dir()
    skip = set(args.skip_demo)
    steps = [
        step
        for step in build_demo_steps(include_pr_worktree=args.include_pr_worktree)
        if step.demo_id not in skip
    ]
    report = execute_demo_steps(
        steps=steps,
        output_dir=output_dir,
        dry_run=args.dry_run,
        audio=not args.no_audio,
        playback=not args.no_playback,
        base_url=args.base_url,
        model=args.model,
        voice=args.voice,
        instructions=args.instructions,
        overlap=not args.no_overlap,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
