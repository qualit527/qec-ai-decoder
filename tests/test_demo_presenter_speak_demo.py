from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(".claude/skills/demo-presenter/scripts/speak_demo.py")
LIVE_SCRIPT_PATH = Path(".claude/skills/demo-presenter/scripts/live_present_demo.py")


def load_script():
    spec = importlib.util.spec_from_file_location("speak_demo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_live_script():
    spec = importlib.util.spec_from_file_location("live_present_demo", LIVE_SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_chunk_text_splits_long_script_without_losing_words():
    module = load_script()
    text = "Intro sentence. " + ("AutoQEC proves auditable demo evidence. " * 20)

    chunks = module.chunk_text(text, max_chars=120)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert " ".join(chunks).replace("  ", " ") == text.strip()


def test_build_parser_defaults_to_presentation_voice_and_mp3():
    module = load_script()

    args = module.build_parser().parse_args(["demo_script.txt"])

    assert args.input == Path("demo_script.txt")
    assert args.output == Path("demo_script.mp3")
    assert args.model == "gpt-4o-mini-tts"
    assert args.voice == "coral"
    assert "technical presentation" in args.instructions


def test_resolve_base_url_prefers_cli_over_environment(monkeypatch):
    module = load_script()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example/v1")

    assert module.resolve_base_url("https://cli.example/v1") == "https://cli.example/v1"
    assert module.resolve_base_url(None) == "https://env.example/v1"


def test_live_plan_marks_worktree_demo_pr_only_by_default():
    module = load_live_script()

    steps = module.build_demo_steps(include_pr_worktree=False)

    assert [step.demo_id for step in steps] == ["1", "2", "3", "4", "5"]
    worktree_step = next(step for step in steps if step.demo_id == "3")
    assert worktree_step.command is None
    assert "PR-only" in worktree_step.status


def test_live_dry_run_writes_manifest_without_running_commands(tmp_path, monkeypatch):
    module = load_live_script()
    step = module.DemoStep(
        demo_id="x",
        name="Synthetic demo",
        status="test",
        command=["definitely-not-run"],
        pre_narration="before",
        post_success="success",
        post_failure="failure",
        artifacts=["artifact.json"],
    )
    monkeypatch.setattr(
        module,
        "run_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("command ran")),
    )

    report = module.execute_demo_steps(
        steps=[step],
        output_dir=tmp_path,
        dry_run=True,
        audio=False,
        playback=False,
        base_url=None,
        model="gpt-4o-mini-tts",
        voice="coral",
        instructions="test voice",
    )

    assert report["steps"][0]["result"] == "dry-run"
    assert (tmp_path / "manifest.json").exists()


def _install_overlap_fakes(module, monkeypatch, events: list[str]):
    def fake_synthesize(*, text, output_path, audio, **_kwargs):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if audio:
            output_path.write_bytes(b"fake-audio")
            return output_path
        txt = output_path.with_suffix(".txt")
        txt.write_text(text)
        return txt

    class FakeAudioProc:
        def __init__(self, label: str) -> None:
            self.label = label

        def wait(self, timeout: float | None = None) -> int:
            events.append(f"wait:{self.label}")
            return 0

    def fake_start_playback(path):
        events.append(f"start:{path.name}")
        return FakeAudioProc(path.name)

    class FakeCompleted:
        returncode = 0
        stdout = "ok"

    def fake_run_command(command, log_path):
        events.append(f"run:{command[-1]}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok")
        return FakeCompleted()

    def fake_play_audio(path):
        events.append(f"play:{path.name}")

    monkeypatch.setattr(module, "synthesize_narration", fake_synthesize)
    monkeypatch.setattr(module, "start_playback", fake_start_playback)
    monkeypatch.setattr(module, "run_command", fake_run_command)
    monkeypatch.setattr(module, "play_audio", fake_play_audio)


def test_live_overlaps_pre_narration_with_demo_command(tmp_path, monkeypatch):
    module = load_live_script()
    events: list[str] = []
    _install_overlap_fakes(module, monkeypatch, events)

    step = module.DemoStep(
        demo_id="x",
        name="overlap test",
        status="test",
        command=["echo", "overlap-marker"],
        pre_narration="pre",
        post_success="ok",
        post_failure="nope",
        artifacts=[],
    )

    module.execute_demo_steps(
        steps=[step],
        output_dir=tmp_path,
        dry_run=False,
        audio=True,
        playback=True,
        base_url=None,
        model="m",
        voice="v",
        instructions="i",
    )

    start_idx = events.index("start:demo-x-before.mp3")
    run_idx = events.index("run:overlap-marker")
    wait_idx = events.index("wait:demo-x-before.mp3")
    assert start_idx < run_idx < wait_idx, events
    assert events[wait_idx + 1] == "play:demo-x-after.mp3", events


def test_live_dry_run_plays_pre_narration_without_overlap(tmp_path, monkeypatch):
    module = load_live_script()
    events: list[str] = []
    _install_overlap_fakes(module, monkeypatch, events)

    def fail_run_command(*_args, **_kwargs):
        raise AssertionError("dry-run must not execute commands")

    monkeypatch.setattr(module, "run_command", fail_run_command)

    step = module.DemoStep(
        demo_id="y",
        name="dry overlap",
        status="test",
        command=["should-not-run"],
        pre_narration="pre",
        post_success="ok",
        post_failure="nope",
        artifacts=[],
    )

    module.execute_demo_steps(
        steps=[step],
        output_dir=tmp_path,
        dry_run=True,
        audio=True,
        playback=True,
        base_url=None,
        model="m",
        voice="v",
        instructions="i",
    )

    assert "start:demo-y-before.mp3" in events
    assert "wait:demo-y-before.mp3" in events
    assert not any(evt.startswith("run:") for evt in events)


def test_live_start_playback_returns_none_when_no_async_player(tmp_path, monkeypatch):
    module = load_live_script()
    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake")

    assert module.start_playback(audio_file) is None


def test_live_start_playback_ignores_text_file(tmp_path):
    module = load_live_script()
    text_file = tmp_path / "narration.txt"
    text_file.write_text("hello")

    assert module.start_playback(text_file) is None
