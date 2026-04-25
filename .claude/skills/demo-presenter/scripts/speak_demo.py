from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "coral"
DEFAULT_INSTRUCTIONS = (
    "Speak in a warm, confident, technical presentation style. "
    "Keep the pacing clear for a live research demo."
)
DEFAULT_MAX_CHARS = 3500
SUPPORTED_FORMATS = {"mp3", "wav", "opus", "aac", "flac"}


class DemoSpeechParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args=args, namespace=namespace)
        if parsed.output is None:
            parsed.output = parsed.input.with_suffix(".mp3")
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = DemoSpeechParser(
        description="Turn an AutoQEC demo narration script into spoken audio."
    )
    parser.add_argument("input", type=Path, help="Plain-text narration script.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output audio path. Defaults to <input>.mp3.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per TTS request.",
    )
    return parser


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("input text is empty")
    if max_chars < 1:
        raise ValueError("--max-chars must be positive")

    words = normalized.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        if len(word) > max_chars:
            raise ValueError(f"single word exceeds max_chars: {word[:40]!r}")
        added_len = len(word) if not current else len(word) + 1
        if current and current_len + added_len > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def response_format_for(output: Path) -> str:
    suffix = output.suffix.lower().lstrip(".")
    return suffix if suffix in SUPPORTED_FORMATS else "mp3"


def get_env_var(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            registry_value, _ = winreg.QueryValueEx(key, name)
            return registry_value or None
    except OSError:
        return None


def resolve_base_url(cli_base_url: str | None) -> str | None:
    return cli_base_url or get_env_var("OPENAI_BASE_URL")


def create_client(base_url: str | None = None):
    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Python package 'openai' is not installed") from exc
    resolved_base_url = resolve_base_url(base_url)
    if resolved_base_url:
        return OpenAI(api_key=api_key, base_url=resolved_base_url)
    return OpenAI(api_key=api_key)


def synthesize_chunk(
    *,
    client,
    text: str,
    output: Path,
    model: str,
    voice: str,
    instructions: str,
    response_format: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format=response_format,
    ) as response:
        response.stream_to_file(output)


def concat_with_ffmpeg(part_paths: list[Path], output: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        list_path = Path(tmp_dir) / "audio_parts.txt"
        lines = [f"file '{path.as_posix()}'" for path in part_paths]
        list_path.write_text("\n".join(lines), encoding="utf-8")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output),
            ],
            check=True,
        )
    return True


def write_playlist(part_paths: list[Path], output: Path) -> Path:
    playlist = output.with_suffix(".m3u")
    playlist.write_text(
        "\n".join(path.name for path in part_paths) + "\n",
        encoding="utf-8",
    )
    return playlist


def synthesize_script(args: argparse.Namespace) -> list[Path]:
    text = args.input.read_text(encoding="utf-8")
    chunks = chunk_text(text, max_chars=args.max_chars)
    response_format = response_format_for(args.output)
    client = create_client(base_url=args.base_url)

    if len(chunks) == 1:
        synthesize_chunk(
            client=client,
            text=chunks[0],
            output=args.output,
            model=args.model,
            voice=args.voice,
            instructions=args.instructions,
            response_format=response_format,
        )
        return [args.output]

    part_paths = [
        args.output.with_name(
            f"{args.output.stem}.part{idx:02d}{args.output.suffix or '.mp3'}"
        )
        for idx in range(1, len(chunks) + 1)
    ]
    for chunk, part_path in zip(chunks, part_paths, strict=True):
        synthesize_chunk(
            client=client,
            text=chunk,
            output=part_path,
            model=args.model,
            voice=args.voice,
            instructions=args.instructions,
            response_format=response_format,
        )

    if concat_with_ffmpeg(part_paths, args.output):
        return [args.output, *part_paths]
    return [write_playlist(part_paths, args.output), *part_paths]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        outputs = synthesize_script(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print("AI-generated voice disclosure: tell listeners this narration is AI-generated.")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
