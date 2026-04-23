#!/usr/bin/env python3
"""Fetch Zulip messages using shared credentials resolved from common paths."""

from __future__ import annotations

import argparse
import base64
import configparser
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Credentials:
    site: str
    email: str
    api_key: str
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Explicit zuliprc path.")
    parser.add_argument(
        "--url",
        help="Zulip channel URL such as "
        "https://.../#channels/591576/VibeYoga-Hackathon-QEC/general",
    )
    parser.add_argument("--channel", help="Channel/stream name or numeric id.")
    parser.add_argument("--topic", help="Topic name.")
    parser.add_argument("--site", help="Override site URL from zuliprc.")
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of recent messages to request before the anchor. Default: 20.",
    )
    parser.add_argument(
        "--anchor",
        default="newest",
        help="Zulip anchor. Default: newest.",
    )
    parser.add_argument(
        "--contains",
        help="Case-insensitive substring filter applied after fetching.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print resolved credential source to stderr.",
    )
    return parser.parse_args()


def candidate_config_paths(explicit: Path | None) -> Iterable[Path]:
    if explicit is not None:
        yield explicit.expanduser()
        return

    raw = os.environ.get("ZULIPRC")
    if raw:
        yield Path(raw).expanduser()
    yield Path.home() / ".zuliprc"
    yield Path.home() / ".config" / "zulip" / "zuliprc"


def normalize_site(site: str) -> str:
    site = site.strip()
    if not site:
        raise SystemExit("Empty site value in zuliprc.")
    if "://" not in site:
        site = f"https://{site}"
    return site.rstrip("/")


def load_credentials(explicit: Path | None, site_override: str | None) -> Credentials:
    attempted: list[Path] = []
    for candidate in candidate_config_paths(explicit):
        attempted.append(candidate)
        if not candidate.is_file():
            continue
        parser = configparser.RawConfigParser()
        parser.read(candidate)
        if not parser.has_section("api"):
            continue
        site = site_override or parser.get("api", "site", fallback="").strip()
        email = parser.get("api", "email", fallback="").strip()
        api_key = parser.get("api", "key", fallback="").strip()
        if not site or not email or not api_key:
            continue
        return Credentials(
            site=normalize_site(site),
            email=email,
            api_key=api_key,
            source=candidate,
        )
    attempted_text = ", ".join(str(path) for path in attempted) or "<none>"
    raise SystemExit(
        "Unable to resolve Zulip credentials from: "
        f"{attempted_text}. "
        "Set ZULIPRC or create ~/.zuliprc or ~/.config/zulip/zuliprc."
    )


def parse_url_defaults(url: str) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    fragment = parsed.fragment.strip("/")
    if not fragment:
        return None, None
    parts = fragment.split("/")
    if len(parts) < 2 or parts[0] != "channels":
        return None, None
    channel = parts[1] or None
    topic = unquote("/".join(parts[3:])).strip() or None if len(parts) > 3 else None
    return channel, topic


def build_narrow(channel: str | None, topic: str | None) -> list[dict[str, object]]:
    narrow: list[dict[str, object]] = []
    if channel:
        operand: object = int(channel) if channel.isdigit() else channel
        narrow.append({"operator": "channel", "operand": operand})
    if topic:
        narrow.append({"operator": "topic", "operand": topic})
    return narrow


def request_json(url: str, credentials: Credentials, params: dict[str, object]) -> dict:
    query = urlencode(
        {
            key: json.dumps(value) if isinstance(value, (dict, list)) else value
            for key, value in params.items()
        }
    )
    auth = base64.b64encode(
        f"{credentials.email}:{credentials.api_key}".encode("utf-8")
    ).decode("ascii")
    request = Request(
        f"{url}?{query}",
        headers={
            "Authorization": f"Basic {auth}",
            "User-Agent": "autoqec-read-zulip/1.0",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Zulip API HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise SystemExit(f"Unable to reach Zulip API: {exc}") from exc


def fetch_messages(
    credentials: Credentials,
    channel: str | None,
    topic: str | None,
    limit: int,
    anchor: str,
) -> list[dict]:
    if limit < 1:
        raise SystemExit("--limit must be >= 1")
    data = request_json(
        f"{credentials.site}/api/v1/messages",
        credentials,
        {
            "anchor": anchor,
            "num_before": limit,
            "num_after": 0,
            "apply_markdown": "false",
            "client_gravatar": "false",
            "narrow": build_narrow(channel, topic),
        },
    )
    if data.get("result") != "success":
        raise SystemExit(f"Zulip API error: {data}")
    return list(data.get("messages", []))


def filter_messages(messages: Iterable[dict], needle: str | None) -> list[dict]:
    items = list(messages)
    if not needle:
        return items
    wanted = needle.casefold()
    filtered = []
    for message in items:
        haystack = " ".join(
            str(part)
            for part in (
                message.get("sender_full_name", ""),
                message.get("display_recipient", ""),
                message.get("subject", ""),
                message.get("content", ""),
            )
        ).casefold()
        if wanted in haystack:
            filtered.append(message)
    return filtered


def iso_timestamp(value: int | float | None) -> str:
    if value is None:
        return ""
    return (
        datetime.fromtimestamp(value, tz=timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )


def format_text(
    messages: Iterable[dict],
    credentials: Credentials,
    channel: str | None,
    topic: str | None,
) -> str:
    header = [
        f"# source_config: {credentials.source}",
        f"# site: {credentials.site}",
        f"# channel: {channel or '<all>'}",
        f"# topic: {topic or '<all>'}",
        "",
    ]
    blocks = []
    for message in messages:
        blocks.extend(
            [
                (
                    f"[{iso_timestamp(message.get('timestamp'))}] "
                    f"{message.get('sender_full_name', '<unknown>')} "
                    f"(id={message.get('id')}, topic={message.get('subject', '')})"
                ),
                str(message.get("content", "")).rstrip(),
                "",
            ]
        )
    return "\n".join(header + blocks).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    url_channel, url_topic = parse_url_defaults(args.url) if args.url else (None, None)
    channel = args.channel or url_channel
    topic = args.topic or url_topic

    credentials = load_credentials(args.config, args.site)
    if args.print_config:
        print(f"Using {credentials.source}", file=sys.stderr)

    messages = fetch_messages(
        credentials=credentials,
        channel=channel,
        topic=topic,
        limit=args.limit,
        anchor=args.anchor,
    )
    messages = filter_messages(messages, args.contains)

    payload = {
        "source_config": str(credentials.source),
        "site": credentials.site,
        "channel": channel,
        "topic": topic,
        "message_count": len(messages),
        "messages": messages,
    }
    if args.format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_text(messages, credentials, channel, topic), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
