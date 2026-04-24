from __future__ import annotations

import json
from pathlib import Path


FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"


def fixture_path(*parts: str) -> Path:
    return FIXTURES_ROOT.joinpath(*parts)


def load_json_fixture(*parts: str):
    return json.loads(fixture_path(*parts).read_text(encoding="utf-8"))
