from __future__ import annotations

from tests.fixture_utils import fixture_path, load_json_fixture


def test_public_api_fixture_manifest_references_existing_files() -> None:
    manifest = load_json_fixture("public_api", "manifest.json")

    for entry in manifest["fixtures"]:
        assert entry["stability"] in {"contract", "smoke"}
        assert fixture_path("public_api", entry["path"]).exists()
