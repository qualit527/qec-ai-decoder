from pathlib import Path


def test_cold_install_benchmark_doc_records_reproducible_gate() -> None:
    text = Path("docs/verification/cold-install-benchmark-2026-04-24.md").read_text(
        encoding="utf-8"
    )

    assert "pip install -e '.[dev]'" in text
    assert "--no-cache-dir" in text
    assert "--target" in text
    assert "180 s" in text
    assert "Linux" in text
    assert "Windows" in text
