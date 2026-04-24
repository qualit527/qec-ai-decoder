from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    gated_markers = (
        ("integration", "--run-integration", "need --run-integration option to run"),
        ("slow", "--run-slow", "need --run-slow option to run"),
    )
    for marker_name, option_name, reason in gated_markers:
        if config.getoption(option_name):
            continue
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if marker_name in item.keywords:
                item.add_marker(skip_marker)
