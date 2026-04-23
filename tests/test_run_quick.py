"""Security / fail-fast tests for scripts/run_quick.py (Sourcery S603 close-out).

We assert that `_validated_env_yaml`:
  (a) rejects a non-existent path before it ever reaches subprocess, and
  (b) returns an absolute resolved path for a real file.

The subprocess call itself is in list-form without shell=True, so shell
injection is structurally impossible even if validation were skipped.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_validated_env_yaml_rejects_missing_path(tmp_path: Path) -> None:
    from scripts.run_quick import _validated_env_yaml

    bogus = tmp_path / "does-not-exist.yaml"
    with pytest.raises(SystemExit, match="env_yaml not found"):
        _validated_env_yaml(str(bogus))


def test_validated_env_yaml_rejects_directory(tmp_path: Path) -> None:
    from scripts.run_quick import _validated_env_yaml

    d = tmp_path / "a_directory"
    d.mkdir()
    with pytest.raises(SystemExit, match="env_yaml not found"):
        _validated_env_yaml(str(d))


def test_validated_env_yaml_accepts_real_file_and_returns_absolute(tmp_path: Path) -> None:
    from scripts.run_quick import _validated_env_yaml

    f = tmp_path / "env.yaml"
    f.write_text("name: x\n", encoding="utf-8")
    out = _validated_env_yaml(str(f))
    assert Path(out).is_absolute()
    assert Path(out) == f.resolve()


def test_validated_env_yaml_rejects_shell_metacharacter_path(tmp_path: Path) -> None:
    """Classic "hostile filename" probe — even if shell metacharacters make
    it into the argv list, subprocess uses shell=False. Validation still
    rejects them because the filesystem lookup fails."""
    from scripts.run_quick import _validated_env_yaml

    # On both Windows and POSIX, a string containing ; or | will resolve to
    # a literal filesystem path that does not exist in the user's tmpdir.
    hostile = tmp_path / "does-not-exist.yaml; rm -rf /"
    with pytest.raises(SystemExit, match="env_yaml not found"):
        _validated_env_yaml(str(hostile))
