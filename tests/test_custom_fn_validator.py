import importlib.util

import pytest

from autoqec.decoders.custom_fn_validator import validate_custom_fn

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="smoke test needs torch")
def test_valid_custom_message_fn() -> None:
    code = """
def message(x_src, x_dst, e_ij, params):
    import torch
    import torch.nn.functional as F
    return F.relu(params["W"](torch.cat([x_src, x_dst], dim=-1)))
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert ok, reason


def test_rejects_os_import() -> None:
    code = """
def message(x_src, x_dst, e_ij, params):
    import os
    os.system("rm -rf /")
    return x_src
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "import" in reason.lower() or "forbidden" in reason.lower()


# ─── Red-team: sandbox-escape patterns the old whitelist let through ────────


def test_rejects_dunder_import_name() -> None:
    """Bare ``__import__('os')`` must be refused even without an import stmt."""
    code = """
def message(x_src, x_dst, e_ij, params):
    __import__("os").system("echo pwned")
    return x_src
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok, "AST walk must flag __import__ as a forbidden name"
    assert "__import__" in reason or "forbidden" in reason.lower()


def test_rejects_getattr_reflection() -> None:
    """``getattr`` is a reflection escape — forbid it."""
    code = """
def message(x_src, x_dst, e_ij, params):
    return getattr(params, "W")(x_src)
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "getattr" in reason.lower() or "forbidden" in reason.lower()


def test_rejects_dunder_attribute_access() -> None:
    """``().__class__.__mro__[1].__subclasses__()`` — classic sandbox escape."""
    code = """
def message(x_src, x_dst, e_ij, params):
    cls = ().__class__
    return cls
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok, "dunder attribute access must be rejected"
    assert "__class__" in reason or "attribute" in reason.lower()


def test_rejects_module_level_code_execution() -> None:
    """Module-level side effects inside custom_fn source must still get scanned.

    Before the hardening, ``_load_function`` exec'd the whole source with the
    real ``__builtins__`` in scope, so a bare ``__import__('os').system(...)``
    at module level would run the moment validate_custom_fn was called.
    """
    code = """
__import__('os')
def message(x_src, x_dst, e_ij, params):
    return x_src
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok, f"module-level __import__ leaked through validator: {reason}"


def test_rejects_open_via_builtins_lookup() -> None:
    """Even if AST check missed a call, exec-time __builtins__ restriction must fire."""
    code = """
def message(x_src, x_dst, e_ij, params):
    with open("/tmp/leak", "w") as f:
        f.write("pwned")
    return x_src
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "open" in reason.lower() or "forbidden" in reason.lower()


def test_rejects_builtins_reference() -> None:
    """``__builtins__`` itself must not be reachable by name."""
    code = """
def message(x_src, x_dst, e_ij, params):
    return __builtins__
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "__builtins__" in reason or "forbidden" in reason.lower()


def test_rejects_importlib_import() -> None:
    code = """
def message(x_src, x_dst, e_ij, params):
    import importlib
    mod = importlib.import_module("os")
    return x_src
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "importlib" in reason.lower() or "forbidden" in reason.lower()


def test_rejects_private_underscore_attribute() -> None:
    """Leading-underscore attrs (``x._private``) are off-limits too."""
    code = """
def message(x_src, x_dst, e_ij, params):
    return params._private
"""
    ok, reason = validate_custom_fn(code, slot="message_fn")
    assert not ok
    assert "_private" in reason or "attribute" in reason.lower()

