"""AST + smoke-test validator for Tier-2 ``custom_fn`` predecoder blocks.

Two-stage gate:

1. Parse the source with ``ast`` and enforce structural rules:
   - exactly one function definition at module level
   - signature matches the slot's expected argument names
   - imports restricted to ``ALLOWED_TOP_IMPORTS`` / ``ALLOWED_FROM_IMPORTS``
   - no ``FORBIDDEN_NAMES`` references (``os``, ``eval``, builtins-level
     escape hatches such as ``__import__``, ``getattr``, ``__builtins__``, ...)
   - no attribute access to dunder names (blocks ``x.__class__.__mro__``)
     or leading-underscore private slots
2. ``exec`` the source with a *restricted* ``__builtins__`` so even if the
   AST check missed something, calls like ``open(...)`` / ``__import__(...)``
   are unresolvable at runtime.

Torch-heavy imports are loaded lazily so the validator module stays cheap
to import from tests / orchestration layers that just need the rule set.
"""
from __future__ import annotations

import ast
from types import FunctionType, MappingProxyType
from typing import Any, Mapping

from autoqec.decoders.custom_fn_rules import (
    ALLOWED_FROM_IMPORTS,
    ALLOWED_TOP_IMPORTS,
    FORBIDDEN_NAMES,
    SLOT_SIGNATURES,
)

__all__ = [
    "ALLOWED_TOP_IMPORTS",
    "ALLOWED_FROM_IMPORTS",
    "FORBIDDEN_NAMES",
    "SLOT_SIGNATURES",
    "SAFE_BUILTINS",
    "validate_custom_fn",
]


# Minimal builtins we expose to Tier-2 user code. Anything not listed here
# resolves to ``NameError`` at call time — a last-mile defense if a static
# check misses a new escape pattern. Keep this set as small as we can
# justify; the Coder subagent's prompt enumerates the public surface it can
# rely on.
_SAFE_BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        # construction / iteration
        "bool",
        "int",
        "float",
        "complex",
        "str",
        "bytes",
        "tuple",
        "list",
        "dict",
        "set",
        "frozenset",
        "range",
        "enumerate",
        "zip",
        "iter",
        "next",
        "reversed",
        "sorted",
        "filter",
        "map",
        # type checks
        "isinstance",
        "issubclass",
        # math / stats
        "abs",
        "min",
        "max",
        "sum",
        "round",
        "pow",
        "len",
        "any",
        "all",
        "divmod",
        # misc pure helpers
        "slice",
        "id",
        # exceptions the user might catch inside their function
        "Exception",
        "ValueError",
        "TypeError",
        "RuntimeError",
        "ZeroDivisionError",
        "IndexError",
        "KeyError",
        "AttributeError",
        "NotImplementedError",
    }
)


def _build_safe_builtins() -> Mapping[str, Any]:
    import builtins as _builtins

    allowed = {
        name: getattr(_builtins, name)
        for name in _SAFE_BUILTIN_NAMES
        if hasattr(_builtins, name)
    }
    return MappingProxyType(allowed)


SAFE_BUILTINS: Mapping[str, Any] = _build_safe_builtins()


def _load_function(code: str) -> FunctionType:
    # Lazy import torch so the validator module itself stays torch-free;
    # smoke tests still run the function against real tensors below.
    import torch

    # Restricted exec namespace. We deliberately shadow ``__builtins__`` with
    # our allow-list instead of the real module — a Tier-2 function that
    # tries ``__import__('os')`` or ``open('/etc/passwd')`` now raises
    # NameError at evaluation time, not just a lint rejection.
    namespace: dict[str, object] = {
        "__builtins__": dict(SAFE_BUILTINS),
        "torch": torch,
    }
    exec(compile(code, "<custom_fn>", "exec"), namespace, namespace)  # noqa: S102 - sandboxed
    funcs = [value for value in namespace.values() if isinstance(value, FunctionType)]
    if len(funcs) != 1:
        raise ValueError("Must define exactly one function")
    return funcs[0]


def _smoke_test(fn: FunctionType, slot: str) -> tuple[bool, str]:
    import torch
    from torch import nn

    try:
        if slot == "message_fn":
            params = {
                "W": nn.Linear(16, 8),
                "W_gate": nn.Linear(4, 1),
                "W_src": nn.Linear(8, 8),
                "W_dst": nn.Linear(8, 8),
            }
            out = fn(torch.randn(6, 8), torch.randn(6, 8), torch.randn(6, 4), params)
            if not isinstance(out, torch.Tensor) or out.shape[0] != 6:
                return False, "message_fn smoke test produced an invalid tensor"
        elif slot == "aggregation":
            out = fn(torch.randn(10, 8), torch.randint(0, 5, (2, 10)))
            if not isinstance(out, torch.Tensor):
                return False, "aggregation smoke test produced a non-tensor"
        elif slot == "head":
            out = fn(torch.randn(4, 12, 8))
            if not isinstance(out, torch.Tensor):
                return False, "head smoke test produced a non-tensor"
    except Exception as exc:  # pragma: no cover - exact errors are input-dependent
        return False, f"Smoke test failed: {exc}"
    if not torch.isfinite(out).all():
        return False, "Smoke test produced NaN/Inf output"
    return True, "ok"


def _static_ast_checks(tree: ast.AST) -> tuple[bool, str]:
    """Pure-AST rejection pass. No code execution, no torch required."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_NAMES:
                    return False, f"Forbidden import: {alias.name}"
                if root not in ALLOWED_TOP_IMPORTS:
                    return False, f"Import not in whitelist: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in FORBIDDEN_NAMES:
                return False, f"Forbidden from-import: {module}"
            if module not in ALLOWED_FROM_IMPORTS:
                return False, f"From-import not in whitelist: {module}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"Forbidden name reference: {node.id}"
        elif isinstance(node, ast.Attribute):
            # Block dunder / private attribute access — closes the classic
            # sandbox escape via ``().__class__.__mro__[1].__subclasses__()``
            # and ``x.__globals__`` / ``fn.__code__`` style poking.
            if node.attr.startswith("_"):
                return False, f"Forbidden attribute access: .{node.attr}"
    return True, "ok"


def validate_custom_fn(code: str, slot: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1:
        return False, "Must define exactly one function"

    expected = SLOT_SIGNATURES.get(slot)
    if expected is None:
        return False, f"Unknown slot: {slot}"
    actual = [arg.arg for arg in functions[0].args.args]
    if actual != expected:
        return False, f"Signature must be {expected}, got {actual}"

    ok, reason = _static_ast_checks(tree)
    if not ok:
        return False, reason

    try:
        fn = _load_function(code)
    except Exception as exc:
        return False, f"Compilation failed: {exc}"
    return _smoke_test(fn, slot)
