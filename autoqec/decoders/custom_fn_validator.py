from __future__ import annotations

import ast
from types import FunctionType

import torch
from torch import nn

ALLOWED_TOP_IMPORTS = {"torch", "typing"}
ALLOWED_FROM_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "typing"}
FORBIDDEN_NAMES = {"os", "subprocess", "sys", "shutil", "socket", "urllib", "eval", "exec", "open"}
SLOT_SIGNATURES = {
    "message_fn": ["x_src", "x_dst", "e_ij", "params"],
    "aggregation": ["messages", "edge_index"],
    "head": ["hidden_state"],
}


def _load_function(code: str) -> FunctionType:
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(code, "<custom_fn>", "exec"), namespace, namespace)
    funcs = [value for value in namespace.values() if isinstance(value, FunctionType)]
    if len(funcs) != 1:
        raise ValueError("Must define exactly one function")
    return funcs[0]


def _smoke_test(fn: FunctionType, slot: str) -> tuple[bool, str]:
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

    try:
        fn = _load_function(code)
    except Exception as exc:
        return False, f"Compilation failed: {exc}"
    return _smoke_test(fn, slot)

