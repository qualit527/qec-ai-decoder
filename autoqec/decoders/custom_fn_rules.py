"""Torch-free constants shared by `custom_fn_validator` and the
orchestration layer's Coder prompt payload.

Extracted so that `autoqec/orchestration/memory.py` can surface the
validator's rules to the Coder subagent without pulling torch through
the import chain. Keep these synced: if you change them here, the
validator picks up the new values automatically.
"""
from __future__ import annotations

ALLOWED_TOP_IMPORTS: frozenset[str] = frozenset({"torch", "typing"})
ALLOWED_FROM_IMPORTS: frozenset[str] = frozenset(
    {"torch", "torch.nn", "torch.nn.functional", "typing"}
)
FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        # process / filesystem / network escape modules
        "os",
        "subprocess",
        "sys",
        "shutil",
        "socket",
        "urllib",
        "pathlib",
        "io",
        "tempfile",
        "pickle",
        "marshal",
        "ctypes",
        "importlib",
        # builtin escape hatches — names must not appear bare even though
        # the validator also shadows __builtins__ at exec time.
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "breakpoint",
        "help",
        "__import__",
        "__builtins__",
        # reflective builtins that can reach out to __class__ / globals etc.
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "dir",
    }
)
SLOT_SIGNATURES: dict[str, list[str]] = {
    "message_fn": ["x_src", "x_dst", "e_ij", "params"],
    "aggregation": ["messages", "edge_index"],
    "head": ["hidden_state"],
}
