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
    {"os", "subprocess", "sys", "shutil", "socket", "urllib", "eval", "exec", "open"}
)
SLOT_SIGNATURES: dict[str, list[str]] = {
    "message_fn": ["x_src", "x_dst", "e_ij", "params"],
    "aggregation": ["messages", "edge_index"],
    "head": ["hidden_state"],
}
