from autoqec.decoders.custom_fn_validator import validate_custom_fn


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

