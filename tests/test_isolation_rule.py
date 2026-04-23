"""independent_eval.py must not import autoqec.runner.*"""
from pathlib import Path
import re


def test_independent_eval_does_not_import_runner():
    src_path = Path("autoqec/eval/independent_eval.py")
    if not src_path.exists():
        return  # File not yet created; will be tested when B1.2 lands
    src = src_path.read_text()
    # no bare 'from autoqec.runner' or 'import autoqec.runner'
    assert not re.search(r"(from|import)\s+autoqec\.runner", src), \
        "independent_eval.py must stay isolated from autoqec.runner.*"
