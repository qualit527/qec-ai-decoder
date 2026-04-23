from pathlib import Path

import yaml

from autoqec.decoders.dsl_compiler import compile_predecoder


def test_all_seed_templates_compile() -> None:
    for path in Path("autoqec/example_db").glob("*.yaml"):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = compile_predecoder(cfg, n_var=40, n_check=24)
        total = sum(parameter.numel() for parameter in model.parameters())
        assert total > 0, f"{path.name} produced an empty model"

