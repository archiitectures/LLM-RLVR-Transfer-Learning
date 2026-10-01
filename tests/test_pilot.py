from pathlib import Path

import pytest

from transferlab.config import DataSource, load_config
from transferlab.data import prepare
from transferlab.pilot import freeze_budget


def test_fixture_cannot_supply_paper_compute_budget(tmp_path):
    template = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    paths = []
    for index, arm in enumerate(("crypto", "benign_tools", "code", "math", "logic")):
        config = template.model_copy(deep=True)
        config.arm = config.name = arm
        config.train = DataSource(kind="procedural", adapter="math", count=2, seed=1000 + index)
        config.validation = DataSource(
            kind="procedural", adapter="math", count=2, seed=2000 + index
        )
        path = tmp_path / arm
        prepare(config, path, offline=True)
        paths.append(path)
    with pytest.raises(ValueError, match="fixture"):
        freeze_budget(paths, tmp_path / "pilot", tmp_path / "budget.json")
