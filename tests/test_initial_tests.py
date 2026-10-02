from pathlib import Path

import pytest

from transferlab.initial_tests import initial_tests
from transferlab.io import read_json

ROOT = Path(__file__).resolve().parents[1]


def test_missing_docker_records_blocked_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr("transferlab.initial_tests.shutil.which", lambda name: None)
    result = initial_tests("sandbox", ROOT, tmp_path / "qualification")
    assert result["status"] == "blocked"
    assert result["rentals_created"] == 0
    assert "Docker" in result["error"]
    assert read_json(tmp_path / "qualification/readiness.json")["status"] == "blocked"


def test_initial_test_outputs_are_never_overwritten(tmp_path):
    (tmp_path / "existing").write_text("user artifact")
    with pytest.raises(ValueError, match="new or empty"):
        initial_tests("cpu", ROOT, tmp_path)
    assert (tmp_path / "existing").read_text() == "user artifact"


def test_gpu_qualification_requires_all_five_real_manifests(tmp_path):
    result = initial_tests("gpu", ROOT, tmp_path / "qualification", prepared=[])
    assert result["status"] == "failed"
    assert "five prepared" in result["error"]
