import json
from pathlib import Path

import pytest

from transferlab.checkpoints import (
    REQUIRED,
    latest_checkpoint,
    retain_checkpoints,
    seal_checkpoint,
    verified_checkpoint,
)
from transferlab.codec import encode
from transferlab.config import SandboxConfig, load_config
from transferlab.data import prepare
from transferlab.io import digest, file_hash, read_json, read_jsonl, write_json
from transferlab.report import report
from transferlab.runner import run
from transferlab.sandbox import Execution
from transferlab.tasks import Task, VerifierInfrastructureError, verify


def checkpoint(root, step):
    path = root / f"checkpoint-{step}"
    path.mkdir()
    for name in REQUIRED:
        (path / name).write_text("{}")
    seal_checkpoint(path)
    return path


def test_recovery_skips_partial_and_corrupt_checkpoints(tmp_path):
    good = checkpoint(tmp_path, 1)
    corrupt = checkpoint(tmp_path, 2)
    (corrupt / "optimizer.pt").write_text("truncated")
    (tmp_path / "checkpoint-3").mkdir()
    assert latest_checkpoint(tmp_path) == good
    assert not verified_checkpoint(corrupt)


def test_retention_keeps_milestone_adapters_and_two_recovery_points(tmp_path):
    paths = [checkpoint(tmp_path, i) for i in range(1, 7)]
    milestones = {"0.25": {"checkpoint": "checkpoint-1"}}
    retain_checkpoints(tmp_path, milestones)
    assert (paths[0] / "adapter_model.safetensors").exists()
    assert not (paths[0] / "optimizer.pt").exists()
    assert not paths[1].exists()
    assert verified_checkpoint(paths[-1]) and verified_checkpoint(paths[-2])
    retain_checkpoints(tmp_path, milestones, finished=True)
    assert not paths[-1].exists()
    assert (paths[0] / "adapter_model.safetensors").exists()


@pytest.fixture
def completed(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    prepared, output = tmp_path / "data", tmp_path / "run"
    prepare(config, prepared, offline=True)
    run(prepared, output, fixture=True)
    return output


def test_reports_reject_failed_runs(completed, tmp_path):
    state = read_json(completed / "run.json")
    state["status"] = "failed"
    write_json(completed / "run.json", state)
    with pytest.raises(ValueError, match="completed"):
        report(completed, tmp_path / "report", allow_fixtures=True)


def test_resume_allows_replacement_host_kernel_but_rejects_changed_packages():
    from transferlab.io import portable_provenance

    previous = {
        "source": "abc",
        "environment": {
            "platform": "Linux-host-one",
            "python": "3.12",
            "packages": [["torch", "2.14"]],
            "gpu": "same GPU",
        },
    }
    replacement = {
        **previous,
        "environment": {**previous["environment"], "platform": "Linux-host-two"},
    }
    assert portable_provenance(previous) == portable_provenance(replacement)
    replacement["environment"]["packages"] = [["torch", "2.15"]]
    assert portable_provenance(previous) != portable_provenance(replacement)


def test_report_rejects_missing_milestone(completed, tmp_path):
    (completed / "eval/fixture-final/predictions.jsonl").unlink()
    with pytest.raises(ValueError, match="checkpoints"):
        report(completed, tmp_path / "report", allow_fixtures=True)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "identity", "contract"])
def test_reports_reject_corrupt_evaluation_even_with_updated_checksums(
    completed, tmp_path, mutation
):
    directory = completed / "eval/fixture-final"
    path = directory / "predictions.jsonl"
    rows = read_jsonl(path)
    manifest = read_json(directory / "manifest.json")
    if mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "missing":
        rows.pop()
    elif mutation == "identity":
        rows[0]["seed"] += 1
    else:
        manifest["contract"]["model"]["id"] = "different/model"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(directory / "manifest.json", manifest)
    metrics = read_json(directory / "metrics.json")
    metrics.update(predictions_sha256=file_hash(path), manifest_sha256=digest(manifest))
    write_json(directory / "metrics.json", metrics)
    with pytest.raises(ValueError):
        report(completed, tmp_path / "report", allow_fixtures=True)


@pytest.mark.parametrize(
    "status,stdout", [("execution_error", ""), ("timeout", ""), ("ok", "malformed")]
)
def test_checker_failure_is_not_a_wrong_answer(status, stdout):
    class Sandbox:
        config = SandboxConfig()

        def execute(self, program, **kwargs):
            assert kwargs["timeout_seconds"] >= 130
            assert kwargs["memory_mb"] >= 512
            return Execution(stdout, "checker problem", status, 0)

    task = Task(
        "a",
        "code",
        "implement f",
        "",
        "evalplus",
        metadata={
            "problem": encode({"entry_point": "f", "base_input": [], "plus_input": [], "atol": 0}),
            "oracle": encode({}),
            "dataset": "humaneval",
        },
    )
    with pytest.raises(VerifierInfrastructureError):
        verify(task, "def f(): pass", Sandbox())
