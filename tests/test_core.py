from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from transferlab.backends import FixtureBackend
from transferlab.cli import main
from transferlab.config import load_config
from transferlab.data import audit_splits, load_prepared, prepare
from transferlab.episodes import ToolEpisode
from transferlab.evaluation import evaluate, pass_at_k
from transferlab.io import read_json, read_jsonl
from transferlab.report import report
from transferlab.runner import run
from transferlab.sandbox import Execution
from transferlab.tasks import Task, extract_boxed, procedural, verify
from transferlab.training import ComputeMeter

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/smoke.yaml")


def test_all_configs_validate():
    for path in (ROOT / "configs").glob("*.yaml"):
        load_config(path)


def test_controlled_recipe_shared():
    configs = [
        load_config(ROOT / f"configs/{arm}.yaml")
        for arm in ("crypto", "benign_tools", "code", "math", "logic")
    ]
    assert all(c.recipe == configs[0].recipe for c in configs)
    assert all(c.evaluation == configs[0].evaluation for c in configs)


def test_freeze_and_detect_tampering(config, tmp_path):
    prepare(config, tmp_path, offline=True)
    _, manifest, splits = load_prepared(tmp_path)
    assert manifest["offline_fixture"]
    assert len(splits["train"]) == 8
    (tmp_path / "train.json").write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(tmp_path)


def test_normalized_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        audit_splits(
            {
                "train": [Task("a", "x", "What is X?", "1")],
                "eval": [Task("b", "x", " what  IS x? ", "1")],
            }
        )


@pytest.mark.parametrize(
    "text,expected",
    [
        (r"work \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"\boxed{2} and \boxed{3}", "3"),
        ("none", None),
    ],
)
def test_boxed_extraction(text, expected):
    assert extract_boxed(text) == expected


def test_numeric_verifier_never_evaluates_python(tmp_path):
    task = Task("a", "math", "x", "0.5", "numeric")
    assert verify(task, "<answer>1/2</answer>")[0]
    assert not verify(task, "__import__('os').system('touch forbidden')")[0]


def test_crypto_known_vectors():
    for family in ("caesar", "xor", "reverse"):
        task = procedural("crypto", 1, 123, [family])[0]
        assert verify(task, f"<answer>{task.answer}</answer>")[0]
        assert not verify(task, task.answer + "x")[0]


def test_procedural_logic_unique_and_disjoint():
    a = procedural("logic", 128, 123, [])
    b = procedural("logic", 32, 124, [])
    assert all(verify(t, t.answer)[0] for t in a)
    audit_splits({"train": a, "test": b})


class FakeSandbox:
    def execute(self, code, stdin=""):
        return Execution("42\n", "", "ok", 0.01)


def test_tool_environment_does_not_expose_reward(config, tmp_path):
    env = ToolEpisode(FakeSandbox(), config.recipe, tmp_path / "trace.jsonl")
    methods = [n for n, m in inspect.getmembers(env, inspect.ismethod) if not n.startswith("_")]
    assert set(methods) == {"reset", "execute_python", "submit_answer"}
    task = Task("a", "crypto", "question", "secret", tools=True)
    env.reset(json.dumps(task.to_dict()))
    assert env.submit_answer("secret") == "answer_recorded"
    assert env.submit_answer("replacement") == "episode_limit"
    assert env._reward() == 1
    env.reset(json.dumps(Task("b", "crypto", "other", "new", tools=True).to_dict()))
    assert env._answer is None
    assert env._reward() == 0


def test_tool_call_limit(config):
    config.recipe.max_tool_calls = 1
    env = ToolEpisode(FakeSandbox(), config.recipe)
    env.reset(json.dumps(Task("a", "x", "x", "42", tools=True).to_dict()))
    assert "42" in env.execute_python("print(42)")
    assert env.submit_answer("42") == "episode_limit"


def test_pass_at_k():
    assert pass_at_k(4, 1, 1) == 0.25
    assert pass_at_k(4, 1, 4) == 1
    assert pass_at_k(4, 0, 4) == 0
    assert pass_at_k(4, 4, 1) == 1
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 4)


def test_eval_resume_does_not_duplicate(config, tmp_path):
    data = tmp_path / "data"
    prepare(config, data, offline=True)
    cfg, _, splits = load_prepared(data)
    path = tmp_path / "eval"
    evaluate(cfg, splits, FixtureBackend(), path, "baseline")
    rows = read_jsonl(path / "predictions.jsonl")
    evaluate(cfg, splits, FixtureBackend(), path, "baseline")
    assert read_jsonl(path / "predictions.jsonl") == rows
    cfg.evaluation.temperature = 0.8
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate(cfg, splits, FixtureBackend(), path, "baseline")


def test_full_fixture_pipeline_and_report(config, tmp_path):
    data, output = tmp_path / "data", tmp_path / "runs"
    prepare(config, data, offline=True)
    result = run(data, output, fixture=True, project=ROOT)
    assert result["status"] == "completed"
    assert read_json(output / "train/training.json")["fixture"]
    run(data, output, fixture=True, resume=True, project=ROOT)
    with pytest.raises(ValueError, match="Fixture"):
        report(output, tmp_path / "report")
    result = report(output, tmp_path / "report", allow_fixtures=True)
    assert result["fixture"]
    assert all(r["delta"] == 0 for r in result["aggregate"])


def test_fixture_manifest_cannot_run_real_model(config, tmp_path):
    prepare(config, tmp_path / "data", offline=True)
    with pytest.raises(ValueError, match="fixture"):
        run(tmp_path / "data", tmp_path / "runs")


def test_compute_counts_decode_and_backward():
    meter = ComputeMeter(1000, 10, 2, 8, checkpointing=False)
    prefill = meter.charge(1, 10, 10, False)
    decode = meter.charge(1, 1, 11, False)
    update = meter.charge(1, 10, 10, True)
    assert decode < prefill < update
    assert meter.forward_tokens == 21
    assert meter.update_tokens == 10
    clone = ComputeMeter(1000, 10, 2, 8, False)
    clone.restore(meter.state())
    assert clone.flops == meter.flops


def test_cli_validate_and_fixture_smoke(tmp_path):
    assert main(["validate", str(ROOT / "configs/smoke.yaml")]) == 0
    assert (
        main(
            [
                "prepare",
                str(ROOT / "configs/smoke.yaml"),
                "--output",
                str(tmp_path / "data"),
                "--offline-fixture",
            ]
        )
        == 0
    )

    assert (
        main(["run", str(tmp_path / "data"), "--output", str(tmp_path / "runs"), "--fixture"]) == 0
    )
    assert (
        main(["evaluate", str(tmp_path / "data"), "--output", str(tmp_path / "eval"), "--fixture"])
        == 0
    )
    assert (
        main(
            [
                "report",
                str(tmp_path / "runs"),
                "--output",
                str(tmp_path / "report"),
                "--allow-fixtures",
            ]
        )
        == 0
    )


def test_all_arm_smoke_command(tmp_path):
    assert main(["smoke", "--output", str(tmp_path / "smoke")]) == 0
    result = read_json(tmp_path / "smoke/report/transfer.json")
    assert result["fixture"]
    assert {r["arm"] for r in result["aggregate"]} == {
        "crypto",
        "benign_tools",
        "code",
        "math",
        "logic",
    }
