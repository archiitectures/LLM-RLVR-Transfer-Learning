from __future__ import annotations

import statistics
from pathlib import Path

from .data import load_prepared
from .io import digest, read_json, read_jsonl, write_json


def freeze_budget(prepared_paths: list[Path], pilot_root: Path, output: Path) -> dict:
    configs = [load_prepared(p)[0] for p in prepared_paths]
    expected = {"crypto", "benign_tools", "code", "math", "logic"}
    if {c.arm for c in configs} != expected or len(configs) != 5:
        raise ValueError("Pilot budgeting requires exactly one prepared config for every arm")
    study = configs[0].study
    rates = []
    diagnostics = []
    evaluation_cost = 0.0
    manifests = [load_prepared(p)[1] for p in prepared_paths]
    if any(m["offline_fixture"] for m in manifests):
        raise ValueError("Cannot freeze paper budgets from fixture data")
    for config in configs:
        path = pilot_root / config.arm
        run_state = read_json(path / "run.json")
        if run_state["status"] != "completed" or run_state["provenance"]["fixture"]:
            raise ValueError("Only completed real pilots can freeze a study")
        if run_state["provenance"]["manifest"] != manifests[configs.index(config)]:
            raise ValueError("Pilot data does not match the prepared paper manifest")
        if run_state["provenance"]["environment"].get("gpu") is None:
            raise ValueError("Paper budgets require measured CUDA GPU pilots")
        training = read_json(path / "train" / "training.json")
        if training.get("fixture"):
            raise ValueError("Cannot derive a paper budget from fixtures")
        reward_rows = read_jsonl(path / "train" / "rewards.jsonl")
        rewards = [float(r["correct"]) for r in reward_rows]
        if not rewards or min(rewards) == max(rewards):
            raise ValueError(
                f"{config.arm}: no usable reward variation; revise pilot before paper runs"
            )
        group_size = config.recipe.group_size
        groups = [rewards[i : i + group_size] for i in range(0, len(rewards), group_size)]
        informative = sum(len(g) == group_size and min(g) != max(g) for g in groups)
        if informative == 0:
            raise ValueError(
                f"{config.arm}: reward variation exists only between groups; no learning signal"
            )
        if training.get("nonzero_gradient_updates", 0) == 0:
            raise ValueError(f"{config.arm}: no nonzero adapter gradient updates observed")
        flops = training["compute"]["flops"]
        seconds = training["seconds"]
        if seconds <= 0 or flops <= 0 or training["compute"]["update_tokens"] <= 0:
            raise ValueError(f"{config.arm}: invalid measured compute/update throughput")
        throughput = flops / seconds
        rate = config.vast.max_rate  # conservative hourly ceiling, not a claimed market price
        runs = len(configs) * len(study.seeds)
        affordable_seconds = study.training_usd / runs / rate * 3600 * 0.75
        capped_seconds = config.vast.max_hours * 3600 * 0.75
        rates.append(min(affordable_seconds, capped_seconds) * throughput)
        baseline_rows = read_jsonl(path / "eval" / "baseline" / "predictions.jsonl")
        if not baseline_rows or any(r.get("fixture") for r in baseline_rows):
            raise ValueError("Pilot evaluation must contain real measured predictions")
        per_benchmark = {}
        for row in baseline_rows:
            per_benchmark.setdefault(row["benchmark"], []).append(
                row["seconds"] + row["verifier_seconds"]
            )
        expected_counts = {b.name: b.source.count for b in config.evaluation.benchmarks}
        expected_counts["validation"] = config.validation.count
        if set(per_benchmark) != set(expected_counts):
            raise ValueError("Pilot is missing an evaluation benchmark")
        eval_seconds = sum(
            statistics.mean(per_benchmark[name]) * count * config.evaluation.samples
            for name, count in expected_counts.items()
        )
        # Baseline + three milestones. Use 50% headroom for sampling/length differences after training.
        arm_eval_cost = eval_seconds * 4 * len(study.seeds) / 3600 * rate * 1.5
        evaluation_cost += arm_eval_cost
        diagnostics.append(
            {
                "arm": config.arm,
                "estimated_flops_per_second": throughput,
                "hourly_ceiling": rate,
                "reward_mean": statistics.mean(rewards),
                "reward_samples": len(rewards),
                "informative_groups": informative,
                "estimated_evaluation_usd": arm_eval_cost,
            }
        )
    common = min(rates)
    if evaluation_cost > study.evaluation_usd:
        raise ValueError(
            f"Estimated evaluation cost ${evaluation_cost:.2f} exceeds ${study.evaluation_usd:.2f} allocation; revise and rerun pilot before freezing"
        )
    # A feasibility estimate is conservative; evaluation/setup overhead still needs explicit headroom.
    result = {
        "schema_version": 1,
        "compute_flops": common,
        "arms": diagnostics,
        "study": study.model_dump(),
        "budget_method": "slowest arm; 25% training headroom; hourly ceilings",
        "estimated_evaluation_usd": evaluation_cost,
        "evaluation_cost_check_passed": True,
        "prepared_manifest_sha256": {
            c.arm: digest(m) for c, m in zip(configs, manifests, strict=True)
        },
    }
    write_json(output, result)
    return result
