from __future__ import annotations

import csv
import random
import statistics
from pathlib import Path

from .evaluation import pass_at_k
from .io import digest, file_hash, portable_provenance, read_json, read_jsonl, write_json


def validated_predictions(path: Path, state: dict) -> tuple[list[dict], dict]:
    manifest = read_json(path.parent / "manifest.json")
    metrics = read_json(path.parent / "metrics.json")
    if metrics.get("predictions_sha256") != file_hash(path) or metrics.get(
        "manifest_sha256"
    ) != digest(manifest):
        raise ValueError(f"Evaluation artifact checksum mismatch: {path}")
    provenance = state["provenance"]
    config = provenance["config"]
    contract = manifest["contract"]
    for field in ("model", "recipe", "evaluation"):
        if contract[field] != config[field]:
            raise ValueError(f"Evaluation {field} differs from run provenance")
    if contract["tasks"] != provenance["evaluation_tasks"]:
        raise ValueError("Evaluation tasks differ from frozen run provenance")
    rows = read_jsonl(path)
    expected = {
        (benchmark, task, sample)
        for benchmark, tasks in contract["tasks"].items()
        for task in tasks
        for sample in range(config["evaluation"]["samples"])
    }
    keys = [(r["benchmark"], r["task_id"], r["sample"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicate, or unexpected evaluation samples")
    for row in rows:
        if any(
            row[field] != manifest[field]
            for field in ("signature", "arm", "seed", "checkpoint", "fixture")
        ):
            raise ValueError("Mixed evaluation provenance")
        if (
            row["arm"] != config["arm"]
            or row["seed"] != config["seed"]
            or row["fixture"] != provenance["fixture"]
        ):
            raise ValueError("Prediction identity differs from run provenance")
        if type(row["correct"]) is not bool:
            raise ValueError("Invalid correctness value")
    return rows, contract


def _task_scores(rows: list[dict], k: int) -> dict[str, float]:
    tasks = {}
    for row in rows:
        tasks.setdefault(row["task_id"], []).append(row)
    return {t: pass_at_k(len(v), sum(r["correct"] for r in v), k) for t, v in tasks.items()}


def bootstrap(values: list[float], seed: int = 1729, repeats: int = 2000) -> list[float]:
    if not values:
        raise ValueError("No paired observations")
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return [draws[int(repeats * 0.025)], draws[min(repeats - 1, int(repeats * 0.975))]]


def report(
    root: Path, destination: Path, *, allow_fixtures: bool = False, plots: bool = False
) -> dict:
    records = []
    contracts = {}
    expected_seeds = {}
    suite_contract = None
    # Pair within the same run: no accidental pairing across seeds, models, or eval recipes.
    for baseline_path in sorted(root.rglob("eval/baseline/predictions.jsonl")):
        state = read_json(baseline_path.parents[2] / "run.json")
        if state["status"] != "completed" or digest(state["provenance"]) != state["fingerprint"]:
            raise ValueError("Reports require completed runs with intact provenance")
        if state["provenance"].get("pilot"):
            raise ValueError("Pilot runs cannot supply paper transfer results")
        baseline, contract = validated_predictions(baseline_path, state)
        provenance = portable_provenance(state["provenance"])
        common = {
            "model": contract["model"],
            "recipe": contract["recipe"],
            "evaluation": {
                **contract["evaluation"],
                "benchmarks": [
                    b for b in contract["evaluation"]["benchmarks"] if b["name"] != "validation"
                ],
            },
            "tasks": {k: v for k, v in contract["tasks"].items() if k != "validation"},
            "source": provenance["source"],
            "environment": provenance["environment"],
            "sandbox_image_ids": provenance["sandbox_image_ids"],
            "study": provenance["config"]["study"],
        }
        if suite_contract is not None and suite_contract != common:
            raise ValueError("Suite runs have incompatible scientific provenance")
        suite_contract = common
        arm = state["provenance"]["config"]["arm"]
        signature = digest(contract)
        if arm in contracts and contracts[arm] != signature:
            raise ValueError("Runs have incompatible model, recipe or evaluation contracts")
        contracts[arm] = signature
        expected_seeds[arm] = set(state["provenance"]["config"]["study"]["seeds"])
        if any(r["fixture"] for r in baseline) and not allow_fixtures:
            raise ValueError(
                "Fixture predictions are not scientific results; pass --allow-fixtures for plumbing reports"
            )
        run_eval = baseline_path.parent.parent
        training = read_json(run_eval.parent / "train" / "training.json")
        expected_checkpoints = (
            {"fixture-final"}
            if state["provenance"]["fixture"]
            else {
                "final",
                *(
                    f"budget-{round(float(k) * 100)}"
                    for k in training["milestones"]
                    if float(k) < 1
                ),
            }
        )
        actual_checkpoints = {
            p.parent.name for p in run_eval.glob("*/predictions.jsonl") if p != baseline_path
        }
        if actual_checkpoints != expected_checkpoints:
            raise ValueError("Missing or unexpected evaluation checkpoints")
        for path in sorted(run_eval.glob("*/predictions.jsonl")):
            if path == baseline_path:
                continue
            trained, trained_contract = validated_predictions(path, state)
            if any(r["checkpoint"] != path.parent.name for r in trained):
                raise ValueError("Checkpoint label differs from evaluation directory")
            if trained_contract != contract:
                raise ValueError("Baseline and trained evaluation contracts differ")
            if any(r["fixture"] for r in trained) and not allow_fixtures:
                raise ValueError("Mixed fixture/scientific predictions")
            for benchmark in sorted({r["benchmark"] for r in baseline}):
                before = [r for r in baseline if r["benchmark"] == benchmark]
                after = [r for r in trained if r["benchmark"] == benchmark]
                if not after:
                    raise ValueError(f"Missing benchmark {benchmark} in {path}")
                for k in (1, 4):
                    b, a = _task_scores(before, k), _task_scores(after, k)
                    if b.keys() != a.keys():
                        raise ValueError(f"Unpaired tasks in {path}")
                    deltas = [a[t] - b[t] for t in sorted(a)]
                    records.append(
                        {
                            "run": str(path.parent.parent.parent),
                            "arm": after[0]["arm"],
                            "seed": after[0]["seed"],
                            "checkpoint": after[0]["checkpoint"],
                            "benchmark": benchmark,
                            "distance": after[0]["distance"],
                            "metric": f"pass@{k}",
                            "baseline": statistics.mean(b.values()),
                            "trained": statistics.mean(a.values()),
                            "delta": statistics.mean(deltas),
                            "paired_ci95": bootstrap(deltas),
                            "tasks": len(deltas),
                            "mean_completion_tokens": statistics.mean(
                                r["completion_tokens"] for r in after
                            ),
                            "mean_tool_calls": statistics.mean(r["tool_calls"] for r in after),
                            "failure_rate": statistics.mean(r["status"] != "ok" for r in after),
                            "task_deltas": dict(zip(sorted(a), deltas, strict=True)),
                            "fixture": after[0]["fixture"],
                        }
                    )
    if not records:
        raise ValueError("No paired baseline/trained predictions found")
    grouped = {}
    for r in records:
        key = (r["arm"], r["checkpoint"], r["benchmark"], r["metric"])
        grouped.setdefault(key, []).append(r)
    aggregate = []
    for key, rows in sorted(grouped.items()):
        if len({r["seed"] for r in rows}) != len(rows):
            raise ValueError(f"Repeated seed/run for {key}; select a single suite root")
        if {r["seed"] for r in rows} != expected_seeds[key[0]]:
            raise ValueError(f"Incomplete study seeds for {key}")
        sets = [set(r["task_deltas"]) for r in rows]
        if any(s != sets[0] for s in sets):
            raise ValueError("Seeds evaluated on different task manifests")
        task_means = [statistics.mean(r["task_deltas"][t] for r in rows) for t in sorted(sets[0])]
        aggregate.append(
            {
                "arm": key[0],
                "checkpoint": key[1],
                "benchmark": key[2],
                "metric": key[3],
                "distance": rows[0]["distance"],
                "seeds": len(rows),
                "delta": statistics.mean(r["delta"] for r in rows),
                "paired_task_ci95": bootstrap(task_means),
                "seed_sd": statistics.stdev(r["delta"] for r in rows) if len(rows) > 1 else None,
            }
        )
    tiers = {}
    for r in aggregate:
        key = (r["arm"], r["checkpoint"], r["metric"], r["distance"])
        tiers.setdefault(key, []).append(r["delta"])
    tier_scores = [
        {
            "arm": k[0],
            "checkpoint": k[1],
            "metric": k[2],
            "distance": k[3],
            "mean_benchmark_delta": statistics.mean(v),
            "benchmark_count": len(v),
        }
        for k, v in sorted(tiers.items())
    ]
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "fixture": any(r["fixture"] for r in records),
        "interval_unit": "paired tasks within benchmark, averaged over seeds; not independent distance observations",
        "per_seed": records,
        "aggregate": aggregate,
        "distance_tiers": tier_scores,
    }
    write_json(destination / "transfer.json", result)
    with (destination / "transfer.csv").open("w", newline="") as f:
        fields = [
            "arm",
            "checkpoint",
            "benchmark",
            "metric",
            "distance",
            "seeds",
            "delta",
            "paired_task_ci95",
            "seed_sd",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    lines = [
        "# Transfer results",
        "",
        "PLUMBING FIXTURE — no model trained."
        if result["fixture"]
        else "Changes relative to the unchanged-model baseline.",
        "",
        "| Arm | Checkpoint | Benchmark | Metric | Distance | Δ | Seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in aggregate:
        lines.append(
            f"| {r['arm']} | {r['checkpoint']} | {r['benchmark']} | {r['metric']} | {r['distance']} | {100 * r['delta']:+.2f} pp | {r['seeds']} |"
        )
    lines += [
        "",
        "Task bootstrap intervals and seed variability are separate. Distance tiers are descriptive; question count does not determine evidence for a distance effect.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines))
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for arm in sorted({r["arm"] for r in tier_scores}):
            points = [
                r
                for r in tier_scores
                if r["arm"] == arm
                and r["metric"] == "pass@1"
                and r["checkpoint"] in {"final", "fixture-final"}
            ]
            points.sort(key=lambda r: r["distance"])
            ax.plot(
                [p["distance"] for p in points],
                [100 * p["mean_benchmark_delta"] for p in points],
                marker="o",
                label=arm,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            xlabel="Predefined conceptual distance (ordinal)",
            ylabel="Mean benchmark gain (percentage points)",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / "distance-transfer.png", dpi=200)
        plt.close(fig)
    return result
