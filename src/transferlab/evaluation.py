from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from .config import Experiment
from .io import append_jsonl, digest, file_hash, read_jsonl, write_json
from .tasks import Task, verify


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("Invalid pass@k inputs")
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return 1.0 - product


def evaluate(
    config: Experiment,
    splits: dict[str, list[Task]],
    backend,
    output: Path,
    checkpoint: str,
    sandbox=None,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.jsonl"
    existing = read_jsonl(predictions) if predictions.exists() else []
    signature = digest(
        {
            "evaluation": config.evaluation.model_dump(),
            "recipe": config.recipe.model_dump(),
            "model": config.model.model_dump(),
            "backend": backend.name,
            "backend_fingerprint": getattr(backend, "fingerprint", None),
            "checkpoint": checkpoint,
            "tasks": {
                k: digest([t.to_dict() for t in v])
                for k, v in splits.items()
                if k.startswith("eval-")
            },
        }
    )
    contract = {
        "model": config.model.model_dump(),
        "recipe": config.recipe.model_dump(),
        "evaluation": config.evaluation.model_dump(),
        "tasks": {
            k.removeprefix("eval-"): {t.id: digest(t.to_dict()) for t in v}
            for k, v in splits.items()
            if k.startswith("eval-")
        },
    }
    evaluation_manifest = {
        "signature": signature,
        "contract": contract,
        "arm": config.arm,
        "seed": config.seed,
        "checkpoint": checkpoint,
        "fixture": backend.name == "fixture",
    }
    if any(row.get("signature") != signature for row in existing):
        raise ValueError("Evaluation resume fingerprint mismatch")
    keys = {(r["benchmark"], r["task_id"], r["sample"]) for r in existing}
    if len(keys) != len(existing):
        raise ValueError("Duplicate saved evaluation samples")
    for benchmark in config.evaluation.benchmarks:
        for task in splits["eval-" + benchmark.name]:
            for sample in range(config.evaluation.samples):
                key = (benchmark.name, task.id, sample)
                if key in keys:
                    continue
                sample_seed = int(
                    digest([config.evaluation.seed, benchmark.name, task.id, sample])[:8], 16
                )
                generation = backend.generate(task, sample_seed)
                verification_started = time.monotonic()
                correct, reason = verify(task, generation.text, sandbox)
                verifier_seconds = time.monotonic() - verification_started
                row = {
                    "signature": signature,
                    "backend": backend.name,
                    "fixture": backend.name == "fixture",
                    "arm": config.arm,
                    "seed": config.seed,
                    "checkpoint": checkpoint,
                    "benchmark": benchmark.name,
                    "task_id": task.id,
                    "family": task.family,
                    "distance": benchmark.distance[config.arm],
                    "skills": benchmark.skills,
                    "sample": sample,
                    "sample_seed": sample_seed,
                    "correct": correct,
                    "reason": reason,
                    "verifier_seconds": verifier_seconds,
                    **asdict(generation),
                }
                append_jsonl(predictions, row)
                keys.add(key)
    rows = read_jsonl(predictions)
    scores = {}
    for benchmark in config.evaluation.benchmarks:
        grouped = {}
        for row in rows:
            if row["benchmark"] == benchmark.name:
                grouped.setdefault(row["task_id"], []).append(row)
        scores[benchmark.name] = {
            "tasks": len(grouped),
            "distance": benchmark.distance[config.arm],
            "pass@1": sum(
                pass_at_k(len(v), sum(r["correct"] for r in v), 1) for v in grouped.values()
            )
            / len(grouped),
            "pass@4": sum(
                pass_at_k(len(v), sum(r["correct"] for r in v), 4) for v in grouped.values()
            )
            / len(grouped),
            "truncation_rate": sum(r["truncated"] for v in grouped.values() for r in v)
            / sum(map(len, grouped.values())),
        }
    result = {
        "schema_version": 1,
        "signature": signature,
        "backend": backend.name,
        "fixture": backend.name == "fixture",
        "checkpoint": checkpoint,
        "scores": scores,
        "predictions_sha256": file_hash(predictions),
        "manifest_sha256": digest(evaluation_manifest),
    }
    write_json(output / "manifest.json", evaluation_manifest)
    write_json(output / "metrics.json", result)
    return result
