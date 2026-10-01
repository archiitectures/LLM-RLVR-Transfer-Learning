from __future__ import annotations

from pathlib import Path

from .config import Experiment, require_sha
from .io import write_json


def evaluate_harness(
    config: Experiment,
    tasks: list[str],
    destination: Path,
    checkpoint: Path | None = None,
    limit: int | None = None,
) -> dict:
    """Supplementary likelihood benchmarks; kept distinct from sampled pass@k."""
    import torch
    from lm_eval import evaluator

    allowed = {"arc_easy", "arc_challenge", "hellaswag", "mmlu"}
    if not tasks or not set(tasks) <= allowed:
        raise ValueError(f"Supported supplementary harness tasks: {sorted(allowed)}")
    require_sha(config.model.revision, "Model")
    model_args = {
        "pretrained": config.model.id,
        "revision": config.model.revision,
        "dtype": config.model.dtype,
        "trust_remote_code": False,
    }
    if checkpoint:
        model_args["peft"] = str(checkpoint)
    result = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=1,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        num_fewshot=0,
        limit=limit,
        log_samples=True,
        random_seed=config.evaluation.seed,
        numpy_random_seed=config.evaluation.seed,
        torch_random_seed=config.evaluation.seed,
    )
    destination.mkdir(parents=True, exist_ok=True)
    # Harness results can contain numpy scalars; convert through its own JSON serializer.
    import json

    from lm_eval.utils import handle_non_serializable

    serial = json.loads(json.dumps(result, default=handle_non_serializable))
    write_json(destination / "harness.json", serial)
    write_json(
        destination / "provenance.json",
        {
            "model": config.model.model_dump(),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "tasks": tasks,
            "limit": limit,
            "metric": "supplementary likelihood accuracy; not sampled pass@k",
        },
    )
    return {"output": str(destination), "tasks": tasks, "results": serial["results"]}
