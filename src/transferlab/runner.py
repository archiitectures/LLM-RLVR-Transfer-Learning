from __future__ import annotations

from pathlib import Path

from .backends import FixtureBackend, HFBackend
from .checkpoints import checkpoint_paths, latest_checkpoint
from .config import Benchmark, Experiment
from .data import load_prepared
from .evaluation import evaluate
from .io import (
    digest,
    environment,
    portable_provenance,
    read_json,
    source_fingerprint,
    timestamp,
    write_json,
)
from .sandbox import DockerSandbox, FixtureSandbox
from .training import train


def run(
    prepared: Path,
    destination: Path,
    *,
    seed: int | None = None,
    fixture: bool = False,
    resume: bool = False,
    baseline_only: bool = False,
    pilot: bool = False,
    project: Path | None = None,
    compute_flops: float | None = None,
    pilot_steps: int = 8,
    pilot_tasks: int = 4,
) -> dict:
    config, manifest, splits = load_prepared(prepared)
    if seed is not None:
        config.seed = seed
    if compute_flops is not None:
        config.recipe.compute_flops = compute_flops
    if pilot:
        config.recipe.compute_flops = None
        config.recipe.max_steps = pilot_steps
        splits = {k: v[:pilot_tasks] if k.startswith("eval-") else v for k, v in splits.items()}
    if "validation" not in {b.name for b in config.evaluation.benchmarks}:
        config.evaluation.benchmarks.append(
            Benchmark(
                name="validation",
                source=config.validation,
                distance={config.arm: 0},
                skills=["training-domain-holdout"],
            )
        )
        splits["eval-validation"] = (
            splits["validation"][:pilot_tasks] if pilot else splits["validation"]
        )
    config = Experiment.model_validate(config.model_dump())
    if manifest["offline_fixture"] and not fixture:
        raise ValueError("Offline fixture manifests cannot run real models")
    if not fixture and config.model.dtype == "bfloat16":
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError(
                "Paper runs require a BF16-compatible CUDA GPU before baseline evaluation"
            )

    sandbox = FixtureSandbox() if fixture else None
    if not fixture and any(
        t.tools or t.verifier in {"stdio", "assertions", "function_cases", "evalplus"}
        for ts in splits.values()
        for t in ts
    ):
        sandbox = DockerSandbox(config.sandbox)
        sandbox.preflight(
            evalplus=any(t.verifier == "evalplus" for ts in splits.values() for t in ts)
        )
    destination.mkdir(parents=True, exist_ok=True)
    env = environment()
    provenance = {
        "config": config.model_dump(),
        "manifest": manifest,
        "source": source_fingerprint(project or Path.cwd()),
        "environment": env,
        "fixture": fixture,
        "pilot": pilot,
        "sandbox_image_ids": getattr(sandbox, "image_ids", {}),
        "evaluation_tasks": {
            k.removeprefix("eval-"): {t.id: digest(t.to_dict()) for t in ts}
            for k, ts in splits.items()
            if k.startswith("eval-")
        },
    }
    fingerprint = digest(provenance)
    state_path = destination / "run.json"
    if state_path.exists():
        previous = read_json(state_path)
        if not resume:
            raise ValueError("Run already exists; use --resume or choose another output directory")
        if digest(previous["provenance"]) != previous["fingerprint"] or digest(
            portable_provenance(previous["provenance"])
        ) != digest(portable_provenance(provenance)):
            raise ValueError("Run resume requires identical code, environment, config, and data")
        if previous["status"] == "completed":
            return previous
    state = {
        "fingerprint": fingerprint,
        "provenance": provenance,
        "started": timestamp(),
        "status": "running",
    }
    write_json(state_path, state)
    try:
        backend = FixtureBackend() if fixture else HFBackend(config, sandbox)
        evaluate(config, splits, backend, destination / "eval" / "baseline", "baseline", sandbox)
        del backend
        if not baseline_only:
            if fixture:
                result = {
                    "fixture": True,
                    "warning": "No training occurred; plumbing fixture only",
                    "milestones": {},
                }
                write_json(destination / "train" / "training.json", result)
                evaluate(
                    config,
                    splits,
                    FixtureBackend(),
                    destination / "eval" / "fixture-final",
                    "fixture-final",
                    sandbox,
                )
            else:
                completed_training = destination / "train" / "training.json"
                checkpoints = checkpoint_paths(destination / "train")
                checkpoint = latest_checkpoint(destination / "train") if resume else None
                if (
                    resume
                    and completed_training.exists()
                    and (destination / "train" / "final" / "adapter_model.safetensors").exists()
                ):
                    result = read_json(completed_training)
                else:
                    if resume and checkpoints and checkpoint is None:
                        raise ValueError(
                            "No verified complete recovery checkpoint; restore an earlier backup"
                        )
                    result = train(
                        config, splits["train"], destination / "train", sandbox, checkpoint, pilot
                    )
                evaluate_checkpoints = {
                    f"budget-{round(float(k) * 100)}": v["checkpoint"]
                    for k, v in result["milestones"].items()
                    if float(k) < 1
                }
                evaluate_checkpoints["final"] = "final"
                for label, name in sorted(evaluate_checkpoints.items()):
                    backend = HFBackend(config, sandbox, destination / "train" / name)
                    evaluate(config, splits, backend, destination / "eval" / label, label, sandbox)
                    del backend
        state["status"] = "completed"
        state["finished"] = timestamp()
        write_json(state_path, state)
        return state
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        write_json(state_path, state)
        raise
