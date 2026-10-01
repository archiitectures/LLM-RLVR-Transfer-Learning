from __future__ import annotations

import csv
import io
import json
import random
import re
from pathlib import Path

import httpx

from .config import DataSource, Experiment, require_sha
from .io import digest, file_hash, read_json, read_jsonl, write_json
from .tasks import Task, extract_boxed, procedural


def normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def audit_splits(splits: dict[str, list[Task]]) -> dict:
    seen = {}
    duplicates = []
    for split, tasks in splits.items():
        local_ids = set()
        for task in tasks:
            if task.id in local_ids:
                raise ValueError(f"Duplicate task ID in {split}: {task.id}")
            local_ids.add(task.id)
            key = normalized_prompt(task.prompt)
            if key in seen:
                duplicates.append({"first": seen[key], "second": split, "id": task.id})
            else:
                seen[key] = split
    cross_split = [d for d in duplicates if d["first"] != d["second"]]
    if cross_split:
        raise ValueError(f"Train/validation/test overlap found: {cross_split[:5]}")
    return {
        "normalized_cross_split_overlaps": 0,
        "within_split_duplicates": duplicates,
        "pretraining_contamination": "unknown; model pretraining data cannot be audited here",
    }


def _hf_tasks(source: DataSource) -> tuple[list[Task], dict]:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    if not source.id:
        raise ValueError("HF source requires id")
    revision = HfApi().dataset_info(source.id, revision=source.revision).sha
    require_sha(revision, source.id)
    ds = load_dataset(source.id, name=source.subset, split=source.split, revision=revision)
    ds = ds.shuffle(seed=source.seed)
    tasks = []
    skipped = 0
    for row in ds:
        ident = digest({"dataset": source.id, "revision": revision, "row": row})
        task = None
        if source.adapter == "numina":
            answer = extract_boxed(row.get("solution", ""))
            if answer is not None:
                task = Task(ident, "math", row["problem"], answer, "math")
        elif source.adapter == "math500":
            task = Task(ident, "math", row["problem"], str(row["answer"]), "math")
        elif source.adapter == "taco":
            io = (
                json.loads(row["input_output"])
                if isinstance(row.get("input_output"), str)
                else row.get("input_output")
            )
            # Callable tasks need a distinct runner. Never silently reinterpret them as stdin programs.
            if (
                io
                and not io.get("fn_name")
                and io.get("inputs")
                and len(io["inputs"]) == len(io["outputs"])
            ):
                if all(isinstance(x, str) for x in io["inputs"] + io["outputs"]):
                    tests = [
                        {"input": i, "output": o}
                        for i, o in zip(io["inputs"], io["outputs"], strict=True)
                    ]
                    task = Task(
                        ident,
                        "stdio",
                        row["question"] + "\nWrite a complete Python stdin/stdout program.",
                        "",
                        "stdio",
                        metadata={"tests": tests},
                    )
        elif source.adapter == "humaneval":
            task = Task(
                ident,
                "function_code",
                "Complete this Python function. Return the complete code:\n" + row["prompt"],
                "",
                "assertions",
                metadata={"test_code": row["test"] + f"\ncheck({row['entry_point']})"},
            )
        elif source.adapter == "mbpp":
            task = Task(
                ident,
                "function_code",
                row.get("text", row.get("prompt", "")),
                "",
                "assertions",
                metadata={"test_code": "\n".join(row["test_list"])},
            )
        elif source.adapter == "arc":
            choices = row["choices"]
            prompt = (
                row["question"]
                + "\n"
                + "\n".join(
                    f"{label}. {text}"
                    for label, text in zip(choices["label"], choices["text"], strict=True)
                )
            )
            task = Task(
                ident,
                "knowledge",
                prompt + "\nReturn only the answer label.",
                str(row["answerKey"]),
            )
        elif source.adapter == "zebra":
            puzzle = row.get("puzzle", row)
            if isinstance(puzzle, str):
                try:
                    puzzle = json.loads(puzzle)
                except ValueError:
                    pass
            solution = row.get("solution", row.get("answer"))
            if solution is not None:
                if "___" in json.dumps(solution):
                    raise ValueError(
                        "Zebra dataset has redacted answers; supply an authorized solved dataset"
                    )
                task = Task(
                    ident,
                    "zebra",
                    json.dumps(puzzle) + "\nReturn the solution as JSON in <answer> tags.",
                    json.dumps(solution, sort_keys=True),
                    "exact_json",
                )
        else:
            raise ValueError(f"Unsupported HF adapter: {source.adapter}")
        if task:
            tasks.append(task)
        else:
            skipped += 1
        if len(tasks) == source.count:
            break
    if len(tasks) < source.count:
        raise ValueError(
            f"{source.id}: only {len(tasks)} usable rows, requested {source.count}; {skipped} skipped"
        )
    return tasks, {**source.model_dump(), "revision": revision, "skipped_rows": skipped}


def load_source(source: DataSource) -> tuple[list[Task], dict]:
    if source.kind == "github":
        if not source.id or not source.path:
            raise ValueError("GitHub data requires owner/repo id and repository-relative path")
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source.id)
            or ".." in Path(source.path).parts
        ):
            raise ValueError("Invalid GitHub repository/path")
        ref = source.revision or "HEAD"
        r = httpx.get(
            f"https://api.github.com/repos/{source.id}/commits/{ref}",
            timeout=30,
            follow_redirects=True,
        )
        r.raise_for_status()
        revision = r.json()["sha"]
        require_sha(revision, source.id)
        r = httpx.get(
            f"https://raw.githubusercontent.com/{source.id}/{revision}/{source.path}",
            timeout=30,
            follow_redirects=True,
        )
        r.raise_for_status()
        raw = r.text
        if source.adapter == "random_crypto":
            rows = list(csv.DictReader(io.StringIO(raw)))
            if source.families:
                rows = [
                    r
                    for r in rows
                    if r.get("subtype", r.get("cipher", "")).casefold()
                    in [f.casefold() for f in source.families]
                ]
            rows = rows[source.offset : source.offset + source.count]
            random.Random(source.seed).shuffle(rows)
            tasks = [
                Task(
                    digest([source.id, revision, row]),
                    row.get("subtype", row.get("cipher", "crypto")),
                    row.get("story", row.get("question", ""))
                    + "\n"
                    + row.get("necessary_info", ""),
                    row["flag"],
                    tools=True,
                    metadata={
                        "difficulty": row.get("difficulty"),
                        "upstream": source.id,
                        "labels_human_verified": "verified_challenges_50" in source.path
                        and "non_verified" not in source.path,
                    },
                )
                for row in rows[: source.count]
            ]
        elif source.adapter == "intercode_inline":
            rows = json.loads(raw)
            # Only explicitly selected, self-contained tasks. Never execute upstream network setup commands.
            safe_ids = {5, 17, 18, 19, 22}
            if not source.task_ids or not set(source.task_ids) <= safe_ids:
                raise ValueError(
                    "intercode_inline supports only audited self-contained IDs 5,17,18,19,22"
                )
            rows = [r for r in rows if r["task_id"] in source.task_ids]
            tasks = [
                Task(
                    f"intercode:{revision}:{r['task_id']}",
                    "/".join(r["tags"]),
                    r["query"] + "\nSubmit the complete picoCTF{...} flag.",
                    r["gold"],
                    tools=True,
                    metadata={
                        "upstream": source.id,
                        "original_task_id": r["task_id"],
                        "subset": "self-contained inline tasks",
                    },
                )
                for r in rows
            ]
        else:
            raise ValueError("Unknown GitHub adapter")
        if len(tasks) != source.count:
            raise ValueError(f"{source.adapter}: requested {source.count}, found {len(tasks)}")
        return tasks, {**source.model_dump(), "revision": revision, "payload_sha256": digest(raw)}
    if source.kind == "evalplus":
        return _evalplus_tasks(source)
    if source.kind == "procedural":
        tasks = procedural(source.adapter, source.count, source.seed, source.families)
        return tasks, source.model_dump()
    if source.kind == "hf":
        return _hf_tasks(source)
    if not source.path:
        raise ValueError("jsonl source requires path")
    path = Path(source.path)
    tasks = [Task.from_dict(row) for row in read_jsonl(path)][
        source.offset : source.offset + source.count
    ]
    if len(tasks) != source.count:
        raise ValueError("Insufficient imported tasks")
    return tasks, {**source.model_dump(), "file_sha256": file_hash(path)}


def prepare(
    config: Experiment,
    destination: Path,
    *,
    offline: bool = False,
    shared_evaluation: Path | None = None,
) -> dict:
    if (destination / "manifest.json").exists():
        raise ValueError(f"Prepared data already exists at {destination}; choose a new destination")
    frozen = config.model_copy(deep=True)
    shared_config = shared_manifest = shared_splits = None
    if shared_evaluation:
        shared_config, shared_manifest, shared_splits = load_prepared(shared_evaluation)
        if (
            shared_config.evaluation != config.evaluation
            or shared_config.model.id != config.model.id
        ):
            raise ValueError(
                "Shared evaluation must have the same model and evaluation configuration"
            )
        if shared_manifest["offline_fixture"] != offline:
            raise ValueError("Shared evaluation cannot mix fixture and real data")
    if offline:
        if any(
            s.kind not in {"procedural", "jsonl"}
            for s in [config.train, config.validation]
            + [b.source for b in config.evaluation.benchmarks]
        ):
            raise ValueError("Offline preparation requires procedural or local data")
    elif shared_config:
        frozen.model.revision = shared_config.model.revision
    else:
        from huggingface_hub import HfApi

        frozen.model.revision = (
            HfApi().model_info(config.model.id, revision=config.model.revision).sha
        )
        require_sha(frozen.model.revision, "Model revision")
    sources = {"train": config.train, "validation": config.validation}
    sources.update({"eval-" + b.name: b.source for b in config.evaluation.benchmarks})
    splits, provenance = {}, {}
    for name, source in sources.items():
        if shared_splits is not None and name.startswith("eval-"):
            splits[name] = shared_splits[name]
            provenance[name] = shared_manifest["sources"][name]
        else:
            splits[name], provenance[name] = load_source(source)
    audit = audit_splits(splits)
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, tasks in splits.items():
        # JSON arrays allow atomic creation and checksumming before a manifest commits the preparation.
        path = destination / f"{name}.json"
        write_json(path, [t.to_dict() for t in tasks])
        files[name] = {"file": path.name, "sha256": file_hash(path), "count": len(tasks)}
    manifest = {
        "schema_version": 1,
        "offline_fixture": offline,
        "config_sha256": digest(config.model_dump()),
        "files": files,
        "sources": provenance,
        "audit": audit,
    }
    write_json(destination / "config.json", frozen.model_dump())
    manifest["frozen_config_sha256"] = file_hash(destination / "config.json")
    write_json(destination / "manifest.json", manifest)
    return manifest


def _evalplus_tasks(source: DataSource) -> tuple[list[Task], dict]:
    """Freeze official base and augmented inputs and compute oracles in Docker."""
    from importlib.metadata import version

    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    from .codec import checker_program, encode, oracle_program
    from .config import SandboxConfig
    from .sandbox import DockerSandbox

    ds = get_human_eval_plus() if source.adapter == "humaneval_plus" else get_mbpp_plus()
    if source.adapter not in {"humaneval_plus", "mbpp_plus"}:
        raise ValueError("Unknown EvalPlus dataset adapter")
    sandbox = DockerSandbox(
        SandboxConfig(timeout_seconds=120, memory_mb=1024, output_bytes=1048576)
    )
    sandbox.preflight(evalplus=True)
    tasks = []
    for ident, row in sorted(ds.items())[source.offset : source.offset + source.count]:
        dataset = "humaneval" if source.adapter == "humaneval_plus" else "mbpp"
        program = oracle_program(row, dataset)
        result = sandbox.execute(program, image=sandbox.config.evalplus_image)
        if result.status != "ok":
            raise RuntimeError(f"EvalPlus oracle failed for {ident}: {result.status}")
        oracle = json.loads(result.stdout.strip().splitlines()[-1])
        checked = sandbox.execute(
            checker_program(row, oracle, dataset, row["prompt"] + row["canonical_solution"]),
            image=sandbox.config.evalplus_image,
            timeout_seconds=sandbox.config.evalplus_timeout_seconds,
            memory_mb=sandbox.config.evalplus_memory_mb,
        )
        if checked.status != "ok" or json.loads(checked.stdout.strip().splitlines()[-1]).get(
            "statuses"
        ) != ["pass", "pass"]:
            raise RuntimeError(
                f"Known-correct EvalPlus solution rejected for {ident}; preparation aborted"
            )
        tasks.append(
            Task(
                ident,
                "function_code",
                "Return a complete Python implementation:\n" + row["prompt"],
                "",
                "evalplus",
                metadata={
                    "problem": encode({k: v for k, v in row.items() if k != "canonical_solution"}),
                    "oracle": oracle,
                    "dataset": dataset,
                    "benchmark": source.adapter,
                    "oracle_comparison": "official evalplus.untrusted_check, base and augmented tests",
                },
            )
        )
    if len(tasks) != source.count:
        raise ValueError("Not enough EvalPlus tasks")
    return tasks, {
        **source.model_dump(),
        "evalplus_version": version("evalplus"),
        "snapshot_sha256": digest(encode(ds)),
    }


def load_prepared(path: Path) -> tuple[Experiment, dict, dict[str, list[Task]]]:
    manifest = read_json(path / "manifest.json")
    if file_hash(path / "config.json") != manifest["frozen_config_sha256"]:
        raise ValueError("Frozen config checksum mismatch")
    config = Experiment.model_validate(read_json(path / "config.json"))
    splits = {}
    for name, info in manifest["files"].items():
        file = path / info["file"]
        if file.parent.resolve() != path.resolve() or file_hash(file) != info["sha256"]:
            raise ValueError(f"Manifest checksum/path mismatch: {name}")
        splits[name] = [Task.from_dict(row) for row in read_json(file)]
        if len(splits[name]) != info["count"]:
            raise ValueError("Manifest count mismatch")
    audit_splits(splits)
    return config, manifest, splits
