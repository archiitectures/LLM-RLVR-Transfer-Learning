"""Bounded initial qualification stages; never provision or rent hardware."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from .config import load_config
from .data import load_prepared
from .io import digest, read_json, source_fingerprint, timestamp, write_json

ARMS = ("crypto", "benign_tools", "code", "math", "logic")


def initial_tests(
    stage: str,
    project: Path,
    output: Path,
    *,
    prepared: list[Path] | None = None,
    timeout: int = 7200,
    steps: int = 2,
    tasks: int = 1,
) -> dict:
    if stage not in {"local", "cpu", "sandbox", "gpu"}:
        raise ValueError("Unknown qualification stage")
    if timeout < 1 or steps < 1 or tasks < 1:
        raise ValueError("Timeout, pilot steps and pilot tasks must be positive")
    project, output = project.resolve(), output.resolve()
    if not (project / "tests").is_dir() or not (project / "configs").is_dir():
        raise ValueError("--project must point to a TransferLab checkout")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Initial-test output must be new or empty; use a different directory")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "stage": stage,
        "status": "running",
        "started": timestamp(),
        "source": source_fingerprint(project),
        "checks": [],
        "rentals_created": 0,
        "scope": "This stage only; paper-study learning and unexecuted stages are not qualified",
    }
    write_json(output / "readiness.json", result)
    started = time.monotonic()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    if stage == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""

    def command(name, argv, *, command_env=None):
        print(f"Initial tests: {name}", flush=True)
        began = time.monotonic()
        log = output / (name + ".log")
        with log.open("w") as stream:
            proc = subprocess.Popen(
                argv,
                cwd=project,
                env=command_env or env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout)
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                raise
        check = {
            "name": name,
            "seconds": time.monotonic() - began,
            "log": log.name,
            "exit_code": proc.returncode,
            "status": "passed" if proc.returncode == 0 else "failed",
        }
        result["checks"].append(check)
        write_json(output / "readiness.json", result)
        if proc.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")

    cli = [sys.executable, "-m", "transferlab.cli"]
    try:
        for name in ARMS:
            load_config(project / f"configs/{name}.yaml")
        result["checks"].append({"name": "five-arm-config-validation", "status": "passed"})
        if stage == "local":
            missing = [
                name
                for name in ("pytest", "torch", "trl", "jmespath", "datasets", "peft")
                if importlib.util.find_spec(name) is None
            ]
            if missing:
                raise RuntimeError(
                    "Install local test dependencies: uv sync --frozen --extra train --group dev; missing "
                    + ", ".join(missing)
                )
            junit = output / "tests.xml"
            command(
                "regression-tests",
                [sys.executable, "-m", "pytest", "-q", "-m", "not sandbox", f"--junitxml={junit}"],
            )
            suites = ET.parse(junit).getroot()
            skipped = sum(int(s.get("skipped", 0)) for s in suites.iter("testsuite"))
            if skipped:
                raise RuntimeError(
                    "Local qualification requires all selected training tests; skipped tests found"
                )
            result["test_counts"] = {
                key: sum(int(s.get(key, 0)) for s in suites.iter("testsuite"))
                for key in ("tests", "failures", "errors", "skipped")
            }
            command("five-arm-smoke", cli + ["smoke", "--output", str(output / "smoke")])
        elif stage == "cpu":
            data, runs = output / "prepared", output / "runs"
            command(
                "tiny-model-prepare",
                cli + ["prepare", str(project / "configs/smoke.yaml"), "--output", str(data)],
            )
            args = cli + ["pilot", str(data), "--output", str(runs), "--steps", "1", "--tasks", "1"]
            command("tiny-model-train-evaluate", args)
            before = {
                str(p.relative_to(runs)): p.read_bytes() for p in runs.rglob("predictions.jsonl")
            }
            command("tiny-model-resume", args + ["--resume"])
            after = {
                str(p.relative_to(runs)): p.read_bytes() for p in runs.rglob("predictions.jsonl")
            }
            if (
                not before
                or before != after
                or read_json(runs / "math/run.json")["status"] != "completed"
            ):
                raise RuntimeError(
                    "Tiny-model resume changed saved predictions or failed to complete"
                )
            result["checks"].append(
                {"name": "resume-without-duplicate-predictions", "status": "passed"}
            )
            result["learning_claim"] = (
                "Execution check only; learning requires informative rewards in the paper pilot"
            )
        elif stage == "sandbox":
            if not shutil.which("docker"):
                raise RuntimeError(
                    "Docker is missing. Install a working Docker engine and build containers/sandbox.Dockerfile"
                )
            command("docker-engine", ["docker", "info"])
            sandbox_env = {**env, "TRANSFERLAB_RUN_DOCKER_TESTS": "1"}
            command(
                "real-sandbox-verifiers",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-m",
                    "sandbox",
                    "tests/test_sandbox_integration.py",
                ],
                command_env=sandbox_env,
            )
        else:
            paths = [p.resolve() for p in prepared or []]
            configs = [load_prepared(p)[0] for p in paths]
            if len(configs) != 5 or {c.arm for c in configs} != set(ARMS):
                raise ValueError(
                    "GPU qualification requires --prepared with exactly the five prepared paper arms"
                )
            for config, path in zip(configs, paths, strict=True):
                if load_prepared(path)[1]["offline_fixture"]:
                    raise ValueError("GPU paper qualification requires real prepared datasets")
                command("preflight-" + config.arm, cli + ["preflight", str(path / "config.json")])
            command(
                "five-arm-gpu-pilot",
                cli
                + [
                    "pilot",
                    *map(str, paths),
                    "--output",
                    str(output / "pilot"),
                    "--steps",
                    str(steps),
                    "--tasks",
                    str(tasks),
                ],
            )
            result["learning_claim"] = (
                "Run pilot-budget on a sufficiently sampled pilot before approving paper compute"
            )
        result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as exc:
        result.update(
            status="blocked"
            if isinstance(exc, (ImportError, FileNotFoundError)) or "missing" in str(exc).lower()
            else "failed",
            error=str(exc),
        )
    finally:
        result.update(finished=timestamp(), seconds=time.monotonic() - started)
        write_json(output / "readiness.json", result)
    return result


def prepare_suite(project: Path, destination: Path) -> dict:
    from .data import prepare

    configs = [load_config(project / f"configs/{arm}.yaml") for arm in ARMS]
    prepared = []
    for config in configs:
        path = destination / config.arm
        if (path / "manifest.json").exists():
            _, manifest, _ = load_prepared(path)
            if manifest["config_sha256"] != digest(config.model_dump()):
                raise ValueError(
                    f"Existing prepared {config.arm} differs from the requested config"
                )
        else:
            prepare(config, path, shared_evaluation=destination / "crypto" if prepared else None)
        prepared.append(path)
    frozen = [load_prepared(p) for p in prepared]
    hashes = [
        {k: v["sha256"] for k, v in m["files"].items() if k.startswith("eval-")}
        for _, m, _ in frozen
    ]
    if any(h != hashes[0] for h in hashes[1:]) or any(
        c.model != frozen[0][0].model for c, _, _ in frozen[1:]
    ):
        raise ValueError("Prepared suite snapshots differ; use a new output directory")
    return {"prepared": [str(p) for p in prepared], "arms": len(prepared)}
