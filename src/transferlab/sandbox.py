from __future__ import annotations

import json
import selectors
import subprocess
import time
import uuid
from dataclasses import dataclass

from .config import SandboxConfig


@dataclass
class Execution:
    stdout: str
    stderr: str
    status: str
    seconds: float


class FixtureSandbox:
    fixture = True

    def execute(self, code, stdin="", **kwargs):
        return Execution("", "Fixture: code was not executed", "fixture_not_executed", 0.0)


class DockerSandbox:
    """Disposable unprivileged execution; never mount the repository or Docker socket."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.image_ids = {}

    def preflight(self, *, evalplus: bool = False) -> None:
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=20)
        for image in [self.config.image] + ([self.config.evalplus_image] if evalplus else []):
            resolved = subprocess.check_output(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True, timeout=20
            ).strip()
            if not resolved.startswith("sha256:"):
                raise RuntimeError("Could not resolve sandbox image to an immutable local ID")
            self.image_ids[image] = resolved
        result = self.execute("print('sandbox-ready')")
        if result.status != "ok" or result.stdout.strip() != "sandbox-ready":
            raise RuntimeError(f"Sandbox preflight failed: {result.status}: {result.stderr}")
        if evalplus:
            from .codec import checker_program, encode

            problem = {"entry_point": "f", "base_input": [[1]], "plus_input": [[2]], "atol": 0}
            oracle = encode({"base": [1], "plus": [2], "base_time": [0.01], "plus_time": [0.01]})
            result = self.execute(
                checker_program(problem, oracle, "humaneval", "def f(x): return x"),
                image=self.config.evalplus_image,
                timeout_seconds=self.config.evalplus_timeout_seconds,
                memory_mb=self.config.evalplus_memory_mb,
            )
            try:
                valid = result.status == "ok" and json.loads(
                    result.stdout.strip().splitlines()[-1]
                )["statuses"] == ["pass", "pass"]
            except (ValueError, IndexError, KeyError):
                valid = False
            if not valid:
                raise RuntimeError(
                    f"EvalPlus known-correct sandbox check failed: {result.status}: {result.stderr}"
                )

    def execute(
        self,
        code: str,
        stdin: str = "",
        *,
        image: str | None = None,
        timeout_seconds: int | None = None,
        memory_mb: int | None = None,
    ) -> Execution:
        timeout_seconds = timeout_seconds or self.config.timeout_seconds
        memory_mb = memory_mb or self.config.memory_mb
        name = "transferlab-" + uuid.uuid4().hex
        launcher = "import sys,json,io; p=json.load(sys.stdin); sys.stdin=io.StringIO(p['stdin']); exec(compile(p['code'],'candidate.py','exec'),{'__name__':'__main__'})"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            f"{memory_mb}m",
            "--memory-swap",
            f"{memory_mb}m",
            "--cpus",
            "1",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "-i",
            self.image_ids.get(image or self.config.image, image or self.config.image),
            "python",
            "-I",
            "-c",
            launcher,
        ]
        started = time.monotonic()
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = bytearray(), bytearray()
        status = "ok"
        try:
            # Bound submitted source too; pipe input remains small enough for a prompt-sized program.
            payload = json.dumps({"code": code, "stdin": stdin}).encode()
            if len(payload) > 1024 * 1024:
                raise ValueError("Sandbox input exceeds 1 MiB")
            import threading

            def feed():
                try:
                    process.stdin.write(payload)
                    process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            writer = threading.Thread(target=feed, daemon=True)
            writer.start()
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, out)
                selector.register(process.stderr, selectors.EVENT_READ, err)
                while selector.get_map():
                    if time.monotonic() - started >= timeout_seconds:
                        status = "timeout"
                        break
                    for key, _ in selector.select(timeout=0.1):
                        block = key.fileobj.read1(4096)
                        if not block:
                            selector.unregister(key.fileobj)
                        else:
                            key.data.extend(block)
                    if len(out) + len(err) > self.config.output_bytes:
                        status = "output_limit"
                        break
            if status == "ok" and process.wait(timeout=2) != 0:
                status = "execution_error"
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
            if cleanup.returncode and b"No such container" not in cleanup.stderr:
                raise RuntimeError(
                    f"Sandbox cleanup unverified for {name}: "
                    + cleanup.stderr.decode(errors="replace")
                )
        return Execution(
            out[: self.config.output_bytes].decode(errors="replace"),
            err[: self.config.output_bytes].decode(errors="replace"),
            status,
            time.monotonic() - started,
        )
