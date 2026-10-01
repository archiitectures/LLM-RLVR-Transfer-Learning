from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".atomic-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_fingerprint(root: Path) -> str:
    files = []
    for folder in (root / "src", root / "configs"):
        if folder.exists():
            for p in sorted(folder.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml"}:
                    files.append((str(p.relative_to(root)), file_hash(p)))
    for name in ("pyproject.toml", "uv.lock"):
        if (root / name).exists():
            files.append((name, file_hash(root / name)))
    return digest(files)


def environment() -> dict:
    packages = sorted((d.metadata["Name"], d.version) for d in distributions())
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    try:
        result["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        result["gpu"] = None
    return result
