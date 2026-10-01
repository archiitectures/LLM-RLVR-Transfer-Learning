"""Commit, verify and retain only owned Trainer checkpoint directories."""

import shutil
from pathlib import Path

from .io import file_hash, read_json, write_json

MARKER = "complete.json"
REQUIRED = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "compute.json",
    "milestones.json",
}


def checkpoint_paths(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()),
        key=lambda p: int(p.name[11:]),
    )


def seal_checkpoint(path: Path) -> None:
    names = {p.name for p in path.iterdir() if p.is_file()}
    if not REQUIRED <= names:
        raise RuntimeError(f"Incomplete checkpoint {path}: missing {sorted(REQUIRED - names)}")
    write_json(
        path / MARKER,
        {
            "files": {
                p.name: file_hash(p)
                for p in sorted(path.iterdir())
                if p.is_file() and p.name != MARKER
            }
        },
    )


def verified_checkpoint(path: Path) -> bool:
    try:
        files = read_json(path / MARKER)["files"]
        return REQUIRED <= files.keys() and all(
            Path(name).name == name and file_hash(path / name) == checksum
            for name, checksum in files.items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_checkpoint(root: Path) -> Path | None:
    return next((p for p in reversed(checkpoint_paths(root)) if verified_checkpoint(p)), None)


def retain_checkpoints(root: Path, milestones: dict, keep_recent: int = 2, *, finished=False):
    paths = checkpoint_paths(root)
    complete = [p for p in paths if verified_checkpoint(p)]
    recent = {p.name for p in complete[-keep_recent:]} if not finished else set()
    protected = {v["checkpoint"] for v in milestones.values()}
    for path in paths:
        if path.name in recent:
            continue
        if path.name in protected:
            # Milestones need adapters for evaluation, not multi-GB optimizer states.
            # Retire the resume marker first so interrupted pruning is never resumable.
            (path / MARKER).unlink(missing_ok=True)
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                (path / name).unlink(missing_ok=True)
        else:
            # Only numeric checkpoint children of this specific training output are owned here.
            if path.is_symlink():
                raise ValueError("Refusing to prune a symlink checkpoint")
            shutil.rmtree(path)


def require_checkpoint_space(root: Path, trainable_parameters: int):
    # FP32 adapter + two Adam moments, with room for serialization and metadata.
    required = max(2 * 1024**3, trainable_parameters * 16)
    if shutil.disk_usage(root).free < required:
        raise RuntimeError(f"Insufficient checkpoint disk space: need {required} free bytes")
