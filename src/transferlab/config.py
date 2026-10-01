from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Model(Strict):
    id: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    dtype: Literal["bfloat16", "float32"] = "bfloat16"


class Recipe(Strict):
    learning_rate: float = Field(1e-5, gt=0)
    lora_rank: int = Field(32, gt=0)
    lora_alpha: int = Field(64, gt=0)
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    group_size: int = Field(4, ge=2)
    batch_size: int = Field(4, ge=1)
    gradient_accumulation: int = Field(1, ge=1)
    max_steps: int = Field(10000, ge=1)
    max_prompt_tokens: int = Field(4096, ge=64)
    max_completion_tokens: int = Field(2048, ge=8)
    max_context_tokens: int = Field(8192, ge=128)
    beta: float = Field(0.01, ge=0)
    temperature: float = Field(0.8, gt=0)
    loss_type: str = "grpo"
    compute_flops: float | None = Field(None, gt=0)
    max_tool_calls: int = Field(8, ge=1, le=32)
    episode_seconds: int = Field(120, ge=1)
    milestones: list[float] = [0.25, 0.5, 1.0]

    @model_validator(mode="after")
    def invariants(self):
        if self.batch_size * self.gradient_accumulation % self.group_size:
            raise ValueError("Effective generation batch must be divisible by group_size")
        if self.max_prompt_tokens + self.max_completion_tokens > self.max_context_tokens:
            raise ValueError("Prompt + completion exceeds context limit")
        if self.milestones != sorted(set(self.milestones)) or self.milestones[-1:] != [1.0]:
            raise ValueError("Milestones must be sorted, unique, end at 1.0")
        if any(not 0 < x <= 1 for x in self.milestones):
            raise ValueError("Invalid milestone")
        return self


class DataSource(Strict):
    kind: Literal["procedural", "hf", "jsonl", "github", "evalplus"]
    adapter: str
    count: int = Field(256, ge=1)
    seed: int = 1729
    families: list[str] = []
    id: str | None = None
    revision: str | None = None
    subset: str | None = None
    split: str = "train"
    path: str | None = None
    offset: int = Field(0, ge=0)
    task_ids: list[int] = []


class Benchmark(Strict):
    name: str
    source: DataSource
    distance: dict[str, int]
    skills: list[str] = []


class Evaluation(Strict):
    samples: int = Field(4, ge=4)
    temperature: float = Field(0.7, gt=0)
    top_p: float = Field(0.95, gt=0, le=1)
    max_new_tokens: int = Field(2048, ge=8)
    seed: int = 3407
    benchmarks: list[Benchmark]


class SandboxConfig(Strict):
    image: str = "python:3.11-slim"
    evalplus_image: str = "transferlab-sandbox:0.1.0"
    evalplus_timeout_seconds: int = Field(180, ge=130, le=600)
    evalplus_memory_mb: int = Field(1024, ge=512)
    timeout_seconds: int = Field(10, ge=1, le=120)
    memory_mb: int = Field(256, ge=32)
    output_bytes: int = Field(16384, ge=128, le=1048576)


class VastConfig(Strict):
    max_rate: float = Field(2.0, gt=0)
    max_cost: float = Field(30.0, gt=0)
    max_hours: float = Field(12.0, gt=0)
    min_vram_gb: int = Field(48, ge=1)
    min_reliability: float = Field(0.98, gt=0, le=1)
    disk_gb: int = Field(100, ge=20)
    image: str | None = None
    image_digest: str | None = None
    readiness_seconds: int = Field(900, ge=1)
    collection_seconds: int = Field(300, ge=1)
    collection_timeout_seconds: int = Field(900, ge=1)
    final_collection_seconds: int = Field(900, ge=1)
    ssh_key: str | None = None
    controller_required: bool = True
    min_cuda_version: float = Field(13.0, ge=11.0)


class Study(Strict):
    total_usd: float = Field(1000, gt=0)
    pilot_usd: float = Field(100, gt=0)
    training_usd: float = Field(550, gt=0)
    evaluation_usd: float = Field(250, gt=0)
    contingency_usd: float = Field(100, ge=0)
    seeds: list[int] = [11, 22, 33]

    @model_validator(mode="after")
    def allocation(self):
        if (
            sum((self.pilot_usd, self.training_usd, self.evaluation_usd, self.contingency_usd))
            > self.total_usd + 1e-8
        ):
            raise ValueError("Budget allocations exceed total ceiling")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds must be nonempty and unique")
        return self


class Experiment(Strict):
    name: str
    arm: Literal["crypto", "benign_tools", "code", "math", "logic"]
    seed: int = 11
    model: Model = Model()
    recipe: Recipe = Recipe()
    train: DataSource
    validation: DataSource
    evaluation: Evaluation
    sandbox: SandboxConfig = SandboxConfig()
    vast: VastConfig = VastConfig()
    study: Study = Study()

    @model_validator(mode="after")
    def labels(self):
        if not self.name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in self.name
        ):
            raise ValueError("name must be a safe directory name")
        names = [b.name for b in self.evaluation.benchmarks]
        if len(names) != len(set(names)):
            raise ValueError("Benchmark names must be unique")
        for b in self.evaluation.benchmarks:
            if self.arm not in b.distance or any(not 0 <= d <= 4 for d in b.distance.values()):
                raise ValueError(f"{b.name}: every evaluated arm needs an ordinal distance in 0..4")
        return self


def _merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        result[k] = (
            _merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
        )
    return result


def load_config(path: Path, stack: tuple[Path, ...] = ()) -> Experiment:
    path = path.resolve()
    if path in stack:
        raise ValueError("Circular config inheritance")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping")
    if parent := raw.pop("extends", None):
        base = _read_raw(path.parent / parent, stack + (path,))
        raw = _merge(base, raw)
    return Experiment.model_validate(raw)


def _read_raw(path: Path, stack: tuple[Path, ...]) -> dict:
    path = path.resolve()
    if path in stack:
        raise ValueError("Circular config inheritance")
    raw = yaml.safe_load(path.read_text())
    if parent := raw.pop("extends", None):
        raw = _merge(_read_raw(path.parent / parent, stack + (path,)), raw)
    return raw


def require_sha(revision: str | None, label: str) -> None:
    if (
        revision is None
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError(f"{label} must be frozen to a 40-character commit SHA; run prepare")


def finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0
