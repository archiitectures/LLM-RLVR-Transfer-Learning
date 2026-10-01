from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Experiment, require_sha
from .episodes import SYSTEM, TOOL_SYSTEM, ToolEpisode
from .io import digest, file_hash
from .tasks import Task
from .tool_parsing import parse_tool_response


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    observation_tokens: int
    truncated: bool
    tool_calls: int
    seconds: float
    events: list
    status: str = "ok"


class FixtureBackend:
    """Deterministic plumbing fixture, never a model or a scientific result."""

    name = "fixture"

    def generate(self, task: Task, seed: int, **kwargs) -> Generation:
        rng = random.Random(seed)
        answer = task.answer if rng.random() < 0.6 else "WRONG"
        if task.verifier in {"stdio", "assertions"}:
            answer = "print('WRONG')"
        return Generation(
            f"<answer>{answer}</answer>",
            len(task.prompt.split()),
            len(answer.split()),
            0,
            False,
            0,
            0.0,
            [],
        )


class HFBackend:
    name = "hf"

    def __init__(self, config: Experiment, sandbox=None, checkpoint=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        require_sha(config.model.revision, "Model")
        self.config = config
        self.sandbox = sandbox
        self.fingerprint = digest(
            {
                "model": config.model.model_dump(),
                "adapter": file_hash(Path(checkpoint) / "adapter_model.safetensors")
                if checkpoint
                else None,
            }
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.id, revision=config.model.revision
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.id,
            revision=config.model.revision,
            dtype=getattr(torch, config.model.dtype),
            device_map={"": "cuda:0" if torch.cuda.is_available() else "cpu"},
            trust_remote_code=False,
        )
        if checkpoint:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(checkpoint))
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, task: Task, seed: int, **kwargs) -> Generation:
        import torch
        from transformers import set_seed
        from transformers.utils import get_json_schema

        set_seed(seed)
        started = time.monotonic()
        cfg = self.config
        messages = [
            {"role": "system", "content": TOOL_SYSTEM if task.tools else SYSTEM},
            {"role": "user", "content": task.prompt},
        ]
        episode = ToolEpisode(self.sandbox, cfg.recipe) if task.tools else None
        if episode:
            episode.reset(json.dumps(task.to_dict()))
        tools = (
            [get_json_schema(episode.execute_python), get_json_schema(episode.submit_answer)]
            if episode
            else None
        )
        total_completion, observations, prompt_tokens = 0, 0, 0
        text = ""
        truncated = False
        status = "ok"
        while True:
            if time.monotonic() - started > cfg.recipe.episode_seconds:
                status = "episode_timeout"
                break
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            length = encoded["input_ids"].shape[-1]
            if not prompt_tokens:
                prompt_tokens = length
            # Match TRL's episode cap, including observations and chat delimiters.
            remaining = min(
                cfg.evaluation.max_new_tokens - (length - prompt_tokens),
                cfg.recipe.max_context_tokens - length,
            )
            if remaining <= 0 or length > cfg.recipe.max_context_tokens:
                truncated = True
                break
            encoded = encoded.to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=remaining,
                    do_sample=True,
                    temperature=cfg.evaluation.temperature,
                    top_p=cfg.evaluation.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            ids = output[0, length:]
            total_completion += len(ids)
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            truncated |= len(ids) >= remaining
            try:
                response = parse_tool_response(text) if episode else {}
            except (ValueError, KeyError, TypeError):
                status = "invalid_tool_call"
                break
            parsed = response.get("tool_calls", [])
            if not parsed:
                break
            messages.append(response)
            for call in parsed:
                function = call["function"]
                try:
                    result = getattr(episode, function["name"])(**function["arguments"])
                except TypeError:
                    result = "invalid_tool_arguments"
                observations += len(self.tokenizer.encode(result, add_special_tokens=False))
                messages.append({"role": "tool", "name": function["name"], "content": result})
            if episode._answer is not None or episode._calls >= cfg.recipe.max_tool_calls:
                break
        if episode and episode._answer is not None:
            text = f"<answer>{episode._answer}</answer>"
        elif episode:
            # Tool arms require the same explicit submission protocol in training and evaluation.
            text = ""
            status = "no_submission" if status == "ok" else status
        return Generation(
            text,
            prompt_tokens,
            total_completion,
            observations,
            truncated,
            episode._calls if episode else 0,
            time.monotonic() - started,
            episode._events if episode else [],
            status,
        )
