from __future__ import annotations

import json
import time
from pathlib import Path

from .checkpoints import require_checkpoint_space, retain_checkpoints, seal_checkpoint
from .config import Experiment, require_sha
from .episodes import SYSTEM, TOOL_SYSTEM, ToolEpisode
from .io import append_jsonl, digest, read_json, read_jsonl, write_json
from .tasks import Task, verify


class ComputeMeter:
    """Architecture-based model-FLOP estimate, not measured hardware FLOPs.

    Includes padded tokens, autoregressive prefill/decode, reference forwards,
    attention, and an input-gradient estimate for frozen LoRA base weights.
    Checkpoint recomputation is accounted by a declared multiplier. GPU profiling
    in the pilot provides an independent throughput/feasibility measurement.
    """

    def __init__(
        self, parameters: int, trainable: int, layers: int, hidden: int, checkpointing: bool = True
    ):
        self.parameters = parameters
        self.trainable = trainable
        self.layers = layers
        self.hidden = hidden
        self.checkpointing = checkpointing
        self.flops = 0.0
        self.forward_tokens = 0
        self.update_tokens = 0
        self.forward_calls = 0

    def charge(self, batch: int, query: int, context: int, gradients: bool) -> float:
        if min(batch, query, context) <= 0:
            raise ValueError("Invalid compute dimensions")
        forward = (
            2 * self.parameters * batch * query
            + 4 * self.layers * self.hidden * batch * query * context
        )
        # Frozen base weights still require activation gradients. Adapter weight gradients
        # cost extra; recomputation requires one additional forward when checkpointing.
        amount = forward
        if gradients:
            amount += forward + 2 * self.trainable * batch * query
            if self.checkpointing:
                amount += forward
            self.update_tokens += batch * query
        self.flops += amount
        self.forward_tokens += batch * query
        self.forward_calls += 1
        return amount

    def state(self) -> dict:
        return {
            "flops": self.flops,
            "forward_tokens": self.forward_tokens,
            "update_tokens": self.update_tokens,
            "forward_calls": self.forward_calls,
            "estimator": "architecture_lora_v1",
            "parameters": self.parameters,
            "trainable": self.trainable,
            "layers": self.layers,
            "hidden": self.hidden,
            "checkpointing_recompute_estimate": self.checkpointing,
        }

    def restore(self, state: dict) -> None:
        for field in ("flops", "forward_tokens", "update_tokens", "forward_calls"):
            setattr(self, field, state[field])


def train(
    config: Experiment,
    tasks: list[Task],
    output: Path,
    sandbox=None,
    resume: Path | None = None,
    pilot: bool = False,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer

    require_sha(config.model.revision, "Model revision")
    if config.recipe.compute_flops is None and not pilot:
        raise ValueError("Paper runs require a frozen compute_flops budget from the pilot")
    if config.model.dtype == "bfloat16" and not torch.cuda.is_available():
        raise ValueError("BF16 study training requires CUDA; choose float32 for tiny CPU checks")
    if config.model.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Selected GPU does not support BF16")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model.id, revision=config.model.revision)
    if any(t.tools for t in tasks):
        from .tool_parsing import configure_tool_tokenizer

        configure_tool_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        dtype=getattr(torch, config.model.dtype),
        trust_remote_code=False,
    )
    if tokenizer.model_max_length < config.recipe.max_context_tokens:
        raise ValueError("Requested context exceeds tokenizer maximum")
    tokenizer.model_max_length = config.recipe.max_context_tokens
    cfg = config.recipe
    rows = [
        {
            "prompt": [
                {"role": "system", "content": TOOL_SYSTEM if t.tools else SYSTEM},
                {"role": "user", "content": t.prompt},
            ],
            "task_json": json.dumps(t.to_dict()),
        }
        for t in tasks
    ]
    if any(t.tools for t in tasks) and not all(t.tools for t in tasks):
        raise ValueError("Each arm must have a consistent tool interface")
    from transformers.utils import get_json_schema

    template_tools = None
    if tasks[0].tools:
        episode = ToolEpisode(sandbox, cfg)
        template_tools = [
            get_json_schema(episode.execute_python),
            get_json_schema(episode.submit_answer),
        ]
    for row in rows:
        size = len(
            tokenizer.apply_chat_template(
                row["prompt"], tools=template_tools, add_generation_prompt=True
            )
        )
        if size > cfg.max_prompt_tokens:
            raise ValueError(
                "Task prompt exceeds the fixed prompt budget; curate data before training"
            )
    started = time.monotonic()
    reward_log = output / "rewards.jsonl"

    def correctness(completions, task_json, **kwargs):
        rewards = []
        for completion, raw in zip(completions, task_json, strict=True):
            text = completion[-1]["content"] if isinstance(completion, list) else completion
            task = Task.from_dict(json.loads(raw))
            correct, reason = verify(task, text or "", sandbox)
            rewards.append(float(correct))
            append_jsonl(
                reward_log,
                {"task_id": task.id, "completion": text, "correct": correct, "reason": reason},
            )
        return rewards

    training_args = GRPOConfig(
        output_dir=str(output),
        learning_rate=cfg.learning_rate,
        num_generations=cfg.group_size,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation,
        max_steps=cfg.max_steps,
        max_completion_length=cfg.max_completion_tokens,
        temperature=cfg.temperature,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        max_tool_calling_iterations=cfg.max_tool_calls,
        bf16=config.model.dtype == "bfloat16",
        gradient_checkpointing=True,
        seed=config.seed,
        data_seed=config.seed,
        save_strategy="steps",
        save_steps=10 if pilot else 50,
        save_only_model=False,
        save_total_limit=None,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
        mask_truncated_completions=True,
        use_vllm=False,
        use_cpu=not torch.cuda.is_available(),
        optim="adamw_torch",
    )
    arguments = {
        "model": model,
        "args": training_args,
        "train_dataset": Dataset.from_list(rows),
        "processing_class": tokenizer,
        "peft_config": LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    }
    if tasks[0].tools:
        if sandbox is None:
            raise ValueError("Tool training requires Docker sandbox")
        arguments["environment_factory"] = lambda: ToolEpisode(sandbox, cfg, reward_log)

        def episode_correctness(completions, environments, **kwargs):
            return [environment._reward() for environment in environments]

        arguments["reward_funcs"] = episode_correctness
    else:
        arguments["reward_funcs"] = correctness
    trainer = GRPOTrainer(**arguments)
    root_model = trainer.model.get_base_model()
    meter = ComputeMeter(
        sum(p.numel() for p in trainer.model.parameters()),
        sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        model.config.num_hidden_layers,
        model.config.hidden_size,
    )
    if resume:
        meter.restore(read_json(resume / "compute.json"))
    require_checkpoint_space(output, meter.trainable)

    def forward_hook(module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            raise RuntimeError("Compute meter cannot identify model inputs")
        batch, query = ids.shape[:2]
        context = query
        past = kwargs.get("past_key_values")
        if past is not None:
            context += (
                past.get_seq_length() if hasattr(past, "get_seq_length") else past[0][0].shape[-2]
            )
        if context > cfg.max_context_tokens:
            raise RuntimeError("Training rollout exceeded the frozen context budget")
        meter.charge(batch, query, context, torch.is_grad_enabled() and module.training)

    handle = root_model.register_forward_pre_hook(forward_hook, with_kwargs=True)
    milestones = (
        read_json(resume / "milestones.json")
        if resume and (resume / "milestones.json").exists()
        else {}
    )

    class BudgetCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if cfg.compute_flops:
                fraction = meter.flops / cfg.compute_flops
                for milestone in cfg.milestones:
                    if fraction >= milestone and str(milestone) not in milestones:
                        milestones[str(milestone)] = {
                            "checkpoint": f"checkpoint-{state.global_step}",
                            "flops": meter.flops,
                            "fraction": fraction,
                        }
                        control.should_save = True
                if fraction >= 1:
                    control.should_training_stop = True
            append_jsonl(
                output / "compute.jsonl",
                {"step": state.global_step, "seconds": time.monotonic() - started, **meter.state()},
            )
            if control.should_save:
                require_checkpoint_space(output, meter.trainable)
            return control

        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / f"checkpoint-{state.global_step}"
            write_json(checkpoint / "compute.json", meter.state())
            write_json(checkpoint / "milestones.json", milestones)
            seal_checkpoint(checkpoint)
            retain_checkpoints(output, milestones)

    trainer.add_callback(BudgetCallback())
    try:
        if resume and cfg.compute_flops and meter.flops >= cfg.compute_flops:
            # A crash after the budget checkpoint must not purchase another update.
            from transformers import TrainerState

            trainer._load_from_checkpoint(str(resume))
            trainer.state = TrainerState.load_from_json(str(resume / "trainer_state.json"))
        else:
            trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if not pilot and meter.flops < cfg.compute_flops:
            raise RuntimeError(
                "max_steps reached before common compute budget; this run is incomplete"
            )
        trainer.save_model(str(output / "final"))
        tokenizer.save_pretrained(str(output / "final"))
        from collections import Counter

        exposures = Counter(row["task_id"] for row in read_jsonl(reward_log))
        result = {
            "fixture": False,
            "seconds": time.monotonic() - started,
            "milestones": milestones,
            "compute": meter.state(),
            "budget_flops": cfg.compute_flops,
            "overshoot_flops": max(0, meter.flops - (cfg.compute_flops or meter.flops)),
            "template_sha256": digest(tokenizer.chat_template),
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else None,
            "nonzero_gradient_updates": sum(
                float(log.get("grad_norm", 0)) > 0 for log in trainer.state.log_history
            ),
            "trainer_metrics": trainer.state.log_history,
            "data_exposure": {
                "unique_tasks": len(exposures),
                "rollouts": sum(exposures.values()),
                "rollouts_per_task": dict(exposures),
            },
        }
        write_json(output / "training.json", result)
        retain_checkpoints(output, milestones, finished=True)
        return result
    finally:
        handle.remove()
        del trainer, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
