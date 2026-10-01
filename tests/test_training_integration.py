"""Offline real-tensor proof. Artificial rewards here are test fixtures, not experiments."""

import pytest


def test_lora_update_and_observation_loss_mask(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from peft import LoraConfig
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from trl import GRPOConfig, GRPOTrainer

    torch.set_num_threads(1)
    torch.manual_seed(11)
    raw = Tokenizer(
        WordLevel(
            {"<pad>": 0, "<unk>": 1, "<eos>": 2, "a": 3, "b": 4, "observation": 5, "task": 6},
            unk_token="<unk>",
        )
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=32,
            eos_token_id=2,
            pad_token_id=0,
        )
    )

    def rollout(prompts, trainer):
        return {
            "prompt_ids": [[6] for _ in prompts],
            "completion_ids": [[3 + i % 2, 5, 2] for i, _ in enumerate(prompts)],
            "logprobs": None,
            "env_mask": [[1, 0, 1] for _ in prompts],
        }

    def reward(completions, **kwargs):
        return [float(i % 2) for i, _ in enumerate(completions)]

    captured = []

    class ObservedTrainer(GRPOTrainer):
        def _get_per_token_logps_and_entropies(self, *args, **kwargs):
            result = super()._get_per_token_logps_and_entropies(*args, **kwargs)
            if result[0].requires_grad:
                result[0].retain_grad()
                captured.append(result[0])
            return result

    trainer = ObservedTrainer(
        model=model,
        reward_funcs=reward,
        rollout_func=rollout,
        train_dataset=Dataset.from_dict({"prompt": ["task"] * 4}),
        processing_class=tokenizer,
        peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], task_type="CAUSAL_LM"),
        args=GRPOConfig(
            output_dir=str(tmp_path),
            max_steps=1,
            per_device_train_batch_size=4,
            num_generations=4,
            max_completion_length=4,
            beta=0,
            loss_type="grpo",
            learning_rate=0.01,
            use_cpu=True,
            bf16=False,
            gradient_checkpointing=False,
            report_to="none",
            save_strategy="no",
            disable_tqdm=True,
            logging_steps=1,
        ),
    )
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    trainer.train()
    assert any(
        not torch.equal(before[n], p)
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    )
    assert captured and captured[-1].grad is not None
    assert torch.count_nonzero(captured[-1].grad[:, 1]) == 0
    assert torch.count_nonzero(captured[-1].grad[:, 0]) > 0
    # Budget-complete recovery loads the saved adapter without another optimizer step.
    checkpoint = tmp_path / "recovery-checkpoint"
    trainer.save_model(str(checkpoint))
    saved = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in trainer.model.parameters():
            if p.requires_grad:
                p.add_(1)
    trainer._load_from_checkpoint(str(checkpoint))
    assert all(
        torch.equal(saved[n], p) for n, p in trainer.model.named_parameters() if p.requires_grad
    )
