from __future__ import annotations

from .config import Experiment
from .io import environment
from .sandbox import DockerSandbox


def preflight(config: Experiment) -> dict:
    import inspect

    import torch
    import transformers
    import trl

    if not torch.cuda.is_available():
        raise RuntimeError("Paper runtime requires a CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("v1 expects exactly one visible GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support the BF16 recipe")
    if "environment_factory" not in inspect.signature(trl.GRPOTrainer).parameters:
        raise RuntimeError("Installed TRL lacks the locked environment API")
    import jmespath  # noqa: F401

    from .tool_parsing import configure_tool_tokenizer

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.id, revision=config.model.revision, trust_remote_code=False
    )
    configure_tool_tokenizer(tokenizer)
    parsed = tokenizer.parse_response(
        '<tool_call>{"name":"submit_answer","arguments":{"answer":"42"}}</tool_call>'
    )
    if parsed["tool_calls"][0]["function"]["arguments"]["answer"] != "42":
        raise RuntimeError("Tool response parser failed")
    sandbox = DockerSandbox(config.sandbox)
    sandbox.preflight(
        evalplus=any(b.source.kind == "evalplus" for b in config.evaluation.benchmarks)
    )
    if any(b.source.kind == "evalplus" for b in config.evaluation.benchmarks):
        result = sandbox.execute(
            "import evalplus.eval; print('evalplus-ready')", image=config.sandbox.evalplus_image
        )
        if result.status != "ok" or result.stdout.strip() != "evalplus-ready":
            raise RuntimeError("EvalPlus sandbox dependencies are not available")
    return {
        "ready": True,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "sandbox_image_ids": sandbox.image_ids,
        "environment": environment(),
    }
