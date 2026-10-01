# TransferLab

Controlled LoRA/GRPO fine-tuning experiments measuring transfer across a predefined conceptual-distance taxonomy. Five training arms share the same model, recipe, evaluation tasks, and estimated model-compute budget:

* cryptographic CTFs;
* benign tool-assisted data transformation/debugging;
* ordinary code generation;
* math;
* Knights-and-Knaves logic.

The default paper model is `Qwen/Qwen2.5-7B-Instruct`. Every arm starts independently from the same model revision. No training, evaluation, or rental is automatically launched by installation.

## Quick start: free local verification

```sh
uv sync --group dev
uv run transferlab validate configs/crypto.yaml
uv run transferlab smoke --output runs/smoke-suite
uv run --group dev pytest -q
```

`smoke` exercises preparation, all five arm interfaces, baseline/final evaluation, saved predictions, and reporting without downloading models, executing generated code, or renting GPUs. Outputs are conspicuously marked **fixtures**. Scientific reports reject them unless `--allow-fixtures` is explicit.

For a real small-model execution check:

```sh
uv sync --extra train --group dev
uv run transferlab prepare configs/smoke.yaml --output data/tiny-real
OMP_NUM_THREADS=1 uv run --extra train transferlab pilot data/tiny-real \
  --output runs/tiny-real --steps 1 --tasks 1
```

This uses SmolLM2-135M in FP32 on CPU when CUDA is unavailable. It proves the training/saving/loading path, not useful learning. The offline tensor integration test separately proves a nonzero LoRA update and zero loss gradients on observation tokens using artificial test rewards. Paper pilots must show real within-group reward variation and nonzero gradients.

## Prepare the paper suite

Install the training/benchmark extras on the Linux GPU runtime or preparation machine:

```sh
uv sync --extra train --extra benchmarks --group dev
docker pull python:3.11-slim
docker build -f containers/sandbox.Dockerfile -t transferlab-sandbox:0.1.0 .
```

Docker is required for generated Python, tool calls, and EvalPlus oracle computation. There is no host-execution fallback. Tool containers have no network, secrets, repository mounts, checkpoint mounts, capabilities, or Docker socket. Each call is disposable; Python variables/files do not persist between calls. A bounded eight-call episode supports iterative computation through returned observations.

```sh
uv run --extra benchmarks transferlab prepare configs/crypto.yaml --output data/study/crypto
uv run --extra benchmarks transferlab prepare configs/benign_tools.yaml \
  --output data/study/benign_tools --shared-evaluation data/study/crypto
uv run --extra benchmarks transferlab prepare configs/code.yaml \
  --output data/study/code --shared-evaluation data/study/crypto
uv run --extra benchmarks transferlab prepare configs/math.yaml \
  --output data/study/math --shared-evaluation data/study/crypto
uv run --extra benchmarks transferlab prepare configs/logic.yaml \
  --output data/study/logic --shared-evaluation data/study/crypto
```

Preparation resolves model/HF/GitHub references to commit SHAs, freezes tasks and official EvalPlus base/augmented inputs, computes typed oracles in Docker, and rejects normalized prompt overlaps across train/validation/evaluation. It never executes upstream generators, challenge setup commands, or reference solutions on the controller. Existing preparations are not overwritten. `--shared-evaluation` reuses exactly the same test snapshot and base-model revision; suites reject differing evaluation hashes.

The Random-Crypto training corpus is published as **not human verified**. Its metadata retains that fact. Inspect labels and pilot learnability before paper runs. The supplied benign arm matches the tool interface and episode limits; its task difficulty is not established as matched to Random-Crypto. Pilot diagnostics and task curation are needed before making a content-specific causal claim.

Default evaluation includes verified Random-Crypto, an explicitly limited five-task **self-contained InterCode subset**, benign tools, 32 HumanEval+ tasks, 32 MBPP+ tasks, 128 MATH500 problems, 64 logic puzzles, and 64 ARC-Challenge questions. These are frozen subsets, not full-benchmark scores. InterCode's live services, asset-heavy challenges, and privileged full environments are not enabled. Broader security environments can be added through task adapters; these crypto/encoding measurements alone do not establish real-world cyber risk.

`configs/zebra.yaml` demonstrates the ZebraLogic adapter. Public releases may contain redacted solutions: preparation rejects them rather than scoring against placeholders. Supply an authorized solved release to use it.

## Pilot, freeze, and run

On the intended CUDA GPU class:

```sh
uv run --extra train --extra benchmarks transferlab pilot \
  data/study/crypto data/study/benign_tools data/study/code data/study/math data/study/logic \
  --output runs/pilot --steps 8 --tasks 4

uv run transferlab pilot-budget \
  data/study/crypto data/study/benign_tools data/study/code data/study/math data/study/logic \
  --pilots runs/pilot --output data/study/budget.json

uv run --extra train --extra benchmarks transferlab suite \
  data/study/crypto data/study/benign_tools data/study/code data/study/math data/study/logic \
  --budget data/study/budget.json --output runs/paper

uv run --extra plots transferlab report runs/paper --output reports/paper --plots
```

The pilot-budget command fails if data are fixtures, a GPU pilot is incomplete, rewards lack within-group variation, gradients remain zero, or projected evaluation exceeds its allocation. It chooses a common FLOP budget using the slowest measured arm, configured hourly ceilings, runtime limits, and 25% training headroom. Evaluation projections include 50% headroom but are estimates from a small sample. An infeasible pilot requires an explicit revised configuration and another pilot; the code does not silently drop arms/seeds or change the model.

The $1,000 allocation is $100 pilot, $550 training, $250 evaluation, and $100 contingency. The default suite has three seeds per arm. Hyperparameters and data are frozen before paper runs; use validation outcomes for feasibility/curation, not held-out test scores to select the best checkpoint.

Single runs support `--seed`, `--baseline-only`, and `--resume`. Resume requires identical code, installed environment, resolved config, model, and data. Training resumes from a full Trainer checkpoint including optimizer/RNG state and compute accounting. Completed prediction samples are skipped. A final saved adapter can be evaluated separately:

```sh
uv run --extra train --extra benchmarks transferlab evaluate data/study/crypto \
  --checkpoint runs/paper/crypto/seed-11/train/final --output runs/extra-eval
```

Additional EleutherAI likelihood benchmarks are deliberately reported separately from generated-answer pass@k:

```sh
uv run --extra benchmarks transferlab harness data/study/crypto \
  --tasks arc_challenge mmlu --output reports/harness-baseline
```

## Vast.ai overnight execution

The runner uses the authenticated Vast REST API, SSH, and rsync. Store `VAST_API_KEY` in the controller environment; register an SSH key through Vast's normal account setup. Credentials are not uploaded. Set `vast.ssh_key` if your key is not in the SSH agent.

Provide `vast.image` and `vast.image_digest: sha256:...` for a GPU image verified to have Python/uv, SSH, rsync, GNU timeout, compatible NVIDIA drivers, and a functioning Docker sandbox. The lock currently selects a CUDA 13 PyTorch build on Linux: check actual CUDA/driver compatibility before a paid pilot. Standard GPU containers do not necessarily permit a nested Docker daemon; use a compatible Vast VM/runtime. Image selection is intentionally not an unverified mutable default.

Run `uv run --extra train transferlab preflight configs/crypto.yaml` on that runtime before the pilot. Offer selection requires advertised CUDA compatibility of at least 13.0; preflight verifies actual CUDA/BF16 availability and imports the official checker inside the sandbox.

```sh
uv run transferlab vast offers configs/crypto.yaml
uv run transferlab vast run configs/crypto.yaml \
  --data data/study --output runs/remote-pilot --category pilot --dry-run -- \
  uv run --frozen --extra train --extra benchmarks transferlab pilot \
  /workspace/data/crypto /workspace/data/benign_tools /workspace/data/code \
  /workspace/data/math /workspace/data/logic --output /workspace/results --steps 8 --tasks 4
```

Remove `--dry-run` to launch. Swap `pilot` for `suite`, pass `/workspace/data/budget.json`, and choose `--category training` for the paper suite. The entire remote command, including dependency setup, is bounded by the watchdog. If a complete suite cannot fit one rental's runtime cap, run individual arm/seed jobs using the same `run` command; the budget ledger spans launches.

The runner filters price, VRAM, reliability, storage, SSH ports, and on-demand availability; reserves estimated rental/storage/transfer exposure before creating; persists a job label **before** creation; reconciles lost creation responses; streams logs; collects artifacts every five minutes and after failure; and destroys/verifies the instance. Any unverified cleanup retains recovery state, exits nonzero, and blocks new rentals. Worst-case reservations remain charged in the local ledger rather than being released using approximate billing.

```sh
uv run transferlab vast status
uv run transferlab vast collect configs/crypto.yaml
uv run transferlab vast cleanup
```

Keep the controller online overnight. The remote timeout survives SSH disconnection and stops the process, but cannot stop Vast billing on its own. Provider/network failure during destruction can continue billing; the local budget is a conservative launch guard, not a provider-enforced billing cap. No account key is placed in task or training containers.

## Outputs and interpretation

Prepared data: `config.json`, `manifest.json`, and checksummed task splits. Runs: `run.json`, reward/episode traces, `compute.jsonl`, full checkpoints, final adapters, and baseline/validation/milestone predictions. Reports: `transfer.json`, `transfer.csv`, `report.md`, and optional `distance-transfer.png`.

Compute is an **architecture-based estimate**, including generation/reference forwards, padded tokens, attention, LoRA backward work, and estimated checkpoint recomputation. It is not profiler-measured FLOPs. Actual GPU memory, runtime, Trainer token metrics, unique task counts, rollout counts, and milestone overshoot are saved independently. Comparisons reaching a milestone can overshoot by one optimizer step; inspect the residual mismatch before making an equal-compute claim.

Evaluation uses the same per-benchmark prompts, sample seeds, temperature, token-span cap, and tool limits for baseline and trained models. Stable labels `budget-25`, `budget-50`, and `final` permit comparisons across seeds. Reports provide per-benchmark paired task bootstrap intervals and separate seed variability; distance-tier averages weight benchmarks equally, not by question count. The example distance labels are proposed study annotations to review and freeze before paper runs. Conceptual distance is ordinal, and task-family counts—not thousands of correlated questions—limit evidence for a distance effect.

## Sources and licenses

* [TRL GRPO](https://huggingface.co/docs/trl/en/grpo_trainer): shared training implementation, pinned by `uv.lock`.
* [HackSynth-GRPO / Random-Crypto](https://github.com/aielte-research/HackSynth-GRPO): published cryptographic task data and recipe reference, upstream AGPL-3.0; retain its attribution/license when distributing derived data.
* [InterCode](https://github.com/princeton-nlp/intercode): the five explicitly named inline evaluation tasks, upstream attribution retained in task metadata.
* [EvalPlus](https://github.com/evalplus/evalplus): official augmented inputs and checker, including special task contracts; MIT.
* [Can One Domain Help Others?](https://arxiv.org/abs/2507.17512): prior math/code/logic transfer study informing controls, not a claim of novelty for reproducing the matrix.

No checkpoints or data are automatically published to Hugging Face. Generated artifacts, credentials, caches, and model weights stay outside source control.
