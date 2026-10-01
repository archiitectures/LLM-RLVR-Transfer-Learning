from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

import httpx

from .backends import FixtureBackend, HFBackend
from .config import load_config
from .data import load_prepared, prepare
from .evaluation import evaluate
from .io import digest, read_json
from .pilot import freeze_budget
from .report import report
from .runner import run
from .sandbox import DockerSandbox, FixtureSandbox
from .vast import VastAPI, cleanup, recover_results, run_remote, select_offers


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="transferlab")
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate an experiment YAML")
    validate.add_argument("config", type=Path)
    flight = sub.add_parser("preflight", help="Verify CUDA, BF16, TRL API, and sandbox images")
    flight.add_argument("config", type=Path)
    prep = sub.add_parser("prepare", help="Freeze model/data revisions and audit splits")
    prep.add_argument("config", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--offline-fixture", action="store_true")
    prep.add_argument("--shared-evaluation", type=Path)
    smoke = sub.add_parser(
        "smoke", help="Exercise all five arms without downloads, Docker, or training"
    )
    smoke.add_argument("--output", type=Path, required=True)
    for name in ("run", "pilot", "suite"):
        cmd = sub.add_parser(name)
        cmd.add_argument("prepared", type=Path, nargs="+" if name in {"suite", "pilot"} else None)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--fixture", action="store_true")
        cmd.add_argument("--resume", action="store_true")
        cmd.add_argument(
            "--budget", type=Path, help="Frozen common compute budget from pilot-budget"
        )
        cmd.add_argument("--baseline-only", action="store_true")
        if name == "run":
            cmd.add_argument("--seed", type=int)
        if name == "pilot":
            cmd.add_argument("--steps", type=int, default=8)
            cmd.add_argument("--tasks", type=int, default=4)
    budget = sub.add_parser("pilot-budget", help="Derive a conservative common model-FLOP budget")
    budget.add_argument("prepared", type=Path, nargs="+")
    budget.add_argument("--pilots", type=Path, required=True)
    budget.add_argument("--output", type=Path, required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("prepared", type=Path)
    ev.add_argument("--checkpoint", type=Path)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--fixture", action="store_true")
    rep = sub.add_parser("report")
    rep.add_argument("runs", type=Path)
    rep.add_argument("--output", type=Path, required=True)
    rep.add_argument("--allow-fixtures", action="store_true")
    rep.add_argument("--plots", action="store_true")
    harness = sub.add_parser("harness", help="Supplementary EleutherAI likelihood benchmarks")
    harness.add_argument("prepared", type=Path)
    harness.add_argument("--tasks", nargs="+", default=["arc_challenge"])
    harness.add_argument("--checkpoint", type=Path)
    harness.add_argument("--limit", type=int)
    harness.add_argument("--output", type=Path, required=True)
    vast = sub.add_parser("vast")
    vs = vast.add_subparsers(dest="vast_command", required=True)
    for name in ("offers", "run"):
        cmd = vs.add_parser(name)
        cmd.add_argument("config", type=Path)
        if name == "run":
            cmd.add_argument("--project", type=Path, default=Path.cwd())
            cmd.add_argument("--data", type=Path, required=True)
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument(
                "--restore",
                type=Path,
                help="Restore a collected results directory onto the new rental",
            )
            cmd.add_argument("--state", type=Path, default=Path(".transferlab"))
            cmd.add_argument(
                "--category",
                choices=["pilot", "training", "evaluation", "contingency"],
                default="pilot",
            )
            cmd.add_argument("--dry-run", action="store_true")
    for name in ("status", "cleanup", "collect"):
        cmd = vs.add_parser(name)
        cmd.add_argument("--state", type=Path, default=Path(".transferlab"))
        if name == "collect":
            cmd.add_argument("config", type=Path)
        if name == "cleanup":
            cmd.add_argument("--discard-uncollected", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    remote_command = []
    if "--" in raw:
        index = raw.index("--")
        remote_command = raw[index + 1 :]
        raw = raw[:index]
    args = parser().parse_args(raw)

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "validate":
            config = load_config(args.config)
            result = {
                "valid": True,
                "name": config.name,
                "arm": config.arm,
                "model": config.model.id,
                "prepared_required": config.model.revision is None,
            }
        elif args.command == "preflight":
            from .preflight import preflight

            result = preflight(load_config(args.config))
        elif args.command == "prepare":
            result = prepare(
                load_config(args.config),
                args.output,
                offline=args.offline_fixture,
                shared_evaluation=args.shared_evaluation,
            )
        elif args.command == "smoke":
            from .config import DataSource

            template = load_config(Path(__file__).resolve().parents[2] / "configs/smoke.yaml")
            for arm in ("crypto", "benign_tools", "code", "math", "logic"):
                cfg = template.model_copy(deep=True)
                cfg.name = cfg.arm = arm
                families = ["caesar", "xor"] if arm == "crypto" else []
                cfg.train = DataSource(
                    kind="procedural", adapter=arm, count=8, seed=123, families=families
                )
                cfg.validation = DataSource(
                    kind="procedural", adapter=arm, count=4, seed=124, families=families
                )
                data = args.output / "prepared" / arm
                prepare(cfg, data, offline=True)
                run(data, args.output / "runs" / arm, fixture=True)
            result = report(args.output / "runs", args.output / "report", allow_fixtures=True)
            result = {
                "fixture": True,
                "arms": 5,
                "comparisons": len(result["aggregate"]),
                "output": str(args.output),
            }
        elif args.command in {"run", "suite", "pilot"}:
            paths = args.prepared if isinstance(args.prepared, list) else [args.prepared]
            budget_document = read_json(args.budget) if args.budget else None
            budget = budget_document["compute_flops"] if budget_document else None
            if budget_document and not budget_document.get("evaluation_cost_check_passed"):
                raise ValueError("Budget has not passed pilot evaluation-cost checks")
            results = []
            if args.command == "suite":
                configs = [load_prepared(p)[0] for p in paths]

                # A suite cannot inadvertently mix models, shared recipes, or evaluation tasks.
                def shared(c):
                    return (
                        c.model.model_dump(),
                        {k: v for k, v in c.recipe.model_dump().items() if k != "compute_flops"},
                        c.evaluation.model_dump(),
                        c.study.model_dump(),
                    )

                if any(shared(c) != shared(configs[0]) for c in configs[1:]):
                    raise ValueError(
                        "Suite arms must share model, recipe, evaluation suite, and study settings"
                    )
                manifests = [load_prepared(p)[1] for p in paths]

                def eval_hashes(m):
                    return {k: v["sha256"] for k, v in m["files"].items() if k.startswith("eval-")}

                if any(eval_hashes(m) != eval_hashes(manifests[0]) for m in manifests[1:]):
                    raise ValueError(
                        "Suite evaluation manifests differ; prepare with --shared-evaluation"
                    )
            for path in paths:
                cfg, manifest, _ = load_prepared(path)
                if budget_document and budget_document["prepared_manifest_sha256"].get(
                    cfg.arm
                ) != digest(manifest):
                    raise ValueError("Budget was frozen for a different prepared dataset")
                selected_seed = getattr(args, "seed", None)
                seeds = (
                    cfg.study.seeds
                    if args.command == "suite"
                    else [cfg.seed if selected_seed is None else selected_seed]
                )
                for seed in seeds:
                    out = args.output
                    if args.command == "suite":
                        out = out / cfg.arm / f"seed-{seed}"
                    elif args.command == "pilot":
                        out = out / cfg.arm
                    results.append(
                        run(
                            path,
                            out,
                            seed=seed,
                            fixture=args.fixture,
                            resume=args.resume,
                            baseline_only=args.baseline_only,
                            pilot=args.command == "pilot",
                            compute_flops=budget,
                            pilot_steps=getattr(args, "steps", 8),
                            pilot_tasks=getattr(args, "tasks", 4),
                        )
                    )
            result = {
                "runs": [{"status": r["status"], "fingerprint": r["fingerprint"]} for r in results]
            }
        elif args.command == "pilot-budget":
            result = freeze_budget(args.prepared, args.pilots, args.output)
        elif args.command == "evaluate":
            cfg, manifest, splits = load_prepared(args.prepared)
            if manifest["offline_fixture"] and not args.fixture:
                raise ValueError("Fixture manifests cannot evaluate real models")
            tasks = [task for values in splits.values() for task in values]
            sandbox = FixtureSandbox() if args.fixture else None
            if not args.fixture and any(
                t.tools or t.verifier in {"stdio", "assertions", "function_cases", "evalplus"}
                for t in tasks
            ):
                sandbox = DockerSandbox(cfg.sandbox)
                sandbox.preflight(evalplus=any(t.verifier == "evalplus" for t in tasks))
            backend = FixtureBackend() if args.fixture else HFBackend(cfg, sandbox, args.checkpoint)
            result = evaluate(
                cfg, splits, backend, args.output, str(args.checkpoint or "baseline"), sandbox
            )
        elif args.command == "report":
            result = report(
                args.runs, args.output, allow_fixtures=args.allow_fixtures, plots=args.plots
            )
            result = {
                "fixture": result["fixture"],
                "comparisons": len(result["aggregate"]),
                "output": str(args.output),
            }
        elif args.command == "harness":
            from .harness import evaluate_harness

            cfg, manifest, _ = load_prepared(args.prepared)
            if manifest["offline_fixture"]:
                raise ValueError("Fixture manifests cannot run benchmark harness")
            result = evaluate_harness(cfg, args.tasks, args.output, args.checkpoint, args.limit)
        elif args.command == "vast":
            if args.vast_command == "status":
                path = args.state / "active.json"
                result = read_json(path) if path.exists() else {"status": "no_tracked_rental"}
            else:
                api = VastAPI()
                if args.vast_command == "cleanup":
                    result = cleanup(api, args.state, discard_uncollected=args.discard_uncollected)
                elif args.vast_command == "collect":
                    result = recover_results(api, load_config(args.config).vast, args.state)
                else:
                    cfg = load_config(args.config)
                    if args.vast_command == "offers":
                        from dataclasses import asdict

                        result = {
                            "candidates": [
                                asdict(o) for o in select_offers(api.offers(cfg.vast), cfg.vast)
                            ]
                        }
                    else:
                        command = remote_command
                        if command[:1] == ["--"]:
                            command = command[1:]
                        result = run_remote(
                            api,
                            cfg.vast,
                            project=args.project,
                            data=args.data,
                            output=args.output,
                            state_dir=args.state,
                            command=command,
                            ceiling=cfg.study.total_usd,
                            allocation=getattr(cfg.study, args.category + "_usd"),
                            category=args.category,
                            dry_run=args.dry_run,
                            restore=args.restore,
                        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        print("Interrupted; inspect persisted run/rental state before resuming.", file=sys.stderr)
        return 130
    except ImportError as exc:
        print(
            f"transferlab: missing optional dependency {exc.name}; install the train/benchmarks extras",
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as exc:
        print(f"transferlab: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
