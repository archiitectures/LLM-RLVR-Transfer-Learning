"""Typed JSON codec for benchmark inputs/oracles; no pickle or executable decoding."""

from __future__ import annotations

import math


def encode(value):
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"type": "float", "value": str(value)}
    if isinstance(value, list):
        return {"type": "list", "items": [encode(x) for x in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [encode(x) for x in value]}
    if isinstance(value, set):
        return {"type": "set", "items": [encode(x) for x in sorted(value, key=repr)]}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    if type(value).__module__.startswith("numpy"):
        if hasattr(value, "shape") and len(value.shape) > 0:
            return {"type": "ndarray", "items": encode(value.tolist()), "dtype": str(value.dtype)}
        return encode(value.item())
    raise TypeError(f"Unsupported benchmark value: {type(value).__name__}")


def decode(value):
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "float":
        return float(value["value"])
    if kind == "dict":
        return {decode(k): decode(v) for k, v in value["items"]}
    if kind in {"list", "tuple", "set"}:
        items = [decode(x) for x in value["items"]]
        return items if kind == "list" else tuple(items) if kind == "tuple" else set(items)
    if kind == "ndarray":
        import numpy as np

        return np.array(decode(value["items"]), dtype=value["dtype"])
    raise ValueError("Unknown benchmark codec type")


PRELUDE = (
    "import sys,json\nsys.path.insert(0,'/opt/transferlab')\nfrom codec import encode,decode\n"
)


def oracle_program(problem: dict, dataset: str) -> str:
    payload = encode(problem)
    return (
        PRELUDE + "from evalplus.gen.util import trusted_exec\n"
        "from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS\n"
        f"p=decode(json.loads({__import__('json').dumps(payload)!r}))\n"
        f"not_none={dataset!r}=='mbpp' and p['entry_point'] in MBPP_OUTPUT_NOT_NONE_TASKS\n"
        "result={}\n"
        "for section in ('base','plus'):\n"
        "    outputs,times=trusted_exec(p['prompt']+p['canonical_solution'],p[section+'_input'],p['entry_point'],record_time=True,output_not_none=not_none)\n"
        "    result[section]=outputs\n    result[section+'_time']=times\n"
        "print(json.dumps(encode(result),allow_nan=False))\n"
    )


def checker_program(problem: dict, oracle: dict, dataset: str, candidate: str) -> str:
    import json

    # No reference implementation is transferred into the candidate checker.
    problem = {k: v for k, v in problem.items() if k != "canonical_solution"}
    return (
        PRELUDE + "from evalplus.eval import untrusted_check\n"
        f"p=decode(json.loads({json.dumps(encode(problem))!r}))\n"
        f"oracle=decode(json.loads({json.dumps(oracle)!r}))\n"
        f"candidate={candidate!r}\nstatuses=[]\n"
        "for section in ('base','plus'):\n"
        f"    status,_=untrusted_check({dataset!r},candidate,p[section+'_input'],p['entry_point'],expected=oracle[section],atol=p['atol'],ref_time=oracle[section+'_time'],fast_check=True)\n"
        "    statuses.append(status)\n"
        "print(json.dumps({'statuses':statuses}))\n"
    )
