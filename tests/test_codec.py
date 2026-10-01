import math

import pytest

from transferlab.codec import checker_program, decode, encode, oracle_program


def test_typed_oracle_roundtrip():
    original = {"a": (1, 2), "b": {1, 2}, 3: [None, True, "text"], "inf": float("inf")}
    assert decode(encode(original)) == original
    assert math.isnan(decode(encode(float("nan"))))


def test_codec_does_not_evaluate_untrusted_strings():
    value = "__import__('os').system('false')"
    assert decode(encode(value)) == value
    with pytest.raises(ValueError):
        decode({"type": "executable", "items": []})


def test_official_checker_uses_safe_literals_and_no_reference():
    problem = {
        "canonical_solution": "SECRET_REFERENCE",
        "entry_point": "f",
        "base_input": [(1,)],
        "plus_input": [(2,)],
        "atol": 0,
        "prompt": "def f(x):",
    }
    candidate = "def f(x):\n    return x\n# $(echo secret) `echo secret`"
    program = checker_program(
        problem,
        encode({"base": [1], "plus": [2], "base_time": [0.1], "plus_time": [0.1]}),
        "humaneval",
        candidate,
    )
    assert "SECRET_REFERENCE" not in program
    assert "untrusted_check" in program
    compile(program, "checker", "exec")
    compile(oracle_program(problem, "humaneval"), "oracle", "exec")
