from __future__ import annotations

import ast
import itertools
import json
import random
import re
import string
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any


@dataclass
class Task:
    id: str
    family: str
    prompt: str
    answer: str
    verifier: str = "exact"
    tools: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> Task:
        return cls(**row)


def answer_text(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if matches:
        return matches[-1].strip()
    boxed = extract_boxed(text)
    return boxed if boxed is not None else text.strip()


def extract_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += 7
    depth = 1
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start:i].strip()
    return None


def _numeric(text: str) -> Fraction:
    """Evaluate only small arithmetic expressions, never model-generated Python."""
    if len(text) > 128:
        raise ValueError("Oversized expression")
    node = ast.parse(text, mode="eval").body

    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return (-1 if isinstance(n.op, ast.USub) else 1) * visit(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
        raise ValueError("Not a numeric expression")

    return visit(node)


def verify(task: Task, completion: str, sandbox=None) -> tuple[bool, str]:
    answer = answer_text(completion)
    if task.verifier == "exact":
        return answer == task.answer.strip(), "exact"
    if task.verifier == "exact_json":
        try:
            return json.loads(answer) == json.loads(task.answer), "json"
        except (ValueError, TypeError):
            return False, "unparseable_json"
    if task.verifier == "numeric":
        try:
            return _numeric(answer) == _numeric(task.answer), "numeric"
        except (ValueError, SyntaxError, ArithmeticError, RecursionError):
            return False, "unparseable"
    if task.verifier == "math":
        try:
            from math_verify import parse
            from math_verify import verify as math_verify
        except ImportError as exc:
            raise RuntimeError(
                "Install the train or benchmarks extra for symbolic math verification"
            ) from exc
        # The external verifier parses mathematical notation, not Python eval.
        return bool(
            math_verify(parse("\\boxed{" + task.answer + "}"), parse(completion))
        ), "math_verify"
    if task.verifier == "evalplus":
        from .codec import checker_program, decode

        if sandbox is None:
            raise RuntimeError("EvalPlus verification requires a sandbox")
        program = checker_program(
            decode(task.metadata["problem"]),
            task.metadata["oracle"],
            task.metadata["dataset"],
            extract_code(answer),
        )
        result = sandbox.execute(program, image=sandbox.config.evalplus_image)
        if result.status != "ok":
            return False, result.status
        try:
            statuses = json.loads(result.stdout.strip().splitlines()[-1])["statuses"]
            return statuses == ["pass", "pass"], "evalplus:" + ",".join(statuses)
        except (ValueError, IndexError, KeyError, TypeError):
            return False, "unparseable_evalplus_result"
    if task.verifier == "function_cases":
        if sandbox is None:
            raise RuntimeError("Function verification requires a sandbox")
        code = extract_code(answer)
        cases = task.metadata["inputs"]
        program = (
            code
            + "\nimport json\n_cases="
            + repr(cases)
            + "\nprint(json.dumps(["
            + task.metadata["entry_point"]
            + "(*args) for args in _cases]))"
        )
        result = sandbox.execute(program)
        if result.status != "ok":
            return False, result.status
        try:
            actual = json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return False, "unparseable_output"

        def equal(a, b):
            if type(a) in (int, float) and type(b) in (int, float):
                import math

                return math.isclose(a, b, rel_tol=1e-7, abs_tol=task.metadata.get("atol", 0))
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b, strict=True))
            return a == b

        return equal(actual, task.metadata["outputs"]), "function_cases"
    if task.verifier in {"stdio", "assertions"}:
        if sandbox is None:
            raise RuntimeError("Code verification requires a sandbox")
        code = extract_code(answer)
        if task.verifier == "stdio":
            for case in task.metadata["tests"]:
                result = sandbox.execute(code, case["input"])
                if result.status != "ok":
                    return False, result.status
                expected = case["output"]

                def normalize(s):
                    return "\n".join(line.rstrip() for line in s.strip().splitlines())

                if normalize(result.stdout) != normalize(expected):
                    return False, "wrong_output"
            return True, "passed"
        # A normal exit alone never counts as successful assertion execution.
        marker = "TRANSFERLAB_TESTS_COMPLETED"
        program = code + "\n" + task.metadata["test_code"] + f"\nprint({marker!r})\n"
        result = sandbox.execute(program)
        return result.status == "ok" and result.stdout.strip().endswith(marker), result.status
    raise ValueError(f"Unknown verifier: {task.verifier}")


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return blocks[-1].strip() if blocks else text.strip()


def procedural(adapter: str, count: int, seed: int, families: list[str]) -> list[Task]:
    rng = random.Random(seed)
    tasks = []
    for index in range(count):
        family = rng.choice(families) if families else adapter
        ident = f"{adapter}:{seed}:{index}"
        if adapter == "crypto":
            flag = "flag{" + "".join(rng.choices(string.ascii_lowercase, k=24)) + "}"
            if family == "caesar":
                shift = rng.randrange(1, 26)
                ciphertext = "".join(
                    chr((ord(c) - 97 + shift) % 26 + 97) if c.islower() else c for c in flag
                )
                prompt = f"Recover the flag from this Caesar-cipher CTF. Ciphertext: {ciphertext}. The plaintext starts with flag{{."
            elif family == "xor":
                key = rng.randrange(1, 256)
                ciphertext = bytes(b ^ key for b in flag.encode()).hex()
                prompt = f"Recover the flag from a single-byte-XOR CTF. Hex ciphertext: {ciphertext}. The plaintext starts with flag{{."
            elif family == "reverse":
                prompt = f"Recover the flag from this reverse-encoding CTF: {flag[::-1]}"
            else:
                raise ValueError(f"Unsupported crypto family: {family}")
            tasks.append(Task(ident, family, prompt, flag, tools=True))
        elif adapter == "benign_tools":
            values = rng.sample(range(-10000, 10000), 24)
            if family in {"benign_tools", "transform"}:
                answer = json.dumps(sorted(v * 3 + 7 for v in values), separators=(",", ":"))
                prompt = f"Transform each number x to 3*x+7, sort ascending, and return a compact JSON array: {values}"
            elif family == "debug":
                answer = str(sum(v for v in values if v % 2 == 0))
                prompt = f"Debug this requirement: sum only the even numbers in {values}. Return the corrected sum, not code."
            else:
                raise ValueError(f"Unsupported benign family: {family}")
            tasks.append(Task(ident, family, prompt, answer, tools=True))
        elif adapter == "math":
            a, b, c = [rng.randint(-1000, 1000) for _ in range(3)]
            tasks.append(
                Task(ident, "arithmetic", f"Compute ({a} * {b}) + {c}.", str(a * b + c), "numeric")
            )
        elif adapter == "logic":
            # Knights tell the truth, knaves lie. Generate and retain only uniquely solvable systems.
            for _ in range(1000):
                n = 6
                statements = [
                    (
                        rng.randrange(n),
                        bool(rng.getrandbits(1)),
                        rng.randrange(n),
                        bool(rng.getrandbits(1)),
                    )
                    for _ in range(n)
                ]
                solutions = []
                for assignment in itertools.product((False, True), repeat=n):
                    truths = [
                        ((assignment[a] == av) and (assignment[b] == bv))
                        for a, av, b, bv in statements
                    ]
                    if list(assignment) == truths:
                        solutions.append(assignment)
                if len(solutions) == 1:
                    break
            else:
                raise RuntimeError("Could not generate unique logic puzzle")
            text = "Knights always tell the truth and knaves always lie. "
            for i, (a, av, b, bv) in enumerate(statements):
                text += f"Person {i} says: 'Person {a} is a {'knight' if av else 'knave'} AND person {b} is a {'knight' if bv else 'knave'}.' "
            text += "Return a comma-separated list of knight/knave labels in person order, without spaces."
            answer = ",".join("knight" if x else "knave" for x in solutions[0])
            tasks.append(Task(ident, "knights_knaves", text, answer))
        elif adapter == "code":
            a = rng.randint(2, 9999)
            prompt = f"Write a Python program reading an integer from stdin and printing that integer multiplied by {a}."
            tests = [
                {"input": str(x) + "\n", "output": str(x * a) + "\n"}
                for x in rng.sample(range(-1000, 1000), 5)
            ]
            tasks.append(Task(ident, "stdio", prompt, "", "stdio", metadata={"tests": tests}))
        else:
            raise ValueError(f"Unsupported procedural adapter: {adapter}")
    return tasks
