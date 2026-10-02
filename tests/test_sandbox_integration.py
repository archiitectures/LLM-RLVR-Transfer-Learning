"""Opt-in Docker qualification; generated candidate code is executed only in Docker."""

import os

import pytest

from transferlab.codec import encode
from transferlab.config import SandboxConfig
from transferlab.sandbox import DockerSandbox
from transferlab.tasks import Task, verify

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(
        os.environ.get("TRANSFERLAB_RUN_DOCKER_TESTS") != "1",
        reason="Use transferlab initial-tests --stage sandbox on a Docker host",
    ),
]


@pytest.fixture(scope="module")
def sandbox():
    value = DockerSandbox(SandboxConfig(timeout_seconds=2))
    value.preflight(evalplus=True)
    return value


def test_python_output_and_error(sandbox):
    assert sandbox.execute("print(6*7)").stdout.strip() == "42"
    assert sandbox.execute("raise ValueError('candidate error')").status == "execution_error"


def test_readonly_filesystem_and_disposable_calls(sandbox):
    result = sandbox.execute("open('/etc/transferlab-write-test','w').write('x')")
    assert result.status == "execution_error"
    assert sandbox.execute("open('/tmp/local-state','w').write('x')").status == "ok"
    assert (
        sandbox.execute("import os; print(os.path.exists('/tmp/local-state'))").stdout.strip()
        == "False"
    )


def test_timeout_and_output_limit(sandbox):
    assert sandbox.execute("while True: pass").status == "timeout"
    assert sandbox.execute("while True: print('x'*4096,flush=True)").status == "output_limit"
    assert sandbox.execute("print('still usable')").status == "ok"


def test_no_secrets_or_network(sandbox):
    result = sandbox.execute(
        "import os,socket; print(os.getenv('VAST_API_KEY')); s=socket.socket(); s.settimeout(.2);\ntry: s.connect(('1.1.1.1',80)); print('connected')\nexcept OSError: print('blocked')"
    )
    assert result.status == "ok"
    assert result.stdout.strip().splitlines() == ["None", "blocked"]


@pytest.mark.parametrize(
    "code,correct",
    [
        ("def f(x): return x", True),
        ("def f(x): return x+1", False),
        ("def f(x):\n    while True: pass", False),
    ],
)
def test_official_evalplus_correct_wrong_and_timeout(sandbox, code, correct):
    task = Task(
        "qualification",
        "code",
        "implement identity",
        "",
        "evalplus",
        metadata={
            "problem": encode(
                {"entry_point": "f", "base_input": [[1]], "plus_input": [[2]], "atol": 0}
            ),
            "oracle": encode({"base": [1], "plus": [2], "base_time": [0.01], "plus_time": [0.01]}),
            "dataset": "humaneval",
        },
    )
    assert verify(task, code, sandbox)[0] is correct
