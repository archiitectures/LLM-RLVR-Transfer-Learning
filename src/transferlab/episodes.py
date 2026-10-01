from __future__ import annotations

import json
import time
from pathlib import Path

from .config import Recipe
from .io import append_jsonl
from .tasks import Task, verify

SYSTEM = "Solve the task carefully. Put the final answer in <answer>...</answer>. For programming tasks put complete Python code inside those tags."
TOOL_SYSTEM = (
    SYSTEM
    + " You may use execute_python for computation. Submit your final answer using submit_answer."
)


class ToolEpisode:
    """Only reset and the two documented tools are public for TRL discovery."""

    def __init__(self, sandbox, recipe: Recipe, trace: Path | None = None):
        self._sandbox = sandbox
        self._recipe = recipe
        self._trace = trace
        self._task = None
        self._answer = None
        self._calls = 0
        self._started = 0.0
        self._events = []

    def reset(self, task_json: str, **kwargs) -> None:
        self._task = Task.from_dict(json.loads(task_json))
        self._answer = None
        self._calls = 0
        self._started = time.monotonic()
        self._events = []

    def _allowed(self) -> bool:
        return (
            self._answer is None
            and self._calls < self._recipe.max_tool_calls
            and time.monotonic() - self._started < self._recipe.episode_seconds
        )

    def execute_python(self, code: str) -> str:
        """Execute Python in an isolated, network-disabled, disposable environment.

        Args:
            code: Complete Python source; print outputs to inspect them.

        Returns:
            A JSON object containing bounded stdout, stderr, and execution status.
        """
        if not self._allowed():
            return '{"status":"episode_limit"}'
        self._calls += 1
        result = self._sandbox.execute(code)
        event = {"tool": "execute_python", "code": code, **vars(result)}
        self._events.append(event)
        return json.dumps(
            {"stdout": result.stdout, "stderr": result.stderr, "status": result.status}
        )

    def submit_answer(self, answer: str) -> str:
        """Submit the final answer once; correctness feedback is not exposed.

        Args:
            answer: Final answer to the task, without explanation.

        Returns:
            Confirmation that the answer was recorded, or an episode limit message.
        """
        if not self._allowed():
            return "episode_limit"
        self._calls += 1
        self._answer = answer
        self._events.append({"tool": "submit_answer", "answer": answer})
        return "answer_recorded"

    def _reward(self) -> float:
        correct, reason = verify(self._task, self._answer or "", self._sandbox)
        if self._trace:
            append_jsonl(
                self._trace,
                {
                    "task_id": self._task.id,
                    "events": self._events,
                    "answer": self._answer,
                    "correct": correct,
                    "reason": reason,
                    "seconds": time.monotonic() - self._started,
                },
            )
        return float(correct)
