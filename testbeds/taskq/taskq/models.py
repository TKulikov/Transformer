"""Domain models: tasks and their results."""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import now_ms


@dataclass
class Task:
    """A unit of work submitted to the queue."""

    task_id: str
    name: str
    args: dict = field(default_factory=dict)
    status: str = "pending"
    created_ms: int = field(default_factory=now_ms)
    attempts: int = 0


@dataclass
class TaskResult:
    """The outcome of executing a task."""

    task_id: str
    ok: bool
    value: object = None
    error: str | None = None
    finished_ms: int = field(default_factory=now_ms)


def new_task(task_id, name, **args):
    """Construct a pending Task with the given arguments."""
    return Task(task_id=task_id, name=name, args=args)
