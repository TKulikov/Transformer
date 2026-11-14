"""In-memory task store with persistence-friendly accessors."""

from __future__ import annotations

from .errors import TaskNotFound
from .models import TaskResult
from .utils import now_ms


class TaskStore:
    """Thread-agnostic in-memory store of tasks and their results."""

    def __init__(self):
        self._tasks = {}
        self._results = {}

    def put(self, task):
        """Insert or replace a task."""
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id):
        """Return a task or raise TaskNotFound."""
        if task_id not in self._tasks:
            raise TaskNotFound(task_id)
        return self._tasks[task_id]

    def set_status(self, task_id, status):
        """Update a task's status and bump its attempt counter."""
        task = self.get(task_id)
        task.status = status
        if status == "running":
            task.attempts += 1
        return task

    def save_result(self, result):
        """Store a TaskResult and mark the task finished."""
        self._results[result.task_id] = result
        self.set_status(result.task_id, "done" if result.ok else "failed")
        return result

    def result(self, task_id):
        """Return the stored result or None."""
        return self._results.get(task_id)

    def pending(self):
        """All tasks still waiting to run, oldest first."""
        items = [t for t in self._tasks.values() if t.status == "pending"]
        return sorted(items, key=lambda t: t.created_ms)

    def failed(self):
        """All tasks whose last execution failed."""
        return [t for t in self._tasks.values() if t.status == "failed"]

    def snapshot(self):
        """A serializable view of the store for debugging."""
        return {"tasks": len(self._tasks), "results": len(self._results), "ts": now_ms()}
