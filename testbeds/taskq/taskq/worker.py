"""Worker that executes scheduled tasks."""

from __future__ import annotations

from .models import TaskResult
from .utils import now_ms, retry


class Worker:
    """Runs tasks from the scheduler, retrying transient failures."""

    def __init__(self, store, scheduler, registry, max_retries=3):
        self._store = store
        self._scheduler = scheduler
        self._registry = registry
        self._max_retries = max_retries

    def run_once(self, task):
        """Execute a single task with retry and store its result."""
        self._store.set_status(task.task_id, "running")
        fn = self._registry.get(task.name)
        if fn is None:
            result = TaskResult(task.task_id, ok=False, error=f"unknown task {task.name!r}")
            return self._store.save_result(result)
        try:
            value = retry(lambda: fn(**task.args), attempts=self._max_retries)
            result = TaskResult(task.task_id, ok=True, value=value)
        except Exception as exc:  # noqa: BLE001 - report any failure as a result
            result = TaskResult(task.task_id, ok=False, error=str(exc))
        return self._store.save_result(result)

    def run_batch(self):
        """Run the next scheduled batch; return the number executed."""
        batch = self._scheduler.next_batch()
        for task in batch:
            self.run_once(task)
        return len(batch)

    def heartbeat(self):
        """A lightweight liveness signal."""
        return {"worker": id(self), "ts": now_ms()}
