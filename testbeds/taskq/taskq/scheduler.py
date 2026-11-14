"""Priority scheduling over the task store."""

from __future__ import annotations

from .utils import chunked, now_ms


class Scheduler:
    """Selects the next batch of tasks to run."""

    def __init__(self, store, batch_size=8):
        self._store = store
        self._batch_size = batch_size

    def next_batch(self):
        """Return up to batch_size pending tasks."""
        pending = self._store.pending()
        return pending[: self._batch_size]

    def iter_batches(self):
        """Yield pending tasks in fixed-size batches."""
        yield from chunked(self._store.pending(), self._batch_size)

    def requeue_failed(self):
        """Move failed tasks back to pending for another attempt."""
        requeued = []
        for task in self._store.failed():
            self._store.set_status(task.task_id, "pending")
            requeued.append(task.task_id)
        return requeued

    def stats(self):
        """Scheduling statistics."""
        return {"pending": len(self._store.pending()), "ts": now_ms()}
