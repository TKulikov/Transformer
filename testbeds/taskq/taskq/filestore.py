"""A file-backed task store (second backend)."""

from __future__ import annotations

import json
import os

from .models import Task
from .storage import TaskStore
from .utils import chunked


class FileStore(TaskStore):
    """Persists tasks to a JSON-lines file in addition to memory."""

    def __init__(self, path):
        super().__init__()
        self._path = path
        self._load()

    def _load(self):
        if not os.path.exists(self._path):
            return
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                data = json.loads(line)
                self._tasks[data["task_id"]] = Task(**data)

    def put(self, task):
        """Insert a task and append it to the backing file."""
        super().put(task)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(task.__dict__) + "\n")
        return task

    def flush_batch(self, tasks):
        """Append a batch of tasks efficiently."""
        with open(self._path, "a", encoding="utf-8") as fh:
            for batch in chunked(tasks, 64):
                fh.writelines(json.dumps(t.__dict__) + "\n" for t in batch)
