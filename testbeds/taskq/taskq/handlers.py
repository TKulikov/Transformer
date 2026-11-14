"""Request handlers for the server router."""

from __future__ import annotations

from .models import new_task
from .utils import dedupe


class SubmitHandler:
    """Accepts a batch of new tasks and stores them."""

    def __init__(self, store):
        self._store = store

    def __call__(self, payload):
        specs = payload.get("tasks", [])
        ids = []
        for spec in dedupe(specs):
            task = new_task(spec["id"], spec["name"], **spec.get("args", {}))
            self._store.put(task)
            ids.append(task.task_id)
        return {"accepted": ids}


class StatusHandler:
    """Reports the status and result of a single task."""

    def __init__(self, store):
        self._store = store

    def __call__(self, payload):
        task = self._store.get(payload["id"])
        result = self._store.result(task.task_id)
        return {"id": task.task_id, "status": task.status,
                "result": None if result is None else result.value}


class CancelHandler:
    """Marks a task as cancelled."""

    def __init__(self, store):
        self._store = store

    def __call__(self, payload):
        self._store.set_status(payload["id"], "cancelled")
        return {"id": payload["id"], "status": "cancelled"}
