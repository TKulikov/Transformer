"""The task queue server."""

from __future__ import annotations

from .config import Config
from .handlers import CancelHandler, StatusHandler, SubmitHandler
from .router import Router
from .scheduler import Scheduler
from .storage import TaskStore
from .worker import Worker


class Server:
    """Wires storage, scheduling, routing and workers together."""

    def __init__(self, config=None, registry=None):
        self.config = config or Config()
        self.store = TaskStore()
        self.scheduler = Scheduler(self.store)
        self.registry = registry or {}
        self.worker = Worker(self.store, self.scheduler, self.registry, self.config.max_retries)
        self.router = self._build_router()

    def _build_router(self):
        router = Router()
        router.register("submit", SubmitHandler(self.store))
        router.register("status", StatusHandler(self.store))
        router.register("cancel", CancelHandler(self.store))
        return router

    def handle(self, msg):
        """Dispatch a protocol message and return the reply."""
        return self.router.dispatch(msg)

    def tick(self):
        """Run one scheduling/execution cycle; return tasks executed."""
        return self.worker.run_batch()
