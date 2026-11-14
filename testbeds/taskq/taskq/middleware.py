"""Cross-cutting middleware for the request pipeline."""

from __future__ import annotations

from .utils import now_ms


class Middleware:
    """Base middleware: wrap a handler call with before/after hooks."""

    def __init__(self, inner):
        self._inner = inner

    def before(self, payload):
        return payload

    def after(self, result):
        return result

    def __call__(self, payload):
        return self.after(self._inner(self.before(payload)))


class LoggingMiddleware(Middleware):
    """Records timing and payload size for each call."""

    def __init__(self, inner, sink):
        super().__init__(inner)
        self._sink = sink
        self._started = 0

    def before(self, payload):
        self._started = now_ms()
        return payload

    def after(self, result):
        self._sink.append({"elapsed_ms": now_ms() - self._started, "ok": True})
        return result
