"""Lightweight counters and gauges for observability."""

from __future__ import annotations

from .utils import now_ms


class Metrics:
    """Accumulates counters and timing samples."""

    def __init__(self):
        self._counters = {}
        self._samples = {}

    def incr(self, name, amount=1):
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + amount
        return self._counters[name]

    def observe(self, name, value):
        """Record a timing/value sample."""
        self._samples.setdefault(name, []).append(value)

    def summary(self):
        """A serializable summary of counters and sample means."""
        means = {k: sum(v) / len(v) for k, v in self._samples.items() if v}
        return {"counters": dict(self._counters), "means": means, "ts": now_ms()}
