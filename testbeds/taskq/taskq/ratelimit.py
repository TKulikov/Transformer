"""Sliding-window rate limiting."""

from __future__ import annotations

from .utils import now_ms


class RateLimiter:
    """Limits calls per key within a sliding time window."""

    def __init__(self, limit=100, window_ms=60_000):
        self._limit = limit
        self._window = window_ms
        self._events = {}

    def allow(self, key):
        """Return True if the key is within its rate budget."""
        now = now_ms()
        events = [t for t in self._events.get(key, []) if now - t < self._window]
        if len(events) >= self._limit:
            self._events[key] = events
            return False
        events.append(now)
        self._events[key] = events
        return True

    def reset(self, key):
        """Clear the recorded events for a key."""
        self._events.pop(key, None)
