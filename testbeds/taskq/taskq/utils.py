"""Shared helpers used across the client and the server.

These are the cross-cutting utilities that many modules call; they are the
natural targets for recall-from-memory experiments.
"""

from __future__ import annotations

import time


def now_ms():
    """Current wall-clock time in integer milliseconds."""
    return int(time.time() * 1000)


def clamp(value, lo, hi):
    """Clamp value into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def parse_duration(text, default=0):
    """Parse a duration like '500ms', '2s', '1m' into milliseconds."""
    if isinstance(text, (int, float)):
        return int(text)
    text = str(text).strip().lower()
    units = {"ms": 1, "s": 1000, "m": 60_000}
    for suffix, factor in units.items():
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * factor)
            except ValueError:
                return default
    try:
        return int(text)
    except ValueError:
        return default


def backoff_delay(attempt, base=100, cap=30_000):
    """Exponential backoff delay in ms for a zero-based attempt index."""
    return clamp(base * (2 ** attempt), base, cap)


def retry(fn, attempts=3, base=100, sleep=time.sleep):
    """Call fn() with exponential backoff; raise RetryExhausted on failure."""
    from .errors import RetryExhausted

    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - deliberate broad retry
            last = exc
            if attempt < attempts - 1:
                sleep(backoff_delay(attempt, base) / 1000.0)
    raise RetryExhausted(attempts, last)


def chunked(items, size):
    """Yield successive fixed-size chunks from a sequence."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def dedupe(items):
    """Stable deduplication preserving first-occurrence order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
