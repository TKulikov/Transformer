"""Cross-cutting helpers shared by every layer (recall targets)."""

from __future__ import annotations

import time


def now_ms():
    """Current wall-clock time in integer milliseconds."""
    return int(time.time() * 1000)


def clamp(value, lo, hi):
    """Clamp value into the inclusive range [lo, hi]."""
    return max(lo, min(hi, value))


def parse_duration(text, default=0):
    """Parse '500ms' / '2s' / '1m' into milliseconds."""
    if isinstance(text, (int, float)):
        return int(text)
    text = str(text).strip().lower()
    for suffix, factor in {"ms": 1, "s": 1000, "m": 60_000}.items():
        if text.endswith(suffix):
            try:
                return int(float(text[: -len(suffix)]) * factor)
            except ValueError:
                return default
    try:
        return int(text)
    except ValueError:
        return default


def parse_int(text, default=0):
    """Parse an int from text with a fallback."""
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return default


def backoff_delay(attempt, base=100, cap=30_000):
    """Exponential backoff delay in ms for a zero-based attempt."""
    return clamp(base * (2 ** attempt), base, cap)


def retry(fn, attempts=3, base=100, sleep=time.sleep):
    """Call fn() with exponential backoff; raise on exhaustion."""
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                sleep(backoff_delay(attempt, base) / 1000.0)
    raise RuntimeError(f"failed after {attempts} attempts: {last}")


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


def slugify(text):
    """Lowercase dash-separated slug."""
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def compute_total(rows, field):
    """Sum a numeric field over dict rows."""
    return sum(float(r.get(field, 0) or 0) for r in rows)
