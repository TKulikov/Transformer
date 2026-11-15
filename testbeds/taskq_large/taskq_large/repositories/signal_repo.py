"""Repository for signal records (persistence layer)."""

from __future__ import annotations

from ..core.utils import chunked, dedupe
from ..models.signal import Signal


class SignalRepository:
    """In-memory store of signal records with batch helpers."""

    def __init__(self):
        self._rows = {}

    def put(self, record):
        """Insert or replace a record keyed by its source."""
        self._rows[record.source] = record
        return record

    def get(self, key):
        """Return a record by source or None."""
        return self._rows.get(key)

    def delete(self, key):
        """Remove a record by source."""
        return self._rows.pop(key, None)

    def list(self):
        """All records in insertion order."""
        return list(self._rows.values())

    def keys(self):
        """Deduplicated record keys."""
        return dedupe(list(self._rows.keys()))

    def bulk_put(self, records):
        """Insert records in fixed-size batches."""
        count = 0
        for batch in chunked(records, 64):
            for record in batch:
                self.put(record)
                count += 1
        return count

    def filter_by(self, field, value):
        """Records whose field equals value."""
        return [r for r in self._rows.values() if getattr(r, field, None) == value]
