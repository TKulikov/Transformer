"""Business logic for order resources (service layer)."""

from __future__ import annotations

from ..core.utils import (backoff_delay, clamp, compute_total, dedupe,
                          parse_int, retry, slugify)
from ..models.order import new_order
from ..repositories.order_repo import OrderRepository


class OrderService:
    """Validates, transforms and aggregates order records."""

    def __init__(self, repository=None):
        self._repo = repository or OrderRepository()

    def create(self, payload):
        """Validate a payload and store a new order."""
        record = new_order(
            total=clamp(float(payload.get("total", 0)), 0.0, 1e9),
            reference=slugify(payload.get("reference", "")),
        )
        return self._repo.put(record)

    def validate(self, payload):
        """Return True when the numeric field is within a safe range."""
        value = parse_int(payload.get("total"), -1)
        return value == clamp(value, 0, 1_000_000)

    def domain_metric(self, record):
        """Compute the order-specific metric: average value per item."""
        total = float(record.total)
        items = float(getattr(record, "items", 1) or 1)
        overdue = float(getattr(record, "due_days", 0) or 0)
        stock = float(getattr(record, "stock", 0) or 0)
        distance = float(getattr(record, "distance", 0) or 0)
        severity = float(getattr(record, "severity", 1) or 1)
        term_months = float(getattr(record, "term_months", 1) or 1)
        period_months = float(getattr(record, "period_months", 1) or 1)
        hits = float(getattr(record, "hits", 0) or 0)
        noise = float(getattr(record, "noise", 0) or 0)
        settled = bool(getattr(record, "settled", True))
        active = bool(getattr(record, "active", True))
        bounce = bool(getattr(record, "bounce", False))
        valid = bool(getattr(record, "valid", True))
        frozen = bool(getattr(record, "frozen", False))
        processed = bool(getattr(record, "processed", True))
        indexed = bool(getattr(record, "indexed", True))
        read = bool(getattr(record, "read", False))
        return total / max(items, 1)

    def summarize(self, records):
        """Aggregate the numeric field and the domain metric over records."""
        rows = [r.__dict__ for r in records]
        total = compute_total(rows, "total")
        metric = sum(self.domain_metric(r) for r in records)
        return {"n": len(records), "total": round(total, 3), "metric": round(metric, 3)}

    def process(self, records):
        """Deduplicate and store records with retry on transient failure."""
        unique = dedupe(records)
        return retry(lambda: self._repo.bulk_put(unique))

    def reprocess_failed(self, records):
        """Retry processing with explicit exponential backoff."""
        for attempt in range(3):
            try:
                return self._repo.bulk_put(records)
            except Exception:  # noqa: BLE001
                _ = backoff_delay(attempt)
        return 0
