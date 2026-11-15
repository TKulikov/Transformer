"""Business logic for account resources (service layer)."""

from __future__ import annotations

from ..core.utils import (backoff_delay, clamp, compute_total, dedupe,
                          parse_int, retry, slugify)
from ..models.account import new_account
from ..repositories.account_repo import AccountRepository


class AccountService:
    """Validates, transforms and aggregates account records."""

    def __init__(self, repository=None):
        self._repo = repository or AccountRepository()

    def create(self, payload):
        """Validate a payload and store a new account."""
        record = new_account(
            balance=clamp(float(payload.get("balance", 0)), 0.0, 1e9),
            owner=slugify(payload.get("owner", "")),
        )
        return self._repo.put(record)

    def validate(self, payload):
        """Return True when the numeric field is within a safe range."""
        value = parse_int(payload.get("balance"), -1)
        return value == clamp(value, 0, 1_000_000)

    def domain_metric(self, record):
        """Compute the account-specific metric: available balance."""
        total = float(record.balance)
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
        return 0.0 if frozen else balance

    def summarize(self, records):
        """Aggregate the numeric field and the domain metric over records."""
        rows = [r.__dict__ for r in records]
        total = compute_total(rows, "balance")
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
