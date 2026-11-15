"""Request handler for metric resources (API layer)."""

from __future__ import annotations

from ..core.utils import parse_duration
from ..services.metric_service import MetricService


class MetricHandler:
    """Routes metric CRUD requests to the service."""

    def __init__(self, service=None):
        self._service = service or MetricService()

    def create(self, payload):
        """Create a metric and return its key."""
        record = self._service.create(payload)
        return {"created": record.__dict__}

    def list(self, payload):
        """List metric records, optionally filtered."""
        field = payload.get("filter_field")
        if field:
            rows = self._service._repo.filter_by(field, payload.get("filter_value"))
        else:
            rows = self._service._repo.list()
        return {"items": [r.__dict__ for r in rows]}

    def summary(self, payload):
        """Return an aggregate summary of metric records."""
        return self._service.summarize(self._service._repo.list())

    def timeout_ms(self, payload):
        """Parse a request timeout into milliseconds."""
        return parse_duration(payload.get("timeout", "5s"))
