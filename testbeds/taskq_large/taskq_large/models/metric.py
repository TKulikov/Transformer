"""Domain model for metric resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Metric:
    """A single metric record."""

    value: float = 0.0
    name: str = ''
    unit: str = 'count'
    tags: str = ''
    created_ms: int = field(default_factory=now_ms)


def new_metric(**kwargs):
    """Construct a metric from keyword fields."""
    return Metric(**kwargs)
