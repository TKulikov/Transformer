"""Domain model for shipment resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Shipment:
    """A single shipment record."""

    weight: float = 0.0
    destination: str = ''
    carrier: str = 'local'
    distance: int = 10
    created_ms: int = field(default_factory=now_ms)


def new_shipment(**kwargs):
    """Construct a shipment from keyword fields."""
    return Shipment(**kwargs)
