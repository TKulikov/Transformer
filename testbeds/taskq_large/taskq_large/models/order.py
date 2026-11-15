"""Domain model for order resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Order:
    """A single order record."""

    total: float = 0.0
    reference: str = ''
    items: int = 1
    status: str = 'new'
    created_ms: int = field(default_factory=now_ms)


def new_order(**kwargs):
    """Construct a order from keyword fields."""
    return Order(**kwargs)
