"""Domain model for subscription resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Subscription:
    """A single subscription record."""

    price: float = 0.0
    plan: str = ''
    renews: bool = True
    period_months: int = 1
    created_ms: int = field(default_factory=now_ms)


def new_subscription(**kwargs):
    """Construct a subscription from keyword fields."""
    return Subscription(**kwargs)
