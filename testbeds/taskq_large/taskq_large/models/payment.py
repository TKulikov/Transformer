"""Domain model for payment resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Payment:
    """A single payment record."""

    amount: float = 0.0
    method: str = ''
    currency: str = 'USD'
    settled: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_payment(**kwargs):
    """Construct a payment from keyword fields."""
    return Payment(**kwargs)
