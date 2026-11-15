"""Domain model for invoice resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Invoice:
    """A single invoice record."""

    amount: float = 0.0
    number: str = ''
    due_days: int = 30
    paid: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_invoice(**kwargs):
    """Construct a invoice from keyword fields."""
    return Invoice(**kwargs)
