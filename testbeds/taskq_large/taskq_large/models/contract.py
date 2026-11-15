"""Domain model for contract resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Contract:
    """A single contract record."""

    value: float = 0.0
    party: str = ''
    term_months: int = 12
    signed: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_contract(**kwargs):
    """Construct a contract from keyword fields."""
    return Contract(**kwargs)
