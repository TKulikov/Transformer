"""Domain model for account resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Account:
    """A single account record."""

    balance: float = 0.0
    owner: str = ''
    tier: str = 'free'
    frozen: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_account(**kwargs):
    """Construct a account from keyword fields."""
    return Account(**kwargs)
