"""Domain model for ticket resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Ticket:
    """A single ticket record."""

    priority: float = 0.0
    title: str = ''
    assignee: str = 'unassigned'
    severity: int = 1
    created_ms: int = field(default_factory=now_ms)


def new_ticket(**kwargs):
    """Construct a ticket from keyword fields."""
    return Ticket(**kwargs)
