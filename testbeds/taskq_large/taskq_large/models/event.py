"""Domain model for event resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Event:
    """A single event record."""

    payload_size: float = 0.0
    kind: str = ''
    source: str = 'app'
    processed: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_event(**kwargs):
    """Construct a event from keyword fields."""
    return Event(**kwargs)
