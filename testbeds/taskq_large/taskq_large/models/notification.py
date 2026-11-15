"""Domain model for notification resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Notification:
    """A single notification record."""

    priority: float = 0.0
    channel: str = ''
    message: str = ''
    read: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_notification(**kwargs):
    """Construct a notification from keyword fields."""
    return Notification(**kwargs)
