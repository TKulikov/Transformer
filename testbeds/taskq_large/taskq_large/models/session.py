"""Domain model for session resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Session:
    """A single session record."""

    duration: float = 0.0
    token: str = ''
    device: str = 'web'
    active: bool = True
    created_ms: int = field(default_factory=now_ms)


def new_session(**kwargs):
    """Construct a session from keyword fields."""
    return Session(**kwargs)
