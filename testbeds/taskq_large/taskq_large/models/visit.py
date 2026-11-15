"""Domain model for visit resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Visit:
    """A single visit record."""

    duration: float = 0.0
    page: str = ''
    referrer: str = 'direct'
    bounce: bool = False
    created_ms: int = field(default_factory=now_ms)


def new_visit(**kwargs):
    """Construct a visit from keyword fields."""
    return Visit(**kwargs)
