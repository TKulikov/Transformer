"""Domain model for signal resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Signal:
    """A single signal record."""

    strength: float = 0.0
    source: str = ''
    frequency: int = 1
    noise: int = 0
    created_ms: int = field(default_factory=now_ms)


def new_signal(**kwargs):
    """Construct a signal from keyword fields."""
    return Signal(**kwargs)
