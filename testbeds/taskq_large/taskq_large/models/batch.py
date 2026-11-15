"""Domain model for batch resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Batch:
    """A single batch record."""

    size: float = 0.0
    kind: str = ''
    checksum: str = ''
    valid: bool = True
    created_ms: int = field(default_factory=now_ms)


def new_batch(**kwargs):
    """Construct a batch from keyword fields."""
    return Batch(**kwargs)
