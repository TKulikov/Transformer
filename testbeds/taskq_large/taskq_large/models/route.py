"""Domain model for route resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Route:
    """A single route record."""

    latency: float = 0.0
    path: str = ''
    method: str = 'GET'
    hits: int = 0
    created_ms: int = field(default_factory=now_ms)


def new_route(**kwargs):
    """Construct a route from keyword fields."""
    return Route(**kwargs)
