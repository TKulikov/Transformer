"""Domain model for product resources."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.utils import now_ms


@dataclass
class Product:
    """A single product record."""

    price: float = 0.0
    sku: str = ''
    stock: int = 0
    category: str = 'general'
    created_ms: int = field(default_factory=now_ms)


def new_product(**kwargs):
    """Construct a product from keyword fields."""
    return Product(**kwargs)
